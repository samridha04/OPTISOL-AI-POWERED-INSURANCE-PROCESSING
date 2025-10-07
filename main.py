import streamlit as st
import os
import json
import re
from datetime import datetime
from io import BytesIO
import openai
from PyPDF2 import PdfReader

# ---- Optional Word support ----
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not available. Word document processing will be limited.")

# ---- Optional OCR support ----
from PIL import Image
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---- Optional LlamaParse support ----
try:
    # pip install llama-parse llama-index-core
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

# (These LangChain imports were present in your original file; keeping them in case you extend later)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Claims Processing",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'claim_data' not in st.session_state:
    st.session_state.claim_data = {}

# Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .fraud-alert {
        background-color: #ffe8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# API Key Configuration
def setup_api_key():
    """Configure OpenAI API Key"""
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = api_key
        return True
    else:
        st.sidebar.warning("Please enter your OpenAI API key to proceed")
        return False

# --------- Text Extraction Helpers ---------

def extract_text_from_pdf(file):
    """Extract text from PDF files (fallback path if LlamaParse not used/available)"""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from Word documents (fallback path if LlamaParse not used/available)"""
    if not DOCX_AVAILABLE:
        st.error("python-docx is not properly installed. Cannot process Word documents.")
        return ""
    try:
        doc = DocxDocument(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_image(file):
    """Extract text from scanned images using OCR"""
    if not OCR_AVAILABLE:
        st.error("pytesseract is not installed. Cannot process images.")
        return ""
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return ""

def extract_text_with_llama_parse(uploaded_file):
    """
    Use LlamaParse to extract text from many file types:
    PDFs, DOCX, PPTX, XLSX, CSV, HTML, TXT, etc.
    Requires llama-parse and LLAMA_CLOUD_API_KEY.
    Returns:
      - string (text) on success
      - "" on failure
      - None if LlamaParse isn't available/configured
    """
    if not LLAMAPARSE_AVAILABLE or not os.environ.get("LLAMA_CLOUD_API_KEY"):
        return None  # not available/configured

    try:
        # Persist the uploaded file to a temp path (LlamaParse expects paths)
        suffix = "." + uploaded_file.name.split(".")[-1]
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type="text",      # "text" | "markdown" | "json"
            max_timeout=600,         # generous timeout for large docs
        )

        docs = parser.load_data([tmp_path])  # list of Documents
        parsed_text = "\n".join([d.text for d in docs if getattr(d, "text", "")])

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return parsed_text.strip() if parsed_text and parsed_text.strip() else ""
    except Exception as e:
        st.warning(f"LlamaParse failed on {uploaded_file.name}: {e}")
        return ""  # attempted but failed

def process_uploaded_file(file):
    """Process any type of uploaded file with best-available extractor.

    Order of preference:
      - Images → pytesseract OCR (if available)
      - Non-images → LlamaParse (if available + key)
      - Fallbacks → native PDF/DOCX handlers
    """
    file_type = file.name.split('.')[-1].lower()

    # 1) Images → OCR
    if file_type in ['png', 'jpg', 'jpeg', 'tiff']:
        return extract_text_from_image(file)

    # 2) Try LlamaParse for ANY non-image file (pdf/docx/pptx/xlsx/csv/html/txt/…)
    parsed = extract_text_with_llama_parse(file)
    if parsed:                     # success
        return parsed
    elif parsed is None:
        # LlamaParse not available → continue to fallbacks
        pass
    else:
        # LlamaParse attempted but returned empty/failed → try fallbacks
        pass

    # 3) Fallbacks (your existing extractors for PDF/DOCX)
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type in ['docx', 'doc']:
        return extract_text_from_docx(file)

    # 4) Last resort: unsupported
    st.warning(f"Unsupported file type or no extractor available for: {file_type}")
    return ""

# --------- LLM Utilities ---------

# Document Classification
def classify_document(text):
    """Classify document type using AI"""
    prompt = """Analyze the following document text and classify it into one of these categories:
    - CLAIM_FORM: Standard insurance claim form
    - MEDICAL_REPORT: Medical diagnosis, treatment records, lab results
    - POLICY_DOCUMENT: Insurance policy terms and conditions
    - BILL_INVOICE: Medical bills, invoices, receipts
    - SUPPORTING_DOCUMENT: Other supporting documents
    
    Document Text:
    {text}
    
    Return only the category name."""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt.format(text=text[:2000])}],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return "UNKNOWN"

# Information Extraction
def extract_claim_information(text, doc_type):
    """Extract key information from documents using AI"""
    prompts = {
        "CLAIM_FORM": """Extract the following information from this claim form:
        - Claim Number
        - Policy Holder Name
        - Policy Number
        - Date of Incident
        - Type of Claim (Medical, Accident, etc.)
        - Claim Amount
        
        Return as JSON format.""",
        
        "MEDICAL_REPORT": """Extract the following information from this medical report:
        - Patient Name
        - Diagnosis/ICD Codes
        - Treatment Details
        - Medical Facility Name
        - Doctor Name
        - Date of Treatment
        - Prescribed Medications
        
        Return as JSON format.""",
        
        "BILL_INVOICE": """Extract the following information from this bill/invoice:
        - Bill Number
        - Medical Facility
        - Date of Service
        - Total Amount
        - Itemized Charges (if available)
        - Disease/Treatment mentioned
        
        Return as JSON format.""",
        
        "POLICY_DOCUMENT": """Extract the following information from this policy document:
        - Policy Number
        - Coverage Amount
        - Exclusions
        - Covered Conditions
        - Deductible Amount
        
        Return as JSON format."""
    }
    
    prompt = prompts.get(doc_type, "Extract all key information from this document as JSON.")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from insurance documents. Always return valid JSON."},
                {"role": "user", "content": f"{prompt}\n\nDocument:\n{text[:3000]}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        extracted_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        extracted_text = re.sub(r'^```json\s*', '', extracted_text)
        extracted_text = re.sub(r'^```\s*', '', extracted_text)
        extracted_text = re.sub(r'\s*```$', '', extracted_text)
        
        # Try to parse as JSON
        try:
            return json.loads(extracted_text)
        except json.JSONDecodeError:
            return {"raw_extraction": extracted_text}
    except Exception as e:
        return {"error": str(e)}

# Fraud Detection Scorecard
def calculate_fraud_score(claim_data, documents_info):
    """Calculate fraud risk score based on multiple indicators"""
    fraud_score = 0
    risk_factors = []
    
    # Factor 1: Claim amount vs typical amounts (0-25 points)
    try:
        claim_amount = float(claim_data.get('claim_amount', 0))
        if claim_amount > 100000:
            fraud_score += 25
            risk_factors.append("Unusually high claim amount (>$100,000)")
        elif claim_amount > 50000:
            fraud_score += 15
            risk_factors.append("High claim amount (>$50,000)")
    except:
        pass
    
    # Factor 2: Missing documents (0-20 points)
    required_docs = ['CLAIM_FORM', 'MEDICAL_REPORT', 'BILL_INVOICE']
    doc_types = [doc['type'] for doc in documents_info]
    missing_docs = [doc for doc in required_docs if doc not in doc_types]
    if missing_docs:
        fraud_score += len(missing_docs) * 10
        risk_factors.append(f"Missing required documents: {', '.join(missing_docs)}")
    
    # Factor 3: Inconsistent information (0-25 points)
    names = []
    for doc in documents_info:
        info = doc.get('extracted_info', {})
        for key in info.keys():
            if 'name' in key.lower():
                name_val = str(info[key]).lower().strip()
                if name_val and len(name_val) > 2:
                    names.append(name_val)
    unique_names = set(names)
    if len(unique_names) > 1 and names:
        fraud_score += 25
        risk_factors.append(f"Inconsistent names across documents: {', '.join(unique_names)}")
    
    # Factor 4: Claim timing (0-15 points)
    try:
        incident_date = str(claim_data.get('incident_date', ''))
        claim_date = str(claim_data.get('claim_date', ''))
        if incident_date and claim_date:
            if incident_date == claim_date:
                fraud_score += 10
                risk_factors.append("Claim filed on same day as incident")
    except:
        pass
    
    # Factor 5: Document quality issues (0-15 points)
    low_quality_count = 0
    for doc in documents_info:
        text = doc.get('text', '')
        if len(text) < 100:
            low_quality_count += 1
    if low_quality_count > 0:
        fraud_score += min(15, low_quality_count * 5)
        risk_factors.append(f"Potentially incomplete or low-quality documents ({low_quality_count} found)")
    
    # Determine risk level
    if fraud_score >= 60:
        risk_level = "HIGH RISK"
        color = "🔴"
    elif fraud_score >= 35:
        risk_level = "MEDIUM RISK"
        color = "🟡"
    else:
        risk_level = "LOW RISK"
        color = "🟢"
    
    return {
        'score': fraud_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'color': color
    }

# Claims Adjuster Summary Generator
def generate_adjuster_summary(claim_data, documents_info, fraud_analysis):
    """Generate comprehensive summary for claims adjuster using LLM"""
    doc_summaries = []
    for doc in documents_info:
        doc_summaries.append(f"- {doc['filename']} ({doc['type']}): {json.dumps(doc.get('extracted_info', {}))}")
    
    prompt = f"""You are an AI assistant helping insurance claims adjusters. Generate a comprehensive, professional summary of the following insurance claim.

CLAIM INFORMATION:
{json.dumps(claim_data, indent=2)}

PROCESSED DOCUMENTS:
{chr(10).join(doc_summaries)}

FRAUD RISK ANALYSIS:
- Risk Score: {fraud_analysis['score']}/100
- Risk Level: {fraud_analysis['risk_level']}
- Risk Factors: {', '.join(fraud_analysis['risk_factors']) if fraud_analysis['risk_factors'] else 'None identified'}

Please provide a summary in the following format:

**CLAIM SUMMARY FOR ADJUSTER**

**Claim Number:** [Extract or indicate if not available]
**Policy Holder:** [Name]
**Policy Number:** [Extract or indicate if not available]
**Type of Claim:** [Type]
**Incident Description:** [Brief 2-3 sentence description of what happened based on the documents]

**Key Details:**
- Date of Incident: [Date]
- Claim Amount: [Amount]
- Medical Facility: [Name if applicable]
- Diagnosis/Treatment: [Summary]

**Document Verification Status:**
- List each document type submitted and verification status

**Potential Issues & Red Flags:**
- List any concerns or discrepancies found
- Mention fraud risk assessment

**Recommendation:**
- Provide a preliminary recommendation (Approve/Review Further/Deny) with brief reasoning

Keep the summary concise, professional, and actionable for quick decision-making."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}\n\nPlease check your OpenAI API key and internet connection."

# Exclusions Check
def check_exclusions(diagnosis, exclusion_list):
    """Check if diagnosis matches any exclusions"""
    if not diagnosis:
        return False, None
    vectorizer = TfidfVectorizer()
    diagnosis_lower = diagnosis.lower()
    for exclusion in exclusion_list:
        texts = [diagnosis_lower, exclusion.lower()]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            if similarity > 0.3:
                return True, exclusion
        except:
            continue
    return False, None

# Main App
def main():
    st.markdown('<div class="main-header">🏥 AI-Powered Insurance Claims Processing System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        if not setup_api_key():
            st.stop()

        # Llama Cloud Key (optional)
        llama_api_key = st.text_input("Llama Cloud API Key (LlamaParse)", type="password", key="llama_api_key")
        if llama_api_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key
        elif LLAMAPARSE_AVAILABLE:
            st.info("Optional: Add Llama Cloud API key to parse PPTX/XLSX/CSV/HTML/TXT and complex PDFs more reliably.")

        st.markdown("---")
        st.header("📋 System Info")
        st.info("""
        **Features:**
        - Multi-document ingestion (LlamaParse + PDF/DOCX/OCR fallbacks)
        - AI-powered classification
        - Fraud detection
        - Automated summarization
        """)
        
        st.markdown("---")
        st.header("📚 Supported Formats")
        st.write("✅ PDF files")
        if DOCX_AVAILABLE:
            st.write("✅ Word documents (.docx)")
        else:
            st.write("❌ Word documents (install python-docx)")
        if OCR_AVAILABLE:
            st.write("✅ Images (OCR)")
        else:
            st.write("❌ Images (install pytesseract)")
        if LLAMAPARSE_AVAILABLE and os.environ.get("LLAMA_CLOUD_API_KEY"):
            st.write("✅ PPTX / XLSX / CSV / HTML / TXT (via LlamaParse)")
        else:
            st.write("ℹ️ Add Llama Cloud key to enable PPTX/XLSX/CSV/HTML/TXT parsing")

    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Document Upload", 
        "🔍 Claims Processing", 
        "⚠️ Fraud Analysis",
        "📊 Adjuster Summary"
    ])
    
    # Tab 1: Document Upload
    with tab1:
        st.markdown('<div class="section-header">Multi-Document Ingestion</div>', unsafe_allow_html=True)
        st.write("Upload all documents related to the insurance claim (PDFs, Word docs, scanned images, PPTX/XLSX/CSV/HTML/TXT via LlamaParse).")
        
        # Determine accepted file types
        accepted_types = ['pdf']
        if DOCX_AVAILABLE:
            accepted_types.extend(['docx', 'doc'])
        if OCR_AVAILABLE:
            accepted_types.extend(['png', 'jpg', 'jpeg', 'tiff'])
        # Add common “other” types that LlamaParse can digest
        accepted_types.extend(['pptx', 'xlsx', 'csv', 'txt', 'html'])

        uploaded_files = st.file_uploader(
            "Upload Claim Documents",
            type=accepted_types,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    processed_docs = []
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        # Extract text (prefers LlamaParse where available)
                        text = process_uploaded_file(file)
                        
                        if text and len(text.strip()) > 0:
                            # Classify document
                            doc_type = classify_document(text)
                            
                            # Extract information
                            extracted_info = extract_claim_information(text, doc_type)
                            
                            processed_docs.append({
                                'filename': file.name,
                                'type': doc_type,
                                'text': text,
                                'extracted_info': extracted_info
                            })
                            
                            st.success(f"✅ Processed: {file.name}")
                        else:
                            st.warning(f"⚠️ Could not extract text from: {file.name}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.session_state.processed_documents = processed_docs
                    
                    if processed_docs:
                        st.success(f"✅ Successfully processed {len(processed_docs)} documents!")
                        
                        # Display processed documents
                        st.markdown("### 📋 Processed Documents")
                        for doc in processed_docs:
                            with st.expander(f"📄 {doc['filename']} - {doc['type']}"):
                                st.json(doc['extracted_info'])
                                with st.expander("View extracted text"):
                                    preview = doc['text'] if len(doc['text']) <= 1000 else doc['text'][:1000] + "..."
                                    st.text_area("Text content", preview, height=200, key=f"text_{doc['filename']}")
                    else:
                        st.error("❌ No documents could be processed successfully.")
    
    # Tab 2: Claims Processing
    with tab2:
        st.markdown('<div class="section-header">Intelligent Claims Processing</div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_documents:
            st.warning("⚠️ Please upload and process documents first in the 'Document Upload' tab.")
        else:
            st.write("Enter additional claim information:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                claim_number = st.text_input("Claim Number", key="claim_num")
                policy_holder = st.text_input("Policy Holder Name", key="policy_holder")
                policy_number = st.text_input("Policy Number", key="policy_num")
                claim_type = st.selectbox(
                    "Type of Claim",
                    ["Medical", "Accident", "Disability", "Critical Illness", "Other"]
                )
            
            with col2:
                incident_date = st.date_input("Date of Incident", key="incident_date")
                claim_date = st.date_input("Claim Filing Date", value=datetime.now(), key="claim_date")
                claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, step=100.0)
                medical_facility = st.text_input("Medical Facility", key="med_facility")
            
            description = st.text_area("Incident Description", key="description")
            
            if st.button("Validate Claim", type="primary"):
                # Store claim data
                st.session_state.claim_data = {
                    'claim_number': claim_number,
                    'policy_holder': policy_holder,
                    'policy_number': policy_number,
                    'claim_type': claim_type,
                    'incident_date': str(incident_date),
                    'claim_date': str(claim_date),
                    'claim_amount': claim_amount,
                    'medical_facility': medical_facility,
                    'description': description
                }
                
                # Validation logic
                exclusion_list = [
                    "HIV/AIDS", "Parkinson's disease", "Alzheimer's disease",
                    "pregnancy", "substance abuse", "self-inflicted injuries",
                    "sexually transmitted diseases", "pre-existing conditions"
                ]
                
                # Check for exclusions
                diagnosis = ""
                for doc in st.session_state.processed_documents:
                    if doc['type'] == 'MEDICAL_REPORT':
                        info = doc.get('extracted_info', {})
                        # Look for diagnosis in various keys
                        for key, value in info.items():
                            if 'diagnosis' in key.lower() or 'icd' in key.lower():
                                diagnosis = str(value)
                                break
                
                if diagnosis:
                    is_excluded, matched_exclusion = check_exclusions(diagnosis, exclusion_list)
                    if is_excluded:
                        st.markdown(f"""
                        <div class="fraud-alert">
                        <h3>❌ Claim Rejected</h3>
                        <p><strong>Reason:</strong> The diagnosis "{diagnosis}" matches the exclusion: "{matched_exclusion}"</p>
                        <p>This condition is not covered under the policy terms.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                        <h3>✅ Initial Validation Passed</h3>
                        <p>No exclusions detected. Proceed to fraud analysis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No diagnosis information found in medical reports. Manual review recommended.")
    
    # Tab 3: Fraud Analysis
    with tab3:
        st.markdown('<div class="section-header">Risk & Fraud Detection</div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_documents or not st.session_state.claim_data:
            st.warning("⚠️ Please complete document processing and claims validation first.")
        else:
            if st.button("Run Fraud Analysis", type="primary"):
                with st.spinner("Analyzing claim for fraud indicators..."):
                    fraud_analysis = calculate_fraud_score(
                        st.session_state.claim_data,
                        st.session_state.processed_documents
                    )
                    
                    st.session_state.fraud_analysis = fraud_analysis
                    
                    # Display fraud score
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Fraud Score", f"{fraud_analysis['score']}/100")
                    
                    with col2:
                        st.metric("Risk Level", fraud_analysis['risk_level'])
                    
                    with col3:
                        st.markdown(f"<h1 style='text-align: center'>{fraud_analysis['color']}</h1>", 
                                  unsafe_allow_html=True)
                    
                    # Display risk factors
                    if fraud_analysis['risk_factors']:
                        st.markdown("### ⚠️ Identified Risk Factors:")
                        for factor in fraud_analysis['risk_factors']:
                            st.warning(f"• {factor}")
                    else:
                        st.success("✅ No significant risk factors identified")
                    
                    # Detailed Scorecard
                    with st.expander("📊 View Detailed Fraud Scorecard"):
                        st.markdown("""
                        **Scoring Criteria:**
                        - **Claim Amount** (0-25 points): Flags unusually high claims
                        - **Missing Documents** (0-20 points): Identifies incomplete submissions
                        - **Inconsistent Information** (0-25 points): Detects name mismatches
                        - **Claim Timing** (0-15 points): Flags suspicious filing patterns
                        - **Document Quality** (0-15 points): Identifies low-quality submissions
                        
                        **Risk Levels:**
                        - 🟢 LOW RISK: 0-34 points
                        - 🟡 MEDIUM RISK: 35-59 points
                        - 🔴 HIGH RISK: 60-100 points
                        """)
                    
                    # Recommendation
                    if fraud_analysis['score'] >= 60:
                        st.markdown("""
                        <div class="fraud-alert">
                        <h3>🚨 HIGH RISK - Immediate Action Required</h3>
                        <p>This claim should be escalated to fraud investigation team for detailed review.</p>
                        <p><strong>Next Steps:</strong> Assign to fraud analyst, request additional documentation, conduct background verification.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif fraud_analysis['score'] >= 35:
                        st.markdown("""
                        <div class="info-box">
                        <h3>⚠️ MEDIUM RISK - Enhanced Review Recommended</h3>
                        <p>Recommend additional verification steps before approval.</p>
                        <p><strong>Next Steps:</strong> Request clarifications, verify medical facility, cross-check policy details.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                        <h3>✅ LOW RISK - Standard Processing</h3>
                        <p>Claim can proceed through standard approval workflow.</p>
                        <p><strong>Next Steps:</strong> Generate adjuster summary and proceed to approval.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Tab 4: Adjuster Summary
    with tab4:
        st.markdown('<div class="section-header">Claims Adjuster Summary</div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_documents or not st.session_state.claim_data:
            st.warning("⚠️ Please complete all previous steps first.")
        else:
            st.info("This AI-generated summary provides claims adjusters with a quick overview of the claim, including key details and potential issues.")
            
            if st.button("Generate Adjuster Summary", type="primary"):
                with st.spinner("Generating comprehensive summary using LLM..."):
                    fraud_analysis = st.session_state.get('fraud_analysis', {
                        'score': 0,
                        'risk_level': 'UNKNOWN',
                        'risk_factors': []
                    })
                    
                    summary = generate_adjuster_summary(
                        st.session_state.claim_data,
                        st.session_state.processed_documents,
                        fraud_analysis
                    )
                    
                    st.markdown(summary)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Summary (TXT)",
                            data=summary,
                            file_name=f"claim_summary_{st.session_state.claim_data.get('claim_number', 'unknown')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Create structured JSON export
                        export_data = {
                            'claim_data': st.session_state.claim_data,
                            'documents': [
                                {
                                    'filename': doc['filename'],
                                    'type': doc['type'],
                                    'extracted_info': doc['extracted_info']
                                } 
                                for doc in st.session_state.processed_documents
                            ],
                            'fraud_analysis': fraud_analysis,
                            'summary': summary
                        }
                        
                        st.download_button(
                            label="📥 Download Full Report (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"claim_report_{st.session_state.claim_data.get('claim_number', 'unknown')}.json",
                            mime="application/json"
                        )

if __name__ == "__main__":
    main()
