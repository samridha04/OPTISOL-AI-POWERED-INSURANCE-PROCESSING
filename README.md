### AI-Powered Insurance Claims Processing 

An end-to-end Intelligent Document Processing (IDP) app that ingests mixed claim documents (PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, images), classifies them with an LLM, extracts structured fields, checks exclusions, runs a rule-based fraud scorecard, and generates a concise claims-adjuster summary. Supports LlamaParse for robust multi-format parsing and OCR for scanned images.

### Features 

1. Multi-Document Ingestion
2. Intelligent Document Processing
3. Exclusion & Validation
4. Risk & Fraud Detection
5. Adjuster Support


### Tech Stack

Frontend / App: Streamlit
LLM: OpenAI API (gpt-3.5-turbo by default)
Parsing
1. LlamaParse (optional, recommended): multi-format extraction
2. PyPDF2 (PDF fallback), python-docx (DOCX fallback)
3. pytesseract + PIL (OCR for images)
NLP / Similarity: scikit-learn (TF-IDF, cosine)

### Requirements

1. Python 3.9+ recommended

2. Packages: streamlit, openai, PyPDF2, pillow, scikit-learn
Optional (enable features): 
    1. python-docx (DOCX fallback) 
    2. pytesseract (OCR) + Tesseract binary
    3. llama-parse and llama-index-core (robust multi-format parsing)
3. Tesseract OCR - Install the Tesseract binary via your OS package manager

### Configuration

1. OpenAI API key - https://platform.openai.com/api-keys
2. Llama Cloud (LlamaParse) API key - https://cloud.llamaindex.ai/

To run the app : "streamlit run main.py"