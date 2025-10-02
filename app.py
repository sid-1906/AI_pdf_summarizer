import os
import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import re
from collections import Counter

# Fix for Streamlit config permission issues
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(os.getcwd(), ".streamlit")
os.makedirs(os.environ["STREAMLIT_CONFIG_DIR"], exist_ok=True)

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return summarizer, qa_pipeline

summarizer, qa_pipeline = load_models()

# PDF text extraction
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Keyword extractor
def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\b\w+\b', text.lower())
    common = Counter(words).most_common(num_keywords)
    return [word for word, _ in common if len(word) > 3]

# --- UI Config ---
st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

# Inject Bootstrap + Custom CSS
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f7f9fc;
        }
        .card-custom {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .card-custom h4 {
            color: #007bff;
        }
        .stDownloadButton > button {
            background-color: #007bff;
            color: white !important;
            border-radius: 8px;
            font-weight: bold;
        }
        .stDownloadButton > button:hover {
            background-color: #0056b3;
            color: white !important;
        }
        .stSidebar {
            background-color: #e9f5f0;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 class='text-center text-primary'>📄 AI PDF Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='text-center lead'>Upload your PDFs and let AI summarize, analyze, and answer your questions ✨</p>",
    unsafe_allow_html=True,
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    summary_length = st.radio("📏 Summary Length:", ["Short", "Medium", "Detailed"])
    if summary_length == "Short":
        min_len, max_len = 30, 80
    elif summary_length == "Medium":
        min_len, max_len = 50, 150
    else:
        min_len, max_len = 80, 250

    st.markdown("---")
    st.info("💡 Tip: You can upload multiple PDFs at once.")

# File uploader
uploaded_files = st.file_uploader("📂 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"## 📘 {uploaded_file.name}")
        with st.spinner("📑 Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)

        # Split into chunks for summarization
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        final_summary = " ".join(summaries)

        # --- Layout with Bootstrap grid ---
        st.markdown("<div class='container'>", unsafe_allow_html=True)
        st.markdown("<div class='row'>", unsafe_allow_html=True)

        # Column 1: Summary
        st.markdown("<div class='col-md-8'>", unsafe_allow_html=True)
        st.markdown("<div class='card-custom'><h4>✨ Summary</h4>", unsafe_allow_html=True)
        st.write(final_summary)
        st.download_button(
            label="💾 Download Summary",
            data=final_summary,
            file_name=f"{uploaded_file.name}_summary.txt",
            mime="text/plain"
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Column 2: Analysis
        st.markdown("<div class='col-md-4'>", unsafe_allow_html=True)
        st.markdown("<div class='card-custom'><h4>📊 Document Analysis</h4>", unsafe_allow_html=True)
        st.write(f"**Word count:** {len(text.split())}")
        keywords = extract_keywords(text)
        st.write("**Top Keywords:**", ", ".join(keywords))
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # --- Q&A Section ---
        st.markdown("<div class='card-custom'><h4>❓ Ask Questions About This PDF</h4>", unsafe_allow_html=True)
        user_question = st.text_input(f"Type your question about **{uploaded_file.name}**:")

        if user_question:
            with st.spinner("🤔 AI is thinking..."):
                try:
                    answer = qa_pipeline(question=user_question, context=text[:3000])  # limit context
                    st.success(f"**Answer:** {answer['answer']}")
                except Exception:
                    st.error("⚠️ Sorry, couldn't process your question.")

            # Download Q&A
            st.download_button(
                label="💾 Download Q&A",
                data=f"Q: {user_question}\nA: {answer['answer']}",
                file_name=f"{uploaded_file.name}_QA.txt",
                mime="text/plain"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Optional: full extracted text in expander
        with st.expander("📖 Full Extracted Text"):
            st.write(text)



