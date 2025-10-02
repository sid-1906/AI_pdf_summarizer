import os
import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import re
from collections import Counter

# Fix: force Streamlit to use local config folder
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(os.getcwd(), ".streamlit")
os.makedirs(os.environ["STREAMLIT_CONFIG_DIR"], exist_ok=True)

# Load summarizer model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Simple keyword extractor
def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\b\w+\b', text.lower())
    common = Counter(words).most_common(num_keywords)
    return [word for word, _ in common if len(word) > 3]

# --- UI starts here ---
st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")

st.title("📄 PDF Summarizer App")
st.write("Upload one or more PDFs and get concise AI summaries. Powered by **Hugging Face Transformers**.")

with st.sidebar:
    st.header("⚙️ Settings")
    summary_length = st.radio("Choose summary length:", ["Short", "Medium", "Detailed"])
    if summary_length == "Short":
        min_len, max_len = 30, 80
    elif summary_length == "Medium":
        min_len, max_len = 50, 150
    else:
        min_len, max_len = 80, 250

    st.info("💡 You can upload multiple PDFs. Each will be summarized separately.")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📘 {uploaded_file.name}")
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        final_summary = " ".join(summaries)

        with st.expander("📖 Full Extracted Text"):
            st.write(text)

        st.markdown("### ✨ Summary")
        st.write(final_summary)

        # Download button
        st.download_button(
            label="💾 Download Summary",
            data=final_summary,
            file_name=f"{uploaded_file.name}_summary.txt",
            mime="text/plain"
        )

        # Extra analysis
        with st.expander("📊 Document Analysis"):
            st.write(f"**Word count:** {len(text.split())}")
            keywords = extract_keywords(text)
            st.write("**Top Keywords:**", ", ".join(keywords))

