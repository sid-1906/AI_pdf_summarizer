import os
import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

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

# UI
st.title("📄 Free PDF Summarizer")
st.write("Upload a PDF and get a summary using Hugging Face models (no API key needed).")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    # Split into chunks to avoid model input limit
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    st.write("### Summary:")
    final_summary = ""
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        final_summary += summary + " "
        st.write(f"**Chunk {i+1} Summary:** {summary}")

    st.subheader("📌 Final Combined Summary")
    st.write(final_summary.strip())
