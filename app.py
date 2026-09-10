import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline

# 1. Page Configuration & Title
st.title("AI Text Summarizer")

# 2. Sidebar / Controls
use_sampling = st.checkbox("Use sampling for summary generation", value=True)

# 3. User Inputs
user_text = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])

# 4. Input Processing Snippet
text_to_summarize = ""

if uploaded_file is not None:
    # Read and decode the uploaded file buffer into a string
    text_to_summarize = uploaded_file.read().decode("utf-8")
elif user_text.strip():
    # Fall back to text area input if no file is uploaded
    text_to_summarize = user_text

# 5. Execution Trigger
if st.button("Summarize"):
    if not text_to_summarize.strip():
        st.warning("Please provide text before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            # Call your summarizer logic here
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            summary = summarizer(text_to_summarize, do_sample=use_sampling)
            
            st.subheader("Summary")
            st.write(summary[0]["summary_text"])