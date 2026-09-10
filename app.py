from typing import Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline


st.set_page_config(
    page_title="Text Summarization",
    page_icon="📝",
    layout="wide",
)


def clean_and_truncate(
    text: str,
    max_chars: int = 2000,
) -> str:
    """Clean and truncate text to a maximum number of characters."""

    cleaned = " ".join(text.split())

    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."

    return cleaned


def fetch_text_from_url(url: str) -> str:
    """Fetch and extract text content from a URL."""

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.content,
            "html.parser",
        )

        paragraphs = [
            paragraph.get_text(" ", strip=True)
            for paragraph in soup.find_all("p")
        ]

        text = " ".join(paragraphs)

        return clean_and_truncate(text)

    except requests.RequestException as error:
        st.error(f"Unable to fetch URL: {error}")
        return ""


class Summarizer:
    """AI text summarization service."""

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
    ):
        self.model_name = model_name
        self.pipeline = pipeline(
            "summarization",
            model=model_name
        )

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False,
    ) -> str:
        """Generate a summary from the provided text."""

        text = clean_and_truncate(text)

        if not text:
            return "No text was provided."

        result = self.pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
        )

        if result:
            return result[0]["summary_text"]

        return "No summary could be generated."


@st.cache_resource
def load_summarizer(model_name: str) -> Summarizer:
    """Load and cache the summarization model."""

    return Summarizer(model_name)


def main():
    """Run the Streamlit application."""

    st.title("📝 Text Summarization App")

    st.markdown(
        "Enter text, upload a file, or provide a URL "
        "to summarize the content."
    )

    left, right = st.columns([1, 2])

    with left:
        input_mode = st.radio(
            "Choose input type:",
            ["Text", "File", "URL"],
            key="input_type",
        )

        model_choice = st.selectbox(
            "Choose summarization model:",
            [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn",
            ],
            key="model_choice",
        )

        min_length = st.number_input(
            "Minimum summary length:",
            min_value=10,
            max_value=500,
            value=30,
            key="min_length",
        )

        max_length = st.number_input(
            "Maximum summary length:",
            min_value=50,
            max_value=1000,
            value=130,
            key="max_length",
        )

        do_sample = st.checkbox(
            "Use sampling for summary generation",
            value=False,
            key="do_sample",
        )

        raw_text: Optional[str] = None

        if input_mode == "Text":
            raw_text = st.text_area(
                "Enter text to summarize:",
                height=200,
                key="text_input",
            )

        elif input_mode == "File":
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=["txt"],
                key="file_input",
            )

            if uploaded_file is not None:
                raw_text = uploaded_file.read().decode(
                    "utf-8"
                )

        elif input_mode == "URL":
            url_input = st.text_input(
                "Enter URL to summarize:",
                key="url_input",
            )

            if url_input:
                raw_text = fetch_text_from_url(
                    url_input
                )

        summarize_btn = st.button(
            "Summarize",
            key="summarize_btn",
            type="primary",
        )

    with right:
        st.subheader("Input Text")

        source_container = st.empty()

        st.subheader("Summary Output")

        summary_container = st.empty()

    if summarize_btn:

        if not raw_text or not raw_text.strip():
            st.warning(
                "Please provide text before summarizing."
            )
            return

        if min_length >= max_length:
            st.warning(
                "Minimum length must be smaller than "
                "maximum length."
            )
            return

        with st.spinner("Loading AI model..."):
            summarizer = load_summarizer(
                model_choice
            )

        with st.spinner("Generating summary..."):
            try:
                summary = summarizer.summarize(
                    text=raw_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                )

                source_container.text_area(
                    "Input",
                    value=raw_text,
                    height=300,
                    label_visibility="collapsed",
                )

                summary_container.text_area(
                    "Summary",
                    value=summary,
                    height=300,
                    label_visibility="collapsed",
                )

            except Exception as error:
                st.error(
                    f"An error occurred: {error}"
                )


if __name__ == "__main__":
    main()