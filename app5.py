import streamlit as st
from transformers import pipeline

st.title("🎓 Student Feedback Sentiment Analyzer")

# Load the model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Hemasanjusha/sentiment-analysis-model")

sentiment_pipeline = load_model()

# UI
text = st.text_area("Enter student feedback:")

if st.button("Analyze"):
    if text.strip():
        result = sentiment_pipeline(text)[0]
        st.success(f"Sentiment: {result['label']}")
        st.info(f"Confidence: {result['score']:.2f}")
    else:
        st.warning("Please enter some feedback text.")
