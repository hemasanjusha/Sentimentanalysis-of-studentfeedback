from transformers import pipeline
import streamlit as st

# Load the model
sentiment_pipeline = pipeline("text-classification", model="Hemasanjusha/sentiment-analysis-model")

# App UI
st.title("Student Feedback Sentiment Analysis")
text = st.text_area("Enter feedback:")

if st.button("Analyze"):
    if text.strip():
        result = sentiment_pipeline(text)[0]
        st.write(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
