import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import nltk
from nltk.corpus import words as nltk_words
from io import BytesIO
import re
import string
import plotly.express as px
import os
from transformers import init_empty_weights
with init_empty_weights():
    model = AutoModel.from_config(config)


# Download NLTK English words
nltk.download('words')
english_words = set(nltk_words.words())

# Set page config
st.set_page_config(page_title="Student Feedback Sentiment Analyzer", layout="centered")
st.title("🎓 Student Feedback Sentiment Analyzer")
from transformers import pipeline

model_path = "D:/sentiment-analysis-model"
pipe = pipeline("sentiment-analysis", model_path)

print(pipe("I love this product!"))

# Function to detect gibberish
def is_gibberish(text):
    if not text:
        return True
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return True
    known_words = [w for w in tokens if w in english_words]
    vowel_ratio = sum(c in 'aeiou' for c in text.lower()) / max(len(text), 1)
    return len(known_words) / len(tokens) < 0.3 or vowel_ratio < 0.2

# Sentiment prediction
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "No Feedback"
    if is_gibberish(text):
        return "Invalid"
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        if "1 star" in label or "2 stars" in label:
            return "Negative"
        elif "4 stars" in label or "5 stars" in label:
            return "Positive"
        else:
            return "Neutral"
    except Exception as e:
        return "Error"

# Generate suggestions based on sentiment
def generate_teacher_suggestion(feedback, sentiment):
    feedback_lower = feedback.lower()
    keyword_suggestions = {
        "boring": "Make the class more engaging.",
        "unclear": "Clarify explanations with examples.",
        "fast": "Slow down the teaching pace.",
        "rude": "Be more respectful to students.",
        "helpful": "Continue being supportive.",
        "interactive": "Great job involving students!"
    }
    if sentiment == "Negative":
        for keyword, suggestion in keyword_suggestions.items():
            if keyword in feedback_lower:
                return suggestion
        return "Consider seeking anonymous student input to identify improvement areas."
    elif sentiment == "Positive":
        return "Keep up the good work and continue engaging students!"
    elif sentiment == "Neutral":
        return "Consider asking for more detailed feedback to better understand student needs."
    return "No suggestion available."

# Pages
page = st.sidebar.selectbox("Select Page", ["📤 Upload Feedback File", "✍ Manual Feedback Entry"])

if page == "📤 Upload Feedback File":
    st.header("📤 Upload Feedback CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'feedback_text' column", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'feedback_text' not in df.columns:
            st.error("Uploaded file must contain a 'feedback_text' column.")
        else:
            with st.spinner("Analyzing feedback..."):
                df['Sentiment'] = df['feedback_text'].apply(predict_sentiment)
                df['Suggestion'] = df.apply(lambda row: generate_teacher_suggestion(row['feedback_text'], row['Sentiment']), axis=1)

            st.success("Analysis complete!")

            st.subheader("📊 Sentiment Distribution")
            fig = px.pie(df, names='Sentiment', title="Sentiment Breakdown")
            st.plotly_chart(fig)

            st.subheader("📄 Feedback Table")
            st.dataframe(df)

            download_option = st.selectbox("Download results as:", ["CSV", "Excel"])
            if download_option == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sentiment')
                    writer.save()
                st.download_button("Download Excel", data=output.getvalue(), file_name="sentiment_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif page == "✍ Manual Feedback Entry":
    st.header("✍ Analyze Feedback Manually")
    feedback_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Feedback"):
        if feedback_input.strip():
            sentiment = predict_sentiment(feedback_input)
            suggestion = generate_teacher_suggestion(feedback_input, sentiment)
            st.write(f"**Predicted Sentiment:** {sentiment}")
            st.write(f"**Suggested Improvement:** {suggestion}")
        else:
            st.warning("Please enter valid feedback.")
