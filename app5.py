import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import torch
import re
import nltk
from nltk.corpus import words
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download English words dataset
nltk.download('words')
english_words = set(words.words())

# Load the Hugging Face model and tokenizer
model_name = "Hemasanjusha/sentiment-analysis-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to check gibberish text
def is_gibberish(text):
    words_list = re.findall(r'\b\w+\b', text.lower())
    if not words_list:
        return True
    gibberish_count = sum(1 for word in words_list if word not in english_words)
    return gibberish_count / len(words_list) > 0.7

# Function to predict sentiment
def predict_sentiment(text):
    if is_gibberish(text):
        return "Invalid Text"
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoding)
        prediction = torch.argmax(output.logits, dim=1).item()
    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return sentiment_map.get(prediction, "Unknown")

# Streamlit UI
st.title('🎓 Student Feedback Sentiment Analyzer')
st.write('Analyze student feedback using a BERT sentiment analysis model.')

# File Upload Section
st.header("📂 Upload Files for Sentiment Analysis")
uploaded_files = st.file_uploader("Upload Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### 📁 Processing file: {uploaded_file.name}")

        # Read File
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        # Handling missing feedback_text values
        if 'feedback_text' in df.columns:
            df['feedback_text'] = df['feedback_text'].astype(str).fillna('').str.strip()
            df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(x) if x else 'No Feedback')
        else:
            st.error(f"❌ No 'feedback_text' column found in {uploaded_file.name}")
            continue

        # Display results
        st.write("### 📊 Sentiment Analysis Results:")
        st.dataframe(df)

        # Visualization
        sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        if not sentiment_counts.empty:
            st.write("### 📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            plt.title(f'Sentiment Distribution for {uploaded_file.name}')
            st.pyplot(fig)

        # **Download Processed File**
        output_file = f"Sentiment_Analysis_Results_{uploaded_file.name}".replace('.csv', '.xlsx')
        df.to_excel(output_file, index=False)
        with open(output_file, "rb") as file:
            st.download_button(label="📥 Download Results", data=file, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Input Section for Direct Text Analysis
st.header("📝 Analyze a Single Feedback")
user_input = st.text_area('Enter your feedback here:', "")

if st.button('Analyze Sentiment'):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        if sentiment == "Invalid Text":
            st.error('❗ Invalid Text. Please enter meaningful feedback.')
        else:
            st.success(f'**Predicted Sentiment:** {sentiment}')
    else:
        st.warning('⚠️ Please enter valid feedback.')
