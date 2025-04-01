import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from nltk.corpus import words

# Download NLTK words dataset
nltk.download('words')
english_words = set(words.words())

# Load Sentiment Model from Hugging Face
model_name = "Hemasanjusha/sentiment-analysis-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to detect gibberish
def is_gibberish(text):
    words_list = re.findall(r'\b\w+\b', text.lower())
    if not words_list:
        return True
    gibberish_count = sum(1 for word in words_list if word not in english_words)
    return gibberish_count / len(words_list) > 0.7

# Function to predict sentiment with softmax and confidence threshold
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None  # No sentiment prediction for empty or invalid input
    
    if is_gibberish(text):
        return "Invalid Input"  # Handle gibberish inputs
    
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits  # Raw logits before softmax
        probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        confidence, prediction = torch.max(probabilities, dim=1)
    
    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    # If the confidence is low (below 0.5), return "Neutral"
    if confidence.item() < 0.5:
        return "Neutral"
    
    return sentiment_map.get(prediction.item(), "Unknown")

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
        
        if 'feedback_text' in df.columns:
            df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(str(x)) if isinstance(x, str) and x.strip() else None)
        else:
            st.error(f"No 'feedback_text' column found in {uploaded_file.name}")
            continue

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

# Input Section
user_input = st.text_area('Or enter your feedback directly:')

if st.button('Analyze Sentiment'):
    if user_input.strip():
        sentiment = predict_sentiment(str(user_input))  # Ensure input is a string
        if sentiment == "Invalid Input":
            st.error('❗ Invalid Text. Please enter meaningful feedback.')
        elif sentiment is None:
            st.warning('⚠️ Please enter valid feedback.')
        else:
            st.success(f'**Predicted Sentiment:** {sentiment}')
    else:
        st.warning('⚠️ Please enter valid feedback.')
