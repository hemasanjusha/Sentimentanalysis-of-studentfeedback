import numpy as np
import pandas as pd
import torch
import streamlit as st
import re
import nltk
from nltk.corpus import words
import plotly.express as px
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download English words dataset (only once)
@st.cache_resource
def load_english_words():
    nltk.download('words')
    return set(words.words())

english_words = load_english_words()

# Load the Hugging Face model and tokenizer (cached to avoid reloading)
@st.cache_resource
def load_model():
    model_name = "Hemasanjusha/sentiment-analysis-model"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to check gibberish text
def is_gibberish(text):
    words_list = re.findall(r'\b\w+\b', text.lower())
    return not words_list or sum(1 for word in words_list if word not in english_words) / len(words_list) > 0.7

# Function to predict sentiment
def predict_sentiment(text):
    if not text.strip():  # Check for empty input
        return "No Prediction"
    if is_gibberish(text):
        return "Invalid Text"
    
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoding)
        prediction = torch.argmax(output.logits, dim=1).item()
    return {0: 'Negative', 1: 'Positive', 2: 'Neutral'}.get(prediction, "Unknown")

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1]
    try:
        df = pd.read_csv(uploaded_file) if file_type == 'csv' else pd.read_excel(uploaded_file)

        if 'feedback_text' not in df.columns:
            st.error(f"❌ No 'feedback_text' column found in {uploaded_file.name}")
            return None

        # Ensure NaN values or empty strings are marked as 'No Prediction'
        df['feedback_text'] = df['feedback_text'].astype(str).replace({'nan': '', 'NaN': '', None: ''}).fillna('').str.strip()
        df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(x) if x else 'No Prediction')

        return df
    except Exception as e:
        st.error(f"⚠️ Error processing file {uploaded_file.name}: {e}")
        return None

# Streamlit UI Setup
st.set_page_config(page_title="Student Feedback Sentiment Analyzer", layout="wide")
st.title('🎓 Student Feedback Sentiment Analyzer')
st.write('Analyze student feedback using a BERT sentiment analysis model.')

# File Upload Section
st.header("📂 Upload Files for Sentiment Analysis")
uploaded_files = st.file_uploader("Upload Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### 📁 Processing: {uploaded_file.name}")
        df = process_uploaded_file(uploaded_file)

        if df is not None:
            st.write("### 📊 Sentiment Analysis Results:")
            st.dataframe(df)

            # Visualization using Plotly
            sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            if not sentiment_counts.empty:
                st.write("### 📊 Sentiment Distribution")
                fig = px.pie(sentiment_counts, names='Sentiment', values='Count', 
                             title=f'Sentiment Distribution for {uploaded_file.name}',
                             color='Sentiment', 
                             color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "blue", "No Prediction": "gray"},
                             hover_data=['Count'], hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

            # Download Processed File
            output_file = f"Sentiment_Analysis_Results_{uploaded_file.name}".replace('.csv', '.xlsx')
            df.to_excel(output_file, index=False)
            with open(output_file, "rb") as file:
                st.download_button(label="📥 Download Results", data=file, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Input Section for Direct Text Analysis
st.header("📝 Analyze a Single Feedback")
user_input = st.text_area('Enter your feedback here:', "")

if st.button('Analyze Sentiment') and user_input.strip():
    sentiment = predict_sentiment(user_input)
    if sentiment == "Invalid Text":
        st.error('❗ Invalid Text. Please enter meaningful feedback.')
    else:
        st.success(f'**Predicted Sentiment:** {sentiment}')
