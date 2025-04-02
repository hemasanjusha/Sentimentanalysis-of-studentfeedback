import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
import torch
import re
import nltk
from nltk.corpus import words
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download words dataset
nltk.download('words')
english_words = set(words.words())

# Load the model from Hugging Face
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

# Function to predict sentiment
def predict_sentiment(text):
    if is_gibberish(text):
        return None
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoding)
        prediction = torch.argmax(output.logits, dim=1).item()
    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return sentiment_map[prediction]

# Streamlit UI
st.set_page_config(page_title='Student Feedback Sentiment Analyzer', layout='wide')
st.sidebar.title('🔍 Sentiment Analysis Options')

st.title('🎓 Student Feedback Sentiment Analyzer')
st.write('Analyze student feedback using a BERT sentiment analysis model.')

# File Upload Section
uploaded_files = st.sidebar.file_uploader("📂 Upload Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f'📁 Processing: {uploaded_file.name}')
        
        # Read File
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        if 'feedback_text' in df.columns:
            df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(x) if isinstance(x, str) and x.strip() else '')
        else:
            st.error(f"❌ No 'feedback_text' column found in {uploaded_file.name}")
            continue
        
        # Data Preview
        st.write("### 📊 Sentiment Analysis Results:")
        st.dataframe(df, use_container_width=True)

        # Visualization using Plotly
        sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Sentiment Distribution - Pie Chart")
            fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Distribution', color='Sentiment', 
                             color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.write("### 📊 Sentiment Distribution - Bar Chart")
            fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', text_auto=True, color='Sentiment', 
                             color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Download Processed File
        output_file = f"Sentiment_Analysis_Results_{uploaded_file.name}".replace('.csv', '.xlsx')
        df.to_excel(output_file, index=False)
        with open(output_file, "rb") as file:
            st.download_button(label="📥 Download Results", data=file, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Real-time Input Analysis
st.sidebar.subheader("📝 Enter Text for Sentiment Analysis")
user_input = st.sidebar.text_area("Enter your feedback:", "")
if st.sidebar.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        if sentiment is None:
            st.sidebar.error('❗ Invalid Text. Please enter meaningful feedback.')
        else:
            st.sidebar.success(f'**Predicted Sentiment:** {sentiment}')
    else:
        st.sidebar.warning('⚠️ Please enter valid feedback.')
