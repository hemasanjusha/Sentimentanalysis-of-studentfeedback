import numpy as np
import plotly.express as px
import joblib
import streamlit as st
import pandas as pd
import torch 
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from nltk.corpus import words

nltk.download('words')
english_words = set(words.words())

# Load the model from Hugging Face
model_name = "Hemasanjusha/sentiment-analysis-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def is_gibberish(text):
    words_list = re.findall(r'\b\w+\b', text.lower())
    if not words_list:
        return True
    gibberish_count = sum(1 for word in words_list if word not in english_words)
    return gibberish_count / len(words_list) > 0.7

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
            df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(x) if isinstance(x, str) and x.strip() else '')
        else:
            st.error(f"No 'feedback_text' column found in {uploaded_file.name}")
            continue
        
        st.write("### 📊 Sentiment Analysis Results:")
        
        # Apply color coding for better readability
        def color_sentiment(val):
            color = 'green' if val == 'Positive' else 'red' if val == 'Negative' else 'blue'
            return f'color: {color}'
        
        styled_df = df.style.applymap(color_sentiment, subset=['Predicted_Sentiment'])
        st.dataframe(styled_df)

        # Visualization using Plotly
        sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        if not sentiment_counts.empty:
            st.write("### 📊 Sentiment Distribution")
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title=f'Sentiment Distribution for {uploaded_file.name}', color='Sentiment', color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'blue'})
            st.plotly_chart(fig)

        # **Download Processed File**
        output_file = f"Sentiment_Analysis_Results_{uploaded_file.name}".replace('.csv', '.xlsx')
        df.to_excel(output_file, index=False)
        with open(output_file, "rb") as file:
            st.download_button(label="📥 Download Results", data=file, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Input Section
user_input = st.text_area('Or enter your feedback directly:',"")
if st.button('Analyze Sentiment'):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        if sentiment is None:
            st.error('❗ Invalid Text. Please enter meaningful feedback.')
        else:
            st.success(f'**Predicted Sentiment:** {sentiment}')
    else:
        st.warning('⚠️ Please enter valid feedback.')
