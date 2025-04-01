import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Replace with your Hugging Face model repository name
model_name = "Hemasanjusha/sentiment-analysis-model"
# Load the model and tokenizer directly from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def predict_sentiment(text):
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
        
        # Read the file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if 'feedback_text' in df.columns:
            df['Predicted_Sentiment'] = df['feedback_text'].apply(lambda x: predict_sentiment(x) if isinstance(x, str) and x.strip() else 'Neutral')
        else:
            st.error(f"No 'feedback_text' column found in {uploaded_file.name}")
            continue

        st.write("### 📊 Sentiment Analysis Results:")
        st.dataframe(df)

        # Visualization: Sentiment Distribution
        sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        if sentiment_counts.empty:
            st.error(f"No valid data to display in the pie chart for {uploaded_file.name}.")
        else:
            st.write("### 📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            plt.title(f'Sentiment Distribution for {uploaded_file.name}')
            st.pyplot(fig)

        # *Download processed file*
        output_file = f"Sentiment_Analysis_Results_{uploaded_file.name}".replace('.xlsx', '.xlsx').replace('.csv', '.xlsx')
        df.to_excel(output_file, index=False)
        with open(output_file, "rb") as file:
            st.download_button(label="📥 Download Results", data=file, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Input Section
user_input = st.text_area('Or enter your feedback directly:', '')
if st.button('Analyze Sentiment'):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f'*Predicted Sentiment:* {sentiment}')
    else:
        st.warning('⚠ Please enter valid feedback.')

# Option to Evaluate Model
def evaluate_model(texts, labels):
    predictions = [predict_sentiment(text) for text in texts]
    st.write('### Classification Report')
    st.text(classification_report(labels, predictions, target_names=['Negative', 'Positive', 'Neutral']))

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive', 'Neutral'], yticklabels=['Negative', 'Positive', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
