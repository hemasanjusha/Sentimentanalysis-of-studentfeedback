import streamlit as st
import pandas as pd
from transformers import pipeline
from huggingface_hub import login
import re
import string
import nltk
from nltk.corpus import words as nltk_words
from io import BytesIO
import plotly.express as px
import os

# Download NLTK words
nltk.download('words')
english_words = set(nltk_words.words())

# Title
st.set_page_config(page_title="Student Feedback Sentiment Analyzer", layout="wide")
st.title("🎓 Student Feedback Sentiment Analyzer")

# Page setup
st.set_page_config(page_title="Student Feedback Sentiment Analyzer", layout="centered")
st.title("🎓 Student Feedback Sentiment Analyzer")

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import words

# Download nltk words corpus (used for gibberish detection)
nltk.download('words')

# Set up Streamlit page
st.set_page_config(page_title="Student Feedback Sentiment Analyzer", layout="centered")
st.title("🎓 Student Feedback Sentiment Analyzer")

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_name = "Hemasanjusha/student-feedback-sentiment-model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"❌ Failed to load sentiment model: {e}")
        return None

sentiment_pipeline = load_model()

# Function to check for gibberish input
def is_gibberish(text):
    english_vocab = set(words.words())
    tokens = text.lower().split()
    matches = [w for w in tokens if w in english_vocab]
    return len(matches) / max(len(tokens), 1) < 0.3

# Text input from user
user_input = st.text_area("✍️ Enter student feedback here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if sentiment_pipeline is None:
        st.warning("⚠️ Sentiment model not loaded.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter valid feedback.")
    elif is_gibberish(user_input):
        st.warning("⚠️ This feedback seems to be gibberish or meaningless.")
    else:
        result = sentiment_pipeline(user_input)[0]
        sentiment = result['label']
        confidence = result['score']
        st.success(f"✅ Sentiment: **{sentiment}** with **{confidence:.2%}** confidence.")

# Gibberish detection
def is_gibberish(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return True
    cleaned = re.sub(r'[\d' + string.punctuation + ']', '', text.lower())
    tokens = cleaned.split()
    if not tokens:
        return True
    valid_word_count = sum(1 for word in tokens if word in english_words or (len(word) > 3 and any(c in word for c in "aeiou")))
    return valid_word_count / len(tokens) < 0.4

# Sentiment prediction
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "No Feedback"
    if is_gibberish(text):
        return "Invalid"
    label = model(text)[0]['label']
    if "1 star" in label or "2 stars" in label:
        return "Negative"
    elif "4 stars" in label or "5 stars" in label:
        return "Positive"
    else:
        return "Neutral"

# Suggestion generator
def generate_teacher_suggestion(feedback, sentiment):
    if is_gibberish(feedback):
        return "⚠ Feedback is invalid or does not contain meaningful content."
    if sentiment == "Positive":
        return "Keep up the good work! Continue adapting to students’ needs."

    feedback = feedback.lower()
    keyword_map = {
        "real": "Incorporate more real-world examples to improve clarity.",
        "interactive": "Increase student participation through interactive sessions.",
        "boring": "Use multimedia or creative methods to make lectures engaging.",
        "fast": "Slow down the teaching pace for better understanding.",
        "slow": "Speed up a bit to maintain student interest.",
        "confusing": "Clarify complex topics using more illustrations or examples.",
        "notes": "Provide organized and detailed notes or study materials.",
        "practical": "Include more hands-on sessions or practical demonstrations.",
        "supportive": "Be available and open to student queries consistently.",
        "bias": "Ensure fair evaluation and avoid partial behavior.",
        "unclear": "Try to explain the concepts in a clearer and more structured way.",
        "doubt": "Spend more time addressing doubts during or after the class.",
        "monotone": "Use varied tone and expression to maintain student attention.",
        "lengthy": "Make the sessions more concise and focused.",
        "engaging": "Maintain this engaging teaching style in future sessions.",
        "rude": "Be more polite and respectful while interacting with students.",
        "friendly": "Maintain a friendly attitude to foster better learning.",
        "discipline": "Establish clear classroom rules to improve discipline.",
        "examples": "Use more practical examples to enhance understanding.",
        "difficult": "Break down difficult topics into smaller, digestible parts.",
        "improve": "Identify key weak areas and revise them with additional explanations.",
        "communication": "Work on communication skills for better clarity of concepts.",
        "presentation": "Improve slide designs or board work for better presentation.",
        "voice": "Use clear and audible voice during the lecture.",
        "time": "Manage class time efficiently to cover the syllabus effectively.",
        "english": "Improve English fluency for better understanding.",
        "telugu": "Avoid excessive use of Telugu if students prefer English medium.",
        "example": "Provide more relatable or daily life examples.",
        "respect": "Maintain respectful communication with students.",
        "encourage": "Encourage students to ask questions and express doubts.",
        "feedback": "Consider collecting regular feedback for continuous improvement.",
        "revision": "Add quick revisions at the end of each session.",
        "test": "Conduct frequent short tests to reinforce learning.",
        "marks": "Be transparent and fair in awarding marks.",
        "strict": "Maintain discipline without being too strict or intimidating.",
        "lenient": "Avoid being overly lenient as it may reduce student seriousness.",
        "focus": "Help students stay focused by keeping sessions engaging and on track.",
        "fun": "Add fun or gamified elements to learning where appropriate.",
        "explain": "Explain the content in a step-by-step manner for better clarity.",
        "topic": "Stick to the topic and avoid digressing during lectures.",
        "material": "Ensure study materials are comprehensive and easy to follow.",
        "doesn’t interact": "Encourage more student participation by initiating discussions and being more approachable in class."
    }

    for keyword, suggestion in keyword_map.items():
        if keyword in feedback:
            return suggestion
    return "Consider reviewing this feedback to find areas for improvement."

# Session state
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Process Feedback",
    "Teacher Dashboard",
    "Visualizations",
    "Improvement Tips",
    "Manual Feedback Entry",
    "Download Results"
])

# Filtering logic
def get_filtered_df(selected_teacher, selected_subject):
    df = st.session_state.df
    if selected_teacher != "All":
        df = df[df["TEACHER"] == selected_teacher]
    if selected_subject != "All":
        df = df[df["SUBJECT"] == selected_subject]
    return df

# Page: Upload Feedback
if page == "Upload & Process Feedback":
    st.header("📤 Upload Feedback File")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        df.columns = df.columns.str.upper()
        required = {"ROLL NUMBER", "FEEDBACK_TEXT", "TEACHER", "SUBJECT"}
        if not required.issubset(df.columns):
            st.error("File must contain: ROLL NUMBER, FEEDBACK_TEXT, TEACHER, SUBJECT.")
        else:
            df = df.drop_duplicates(subset="ROLL NUMBER")
            df["FEEDBACK_TEXT"] = df["FEEDBACK_TEXT"].fillna("")
            df["Predicted_Sentiment"] = df["FEEDBACK_TEXT"].apply(predict_sentiment)
            df["Suggestion"] = df.apply(lambda row: generate_teacher_suggestion(row["FEEDBACK_TEXT"], row["Predicted_Sentiment"]), axis=1)
            st.session_state.df = df
            st.success("✅ Feedback processed successfully!")
            st.dataframe(df, use_container_width=True)

# Page: Teacher Dashboard
elif page == "Teacher Dashboard":
    st.header("👨‍🏫 Teacher-wise Sentiment Dashboard")
    if st.session_state.df.empty:
        st.info("Please upload and process feedback first.")
    else:
        teachers = st.session_state.df["TEACHER"].unique().tolist()
        subjects = st.session_state.df["SUBJECT"].unique().tolist()
        selected_teacher = st.selectbox("Filter by Teacher", ["All"] + teachers)
        selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects)
        df_filtered = get_filtered_df(selected_teacher, selected_subject)
        st.dataframe(df_filtered, use_container_width=True)

        fig = px.pie(df_filtered, names="Predicted_Sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Page: Visualizations
elif page == "Visualizations":
    st.header("📊 Overall Sentiment Visualizations")
    df = st.session_state.df
    if df.empty:
        st.info("Please upload and process a file first.")
    else:
        sentiment_counts = df["Predicted_Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        st.plotly_chart(px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution'), use_container_width=True)
        st.plotly_chart(px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', text='Count', title='Sentiment Count'), use_container_width=True)

# Page: Improvement Tips
elif page == "Improvement Tips":
    st.header("🛠️ Suggestions to Improve")
    df = st.session_state.df
    if df.empty:
        st.info("Please upload and process a file first.")
    else:
        teachers = df["TEACHER"].unique().tolist()
        subjects = df["SUBJECT"].unique().tolist()
        selected_teacher = st.selectbox("Filter by Teacher", ["All"] + teachers, key="tip_teacher")
        selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects, key="tip_subject")
        df_filtered = get_filtered_df(selected_teacher, selected_subject)

        tips_df = df_filtered[["ROLL NUMBER", "TEACHER", "SUBJECT", "FEEDBACK_TEXT", "Predicted_Sentiment", "Suggestion"]]
        st.dataframe(tips_df, use_container_width=True)

        file_format = st.selectbox("Choose file format", ["CSV", "Excel"], key="tip_format")
        file_name = st.text_input("Enter file name:", value="filtered_suggestions", key="tip_filename")
        if file_format == "CSV":
            st.download_button("Download CSV", data=tips_df.to_csv(index=False).encode('utf-8'), file_name=f"{file_name}.csv", mime="text/csv")
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                tips_df.to_excel(writer, index=False)
            st.download_button("Download Excel", data=output.getvalue(), file_name=f"{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        suggestion_counts = tips_df["Suggestion"].value_counts().reset_index()
        suggestion_counts.columns = ["Suggestion", "Count"]
        if not suggestion_counts.empty:
            st.plotly_chart(px.pie(suggestion_counts, names='Suggestion', values='Count', title='Improvement Suggestion Distribution'), use_container_width=True)

# Page: Manual Entry
elif page == "Manual Feedback Entry":
    st.header("✍ Analyze Feedback Manually")
    feedback_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Feedback"):
        if feedback_input.strip():
            sentiment = predict_sentiment(feedback_input)
            if sentiment == "Invalid":
                st.warning("⚠️ The feedback appears to be nonsensical or too short to analyze.")
            elif sentiment == "No Feedback":
                st.warning("⚠️ Please enter some meaningful feedback.")
            else:
                suggestion = generate_teacher_suggestion(feedback_input, sentiment)
                st.success("✅ Analysis Completed")
                st.markdown(f"*Sentiment:* `{sentiment}`")
                st.markdown(f"*Suggestion:* `{suggestion}`")
        else:
            st.warning("⚠️ Please enter some feedback.")

# Page: Download Results
elif page == "Download Results":
    st.header("📥 Download Results")
    df = st.session_state.df
    if df.empty:
        st.info("Please upload and process feedback first.")
    else:
        file_format = st.selectbox("Choose file format", ["CSV", "Excel"])
        filename = st.text_input("Enter file name:", value="sentiment_feedback")
        if file_format == "CSV":
            st.download_button("Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=f"{filename}.csv", mime="text/csv")
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("Download Excel", data=output.getvalue(), file_name=f"{filename}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
