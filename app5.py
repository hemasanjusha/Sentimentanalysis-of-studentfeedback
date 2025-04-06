
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import re

# Page configuration
st.set_page_config(page_title="📊 Student Feedback Sentiment Analysis", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

model = load_model()

# Sentiment prediction function
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "No Feedback"
    
    gibberish_pattern = r'[^a-zA-Z0-9\s]'
    if re.search(gibberish_pattern, text.strip()):
        return "Invalid Text, Please enter valid text"
    
    result = model(text)[0]['label']
    if "1 star" in result or "2 stars" in result:
        return "Negative"
    elif "4 stars" in result or "5 stars" in result:
        return "Positive"
    else:
        return "Neutral"

# Enhanced custom improvement suggestions
def generate_teacher_suggestion(feedback):
    feedback = feedback.lower()

    if any(word in feedback for word in ["real time", "real-time", "real-world", "industry example", "practical example"]):
        return "Include more real-world or industry-based examples during teaching."
    
    elif any(word in feedback for word in ["interactive", "participate", "engage", "discussion", "two-way"]):
        return "Make the sessions more interactive with questions, discussions, and student participation."
    
    elif any(word in feedback for word in ["slow", "drag", "delayed", "takes too long","bad","worst","not good"]):
        return "Consider improving the pace of teaching to keep students attentive."
    
    elif any(word in feedback for word in ["fast", "too quick", "rushed", "quickly covered"]):
        return "Try to slow down and ensure everyone understands before moving on."
    
    elif any(word in feedback for word in ["practical", "hands-on", "experiment", "lab"]):
        return "Incorporate more practical sessions or demonstrations to enhance learning."
    
    elif any(word in feedback for word in ["doubt", "unclear", "confusing", "clarity", "not understood"]):
        return "Explain concepts more clearly and provide opportunities to clarify doubts."
    
    elif any(word in feedback for word in ["boring", "monotone", "dull"]):
        return "Make the lectures more engaging with interesting examples or activities."
    
    elif any(word in feedback for word in ["notes", "material", "resources", "content not provided"]):
        return "Share proper study materials and resources after the class."
    
    elif any(word in feedback for word in ["repetitive", "redundant"]):
        return "Avoid repeating the same content and focus on adding new insights."
    
    elif any(word in feedback for word in ["attentive", "listens", "cares", "supportive"]):
        return "You're doing a great job being attentive to student needs—keep it up!"

    elif any(word in feedback for word in ["fun", "interesting", "exciting"]):
        return "Continue using fun and interesting methods to keep students engaged."

    else:
        return "Continue to maintain quality and adapt based on feedback."


# Suggestion based on sentiment
def give_basic_suggestion(sentiment):
    if sentiment == "Negative":
        return "⚠ Please address the student's concerns."
    elif sentiment == "Neutral":
        return "📝 Encourage more detailed feedback."
    elif sentiment == "Positive":
        return "✅ Keep up the great work!"
    else:
        return ""


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📤 Upload & Process Feedback",
    "💡 Suggestions for Teachers",
    "🔧 Improvement Tips based on Feedback",
    "📊 Visualizations",
    "✍ Manual Feedback Entry"
])

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Page 1: Upload & process feedback
if page == "📤 Upload & Process Feedback":
    st.header("📤 Upload Your Feedback File")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        df.columns = df.columns.str.upper()

        if "ROLL NUMBER" not in df.columns or "FEEDBACK_TEXT" not in df.columns:
            st.error("Your file must contain 'ROLL NUMBER' and 'FEEDBACK_TEXT' columns.")
        else:
               df = df.drop_duplicates(subset="ROLL NUMBER", keep="first")

    # Fill NaNs with empty strings to handle in prediction
               df["FEEDBACK_TEXT"] = df["FEEDBACK_TEXT"].fillna("")

    # Predict sentiment (including for empty strings)
               df["Predicted_Sentiment"] = df["FEEDBACK_TEXT"].apply(predict_sentiment)

    # Suggestion and Improvement Tips only if there's a valid sentiment
               df["Suggestion"] = df["Predicted_Sentiment"].apply(lambda s: give_basic_suggestion(s) if s != "No Feedback" else "No Feedback Provided")
               df["Improvement_Tips"] = df["FEEDBACK_TEXT"].apply(lambda f: generate_teacher_suggestion(f) if f.strip() else "No Feedback Provided")

               st.session_state.df = df
               st.success("✅ Feedback processed successfully!")
               st.dataframe(df, use_container_width=True)

# Page 2: Suggestions for teachers
elif page == "💡 Suggestions for Teachers":
    st.header("💡 Suggestions for Teachers")
    if st.session_state.df.empty:
        st.info("ℹ Please upload and process a file first.")
    else:
        suggestions_df = st.session_state.df[["ROLL NUMBER", "FEEDBACK_TEXT", "Predicted_Sentiment", "Suggestion"]]
        st.dataframe(suggestions_df, use_container_width=True)

# Page 3: Improvement tips
elif page == "🔧 Improvement Tips based on Feedback":
    st.header("🔧 Improvement Tips for Teachers")
    if st.session_state.df.empty:
        st.info("ℹ Please upload and process a file first.")
    else:
        tips_df = st.session_state.df[["ROLL NUMBER", "FEEDBACK_TEXT", "Improvement_Tips"]]
        st.dataframe(tips_df, use_container_width=True)

# Page 4: Visualizations
elif page == "📊 Visualizations":
    st.header("📊 Sentiment Visualizations")
    if st.session_state.df.empty:
        st.info("ℹ Please upload and process a file first.")
    else:
        sentiment_counts = st.session_state.df["Predicted_Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        pie_fig = px.pie(sentiment_counts, names='Sentiment', values='Count',
                         title='Sentiment Distribution',
                         color_discrete_sequence=px.colors.sequential.RdBu)
        bar_fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                         title='Sentiment Count', text='Count')

        st.plotly_chart(pie_fig, use_container_width=True)
        st.plotly_chart(bar_fig, use_container_width=True)

# Page 5: Manual feedback entry
elif page == "✍ Manual Feedback Entry":
    st.header("✍ Enter Feedback Manually")
    feedback_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Feedback"):
        if feedback_input.strip():
            sentiment = predict_sentiment(feedback_input)
            suggestion = give_basic_suggestion(sentiment)
            improvement = generate_teacher_suggestion(feedback_input)

            st.success("✅ Analysis Completed")
            st.markdown(f"*Sentiment:* {sentiment}")
            st.markdown(f"*Suggestion:* {suggestion}")
            st.markdown(f"*Improvement Tip:* {improvement}")
        else:
            st.warning("⚠ Please enter some feedback to analyze.")
