import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import re
import string
import nltk
from nltk.corpus import words as nltk_words

# Download English word list (for gibberish detection)
nltk.download('words')
english_words = set(nltk_words.words())

# Page configuration
st.set_page_config(page_title="📊 Student Feedback Sentiment Analysis", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

model = load_model()

# Gibberish detection function
def is_gibberish(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return True

    text_clean = re.sub(r'[\d' + string.punctuation + ']', '', text.lower())
    tokens = text_clean.split()

    if not tokens:
        return True

    real_words = sum(1 for word in tokens if word in english_words or (len(word) > 3 and any(v in word for v in "aeiou")))

    return real_words / len(tokens) < 0.4

# Sentiment prediction function
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "No Feedback"
    result = model(text)[0]['label']
    if "1 star" in result or "2 stars" in result:
        return "Negative"
    elif "4 stars" in result or "5 stars" in result:
        return "Positive"
    else:
        return "Neutral"

# Generate suggestions based on feedback and sentiment
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
        "communication": "Work on communication skills for better clarity of concepts."
    }
    for keyword, suggestion in keyword_map.items():
        if keyword in feedback:
            return suggestion

    return "Consider reviewing this feedback to find areas for improvement."

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📤 Upload & Process Feedback",
    "🔧 Improvement Tips based on Feedback",
    "📊 Visualizations",
    "✍ Manual Feedback Entry",
    "📥 Download Results"
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
            df["FEEDBACK_TEXT"] = df["FEEDBACK_TEXT"].fillna("")
            df["Predicted_Sentiment"] = df["FEEDBACK_TEXT"].apply(predict_sentiment)
            df["Suggestion"] = df.apply(lambda row: generate_teacher_suggestion(row["FEEDBACK_TEXT"], row["Predicted_Sentiment"]), axis=1)
            st.session_state.df = df
            st.success("✅ Feedback processed successfully!")
            st.dataframe(df, use_container_width=True)

# Page 2: Improvement tips
elif page == "🔧 Improvement Tips based on Feedback":
    st.header("🔧 Improvement Tips for Teachers")
    if st.session_state.df.empty:
        st.info("ℹ Please upload and process a file first.")
    else:
        tips_df = st.session_state.df[["ROLL NUMBER", "FEEDBACK_TEXT", "Predicted_Sentiment", "Suggestion"]]
        st.dataframe(tips_df, use_container_width=True)

# Page 3: Visualizations
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

        if "TIMESTAMP" in st.session_state.df.columns:
            df_time = st.session_state.df.copy()
            df_time["TIMESTAMP"] = pd.to_datetime(df_time["TIMESTAMP"], errors="coerce")
            df_time = df_time.dropna(subset=["TIMESTAMP"])
            df_time["DATE"] = df_time["TIMESTAMP"].dt.date

            trend_data = df_time.groupby(["DATE", "Predicted_Sentiment"]).size().reset_index(name="Count")

            trend_fig = px.line(trend_data, x="DATE", y="Count", color="Predicted_Sentiment",
                                title="📈 Sentiment Trend Over Time", markers=True)

            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("📅 Timestamp column not found. Skipping trend analysis.")

# Page 4: Manual feedback entry
elif page == "✍ Manual Feedback Entry":
    st.header("✍ Enter Feedback Manually")
    feedback_input = st.text_area("Enter your feedback:")

    if st.button("Analyze Feedback"):
        if feedback_input.strip():
            if is_gibberish(feedback_input):
                st.warning("⚠ The feedback appears to be nonsensical. Please enter a valid sentence.")
            else:
                sentiment = predict_sentiment(feedback_input)
                suggestion = generate_teacher_suggestion(feedback_input, sentiment)

                st.success("✅ Analysis Completed")
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown(f"**Suggestion:** {suggestion}")
        else:
            st.warning("⚠ Please enter some feedback to analyze.")

# Page 5: Download results
elif page == "📥 Download Results":
    st.header("📥 Download Feedback Analysis Results")
    if st.session_state.df.empty:
        st.info("ℹ Please upload and process feedback first.")
    else:
        file_format = st.selectbox("Choose file format to download", ["CSV", "Excel"])
        filename = st.text_input("Enter file name:", value="sentiment_feedback")

        if file_format == "CSV":
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
        else:
            excel = st.session_state.df.to_excel(index=False, engine='openpyxl')
            st.download_button(
                label="📥 Download Excel",
                data=excel,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
