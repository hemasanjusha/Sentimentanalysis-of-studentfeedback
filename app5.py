import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load model
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Gibberish detection function
def is_gibberish(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return True
    text = text.strip()
    if len(text) < 4:
        return True
    total = len(text)
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return (non_alpha / total) > 0.5

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# Predict sentiment
def predict_sentiment(text):
    if is_gibberish(text):
        return "invaild text pleased provide valid text"
    if not text.strip():
        return "No Feedback"
    result = sentiment_model(text)[0]['label']
    if "1 star" in result or "2 stars" in result:
        return "Negative"
    elif "4 stars" in result or "5 stars" in result:
        return "Positive"
    else:
        return "Neutral"

# Suggestion generator
def generate_suggestion(feedback):
    if not isinstance(feedback, str) or feedback.strip() == "" or is_gibberish(feedback):
        return "Invalid or gibberish feedback."

    feedback_lower = feedback.lower()
    suggestions = []

    if any(word in feedback_lower for word in ["practical", "real-world", "real time", "application", "case study"]):
        suggestions.append("Add more practical or real-world examples to improve clarity.")
    if any(word in feedback_lower for word in ["technical issue", "network", "connectivity", "online issue"]):
        suggestions.append("Improve technical delivery such as internet connection or audio quality.")
    if any(word in feedback_lower for word in ["quiz", "assignment", "homework", "test", "exam"]):
        suggestions.append("Continue regular quizzes and assignments for better learning retention.")
    if any(word in feedback_lower for word in ["engaging", "boring", "interactive", "interesting", "monotonous"]):
        suggestions.append("Make sessions more engaging using visuals or interactive elements.")
    if any(word in feedback_lower for word in ["fast", "slow", "pace", "speed", "rushed"]):
        suggestions.append("Adjust the teaching pace to suit student understanding.")
    if any(word in feedback_lower for word in ["explain", "clarity", "understand", "clear"]):
        suggestions.append("Focus more on clear explanations with student-friendly language.")
    if "no" in feedback_lower or "nil" in feedback_lower:
        suggestions.append("Maintain current quality and adapt based on more feedback.")

    if not suggestions:
        return "Maintain current quality and adapt based on more feedback."

    return " ".join(suggestions)

# Streamlit UI
st.title("📊 Student Feedback Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload your feedback file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.upper()

    if "ROLL NUMBER" not in df.columns or "FEEDBACK_TEXT" not in df.columns:
        st.error("❌ Dataset must contain 'ROLL NUMBER' and 'FEEDBACK_TEXT' columns.")
    else:
        df = df.drop_duplicates(subset="ROLL NUMBER", keep="first")
        df["FEEDBACK_TEXT"] = df["FEEDBACK_TEXT"].fillna("")
        df["Cleaned_Text"] = df["FEEDBACK_TEXT"].apply(preprocess_text)
        df["Predicted_Sentiment"] = df["FEEDBACK_TEXT"].apply(predict_sentiment)
        df["AI_Suggestion"] = df["FEEDBACK_TEXT"].apply(generate_suggestion)

        st.success("✅ Feedback Processed!")
        st.write(df.head())

        st.subheader("📈 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Predicted_Sentiment", palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("☁️ Word Clouds")
        for sentiment in ["Positive", "Negative"]:
            text_data = df[df["Predicted_Sentiment"] == sentiment]["Cleaned_Text"]
            text = " ".join(text_data.dropna().astype(str))
            if text.strip():
                wc = WordCloud(width=600, height=400, background_color='white').generate(text)
                st.markdown(f"#### {sentiment} Feedback")
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.warning(f"⚠️ No words found for '{sentiment}' sentiment. Skipping WordCloud.")

        # Download processed CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Processed Feedback", csv, "processed_feedback.csv", "text/csv")
