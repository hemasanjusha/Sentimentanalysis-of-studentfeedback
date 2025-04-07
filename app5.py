from transformers import pipeline

sentiment_pipeline = pipeline("text-classification", model="Hemasanjusha/sentiment-analysis-model")

result = sentiment_pipeline("The teacher explained clearly.")
st.write(result)
