from transformers import pipeline

model_path = "D:/student-feedback-sentiment-model"

# Create sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Run a test
result = sentiment_pipeline("The teaching style is excellent and very clear.")
print(result)
