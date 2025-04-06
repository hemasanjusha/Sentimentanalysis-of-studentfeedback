from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = r"D:\student-feedback-sentiment-model"  # Use raw string for Windows paths

# Load tokenizer and model explicitly
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test input
result = sentiment_pipeline("The teaching style is excellent and very clear.")
print(result)
