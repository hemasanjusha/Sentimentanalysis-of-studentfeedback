from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Use raw string for Windows path and point to correct folder
model_path = r"D:\student-feedback-sentiment-model"

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

# Create sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test with an example
print(sentiment_pipeline("Teaching style was very clear and helpful."))
