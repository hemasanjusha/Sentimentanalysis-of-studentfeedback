from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "./student-feedback-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Create sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test with an example
print(sentiment_pipeline("Teaching style was very clear and helpful."))
