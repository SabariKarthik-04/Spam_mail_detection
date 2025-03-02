from fastapi import FastAPI
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = FastAPI()

# Load model from Hugging Face Hub
MODEL_NAME = "sabari04/spam-detector"

model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model.eval()

@app.post("/detect_spam/")
async def detect_spam(email: dict):
    text = email.get("text", "")
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return {"prediction": "not spam" if predicted_class == 1 else "spam"}
