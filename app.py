import torch
from fastapi import FastAPI
from transformers import BertForSequenceClassification, BertTokenizer
import os
import requests
import zipfile

app = FastAPI()

MODEL_DIR = "./model"
GITHUB_URL = "https://github.com/your-username/your-repo/raw/main/model.zip"  # Change this

# Function to download the model from GitHub
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        response = requests.get(GITHUB_URL)
        with open("model.zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove("model.zip")
        print("✅ Model downloaded and extracted.")

# Ensure model is downloaded
download_model()

# Load model
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.eval()

# Function to classify emails
def predict_spam(email_text):
    inputs = tokenizer(
        email_text,
        padding="max_length",
        truncation=True,
        max_length=100,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return "not spam" if predicted_class == 0 else "spam"

# API Endpoint
@app.post("/detect_spam/")
async def detect_spam(email: dict):
    text = email.get("text", "")
    prediction = predict_spam(text)
    return {"prediction": prediction}
