from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import numpy as np
import io
import logging
import time
import csv
from datetime import datetime
from src.model import SimpleCNN

app = FastAPI(title="Cats vs Dogs Classifier")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_logger")

# Simple in-memory request counter
request_count = 0
device = torch.device("cpu")

# Load trained model
import os

model = SimpleCNN()
model_path = "models/model.pt"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print("Warning: model file not found. Running without loaded weights.")

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    global request_count
    start_time = time.time()

    request_count += 1

    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        confidence = output.item()

    label = "dog" if confidence >= 0.5 else "cat"

    latency = time.time() - start_time

    # Log request info (no sensitive data)
    logger.info(
        f"Request #{request_count} | Prediction: {label} | "
        f"Confidence: {confidence:.4f} | Latency: {latency:.4f}s"
    )

    # Save prediction to CSV (monitoring file)
    with open("prediction_logs.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(),
            label,
            confidence,
            latency
        ])

    return {
        "prediction": label,
        "confidence": float(confidence),
        "latency_seconds": latency,
        "total_requests": request_count
    }