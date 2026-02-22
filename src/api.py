from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import numpy as np
import io

from src.model import SimpleCNN

app = FastAPI(title="Cats vs Dogs Classifier")

device = torch.device("cpu")

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.eval()

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
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        confidence = float(output.item())

    label = "dog" if confidence >= 0.5 else "cat"

    return {
        "prediction": label,
        "confidence": confidence
    }