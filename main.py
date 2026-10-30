"""
Deep Learning & AI Internship — Week 1
FastAPI Real-Time Digit Prediction Server
"""

import io
import re
import base64
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="MNIST Real-Time Digit Recognizer",
    description="Inference API powered by TensorFlow Neural Network",
    version="1.0.0"
)

# Load pre-trained model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model("mnist_model.h5")
        print("[✔] TensorFlow MNIST Model loaded into memory!")
    except Exception as e:
        print(f"[!] Model loading warning: {e}. Run train.py first!")

class CanvasImagePayload(BaseModel):
    image_data: str

def preprocess_image(base64_string: str) -> np.ndarray:
    """Decodes base64 canvas image, converts to 28x28 grayscale tensor."""
    image_data = re.sub('^data:image/.+;base64,', '', base64_string)
    image_bytes = base64.b64decode(image_data)
    
    # Open image using PIL
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize canvas to MNIST 28x28 resolution
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Reshape for model input (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the interactive drawing canvas interface."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict_digit(payload: CanvasImagePayload):
    """Predicts drawn digit from base64 canvas image tensor."""
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Train model using train.py"})
    
    try:
        processed_tensor = preprocess_image(payload.image_data)
        predictions = model.predict(processed_tensor)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": [float(p) for p in predictions[0]]
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)