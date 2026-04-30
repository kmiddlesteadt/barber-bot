"""
BarberBot FastAPI backend.

Receives questionnaire answers plus a Base64 face photo, detects face shape with
ResNet50, recommends a haircut with an SVM, and returns a saved image path.
"""

import base64
import io
import os
import pickle
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

try:
    from .train_svm import generate_synthetic_data, train_svm_model
    from .vision_service import initialize_vision_service
except ImportError:
    from train_svm import generate_synthetic_data, train_svm_model
    from vision_service import initialize_vision_service


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(
    title="BarberBot Backend",
    version="1.0",
    description="Face-shape detection plus SVM haircut recommendation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class AnalyzeRequest(BaseModel):
    hairType: str
    hairLength: str
    stylePref: str
    facialHair: str
    image: str


class TryOnRequest(BaseModel):
    image: str
    haircut: str


FEATURE_ENCODING = {
    "hairType": {"Straight": 0, "Wavy": 1, "Curly": 2},
    "hairLength": {"Short": 0, "Medium": 1, "Long": 2},
    "stylePref": {"Low Maintenance": 0, "Trendy": 1, "Professional": 2},
    "facialHair": {"No": 0, "Yes": 1},
    "faceShape": {"Oval": 0, "Square": 1, "Round": 2, "Heart": 3, "Diamond": 4},
}

HAIRSTYLES: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Fringe",
        "slug": "fringe",
        "description": "Textured Fringe - Adds volume and modern style.",
    },
    1: {
        "name": "Fade",
        "slug": "fade",
        "description": "Classic Fade - Clean sides, timeless look.",
    },
}


def load_or_create_svm():
    """Load trained SVM assets, or build a demo model when assets are absent."""
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as model_file:
            svm_model = pickle.load(model_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return svm_model, scaler, False

    X, y = generate_synthetic_data(n_samples=200)
    svm_model, scaler = train_svm_model(X, y)
    return svm_model, scaler, True


svm_model, scaler, using_demo_svm = load_or_create_svm()
vision_service = initialize_vision_service(MODELS_DIR)


def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode a browser data URL or raw Base64 image string into an OpenCV image."""
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Base64 image") from exc

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode_answers(payload: AnalyzeRequest, face_shape: str) -> np.ndarray:
    """Convert questionnaire answers plus detected face shape into SVM features."""
    try:
        return np.array(
            [
                FEATURE_ENCODING["hairType"][payload.hairType],
                FEATURE_ENCODING["hairLength"][payload.hairLength],
                FEATURE_ENCODING["stylePref"][payload.stylePref],
                FEATURE_ENCODING["facialHair"][payload.facialHair],
                FEATURE_ENCODING["faceShape"][face_shape],
            ]
        ).reshape(1, -1)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported answer: {exc}") from exc


def confidence_for_prediction(probabilities: List[float], prediction: int) -> float:
    if len(probabilities) > prediction:
        return float(probabilities[prediction])
    return float(max(probabilities)) if len(probabilities) else 0.0


def hairstyle_from_slug_or_name(haircut: str) -> Dict[str, str]:
    normalized = haircut.strip().lower()
    for hairstyle in HAIRSTYLES.values():
        if normalized in {hairstyle["slug"], hairstyle["name"].lower()}:
            return hairstyle
    raise HTTPException(status_code=400, detail=f"Unsupported haircut: {haircut}")


@app.get("/api/health")
def health():
    return {
        "status": "OK",
        "message": "BarberBot backend is running",
        "svm_mode": "demo fallback" if using_demo_svm else "trained model",
    }


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    image = decode_base64_image(payload.image)

    vision_result = vision_service.detect_and_classify(image)
    if not vision_result["success"]:
        raise HTTPException(
            status_code=400,
            detail=f"Face detection failed: {vision_result['error']}",
        )

    detected_face_shape = vision_result["face_shape"]
    features = encode_answers(payload, detected_face_shape)
    features_scaled = scaler.transform(features)

    prediction = int(svm_model.predict(features_scaled)[0])
    probabilities = svm_model.predict_proba(features_scaled)[0]
    confidence = confidence_for_prediction(probabilities, prediction)

    hairstyle = HAIRSTYLES.get(prediction, HAIRSTYLES[0])
    image_path = f"/images/{hairstyle['slug']}.jpg"

    return {
        "success": True,
        "style_name": hairstyle["name"],
        "style_slug": hairstyle["slug"],
        "description": hairstyle["description"],
        "confidence": round(confidence, 4),
        "confidence_percent": f"{round(confidence * 100)}%",
        "image_path": image_path,
        "svm_scores": {
            HAIRSTYLES[i]["name"]: round(float(score), 4)
            for i, score in enumerate(probabilities)
            if i in HAIRSTYLES
        },
        "user_profile": {
            "hairType": payload.hairType,
            "hairLength": payload.hairLength,
            "stylePref": payload.stylePref,
            "facialHair": payload.facialHair,
            "faceShape": detected_face_shape,
        },
        "face_shape_analysis": {
            "detected_shape": detected_face_shape,
            "confidence": round(float(vision_result["confidence"]), 4),
            "probabilities": vision_result["probabilities"],
        },
    }


@app.post("/api/try-on-haircut")
def try_on_haircut(payload: TryOnRequest):
    """
    Fake virtual try-on endpoint.

    This keeps the same request shape a real image-editing endpoint would use,
    but returns the pre-generated haircut image for a reliable demo.
    """
    decode_base64_image(payload.image)
    hairstyle = hairstyle_from_slug_or_name(payload.haircut)

    return {
        "success": True,
        "preview_mode": "fake",
        "style_name": hairstyle["name"],
        "style_slug": hairstyle["slug"],
        "edited_image_path": f"/images/{hairstyle['slug']}.jpg",
        "message": "Demo preview: showing a pre-generated haircut image instead of editing the uploaded photo.",
    }


@app.get("/")
def root():
    return {
        "name": "BarberBot Backend",
        "version": "1.0",
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/analyze": "POST - Analyze questionnaire + Base64 image",
            "/api/try-on-haircut": "POST - Fake virtual haircut preview",
            "/images/{style}.jpg": "Static pre-generated haircut images",
        },
    }
