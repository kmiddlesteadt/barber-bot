"""
Vision Service: face detection and face-shape classification.

Uses Haar Cascade to locate the face and a fine-tuned ResNet50 final layer to
classify the detected face as Oval, Square, Round, Heart, or Diamond.
"""

import os

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class VisionService:
    FACE_SHAPES = ["Oval", "Square", "Round", "Heart", "Diamond"]

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.resnet_model = self._load_resnet50_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        print(f"Vision Service initialized with ResNet50 pipeline on {self.device}")

    def _load_resnet50_model(self):
        model_path = os.path.join(self.models_dir, "resnet50_face_shape.pth")

        if not os.path.exists(model_path):
            print(f"Warning: ResNet50 model not found at {model_path}")
            print("Run train_resnet.py with a face-shape dataset to train it.")
            return None

        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.FACE_SHAPES))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        print(f"ResNet50 model loaded from {model_path}")
        return model

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return None

        return max(faces, key=lambda face: face[2] * face[3])

    def classify_face_shape(self, image, face_box):
        if self.resnet_model is None:
            return "Oval", 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}

        try:
            x, y, w, h = face_box
            padding = int(h * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            x2 = min(image.shape[1], x + w + (2 * padding))
            y2 = min(image.shape[0], y + h + (2 * padding))

            face_crop = image[y:y2, x:x2]
            if face_crop.size == 0:
                return "Oval", 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.resnet_model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)

            prob_dict = {
                self.FACE_SHAPES[i]: float(probabilities[i].item())
                for i in range(len(self.FACE_SHAPES))
            }

            return (
                self.FACE_SHAPES[int(predicted_idx.item())],
                float(confidence.item()),
                prob_dict,
            )
        except Exception as exc:
            print(f"Warning: face shape classification failed: {exc}")
            return "Oval", 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}

    def detect_and_classify(self, image):
        try:
            face_box = self.detect_face(image)
            if face_box is None:
                return {
                    "success": False,
                    "face_shape": None,
                    "confidence": None,
                    "probabilities": None,
                    "face_box": None,
                    "error": "No face detected in image",
                }

            face_shape, confidence, probabilities = self.classify_face_shape(image, face_box)
            return {
                "success": True,
                "face_shape": face_shape,
                "confidence": confidence,
                "probabilities": probabilities,
                "face_box": tuple(int(value) for value in face_box),
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "face_shape": None,
                "confidence": None,
                "probabilities": None,
                "face_box": None,
                "error": str(exc),
            }


_vision_service = None


def get_vision_service(models_dir):
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService(models_dir)
    return _vision_service


def initialize_vision_service(models_dir):
    global _vision_service
    _vision_service = VisionService(models_dir)
    return _vision_service

