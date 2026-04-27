"""
Vision Service: Face Detection and Face Shape Classification
Uses Haar Cascade for face detection and ResNet18 for face shape classification
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from PIL import Image
import io


class VisionService:
    """
    Handles all computer vision tasks:
    - Face detection using Haar Cascade
    - Face shape classification using ResNet18
    """
    
    FACE_SHAPES = ['Oval', 'Square', 'Round', 'Heart', 'Diamond']
    
    def __init__(self, models_dir):
        """
        Initialize vision service with pre-trained models
        
        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load ResNet18 model
        self.resnet_model = self._load_resnet_model()
        
        # Image transformation pipeline for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Vision Service initialized (device: {self.device})")
    
    def _load_resnet_model(self):
        """Load pre-trained ResNet18 model for face shape classification"""
        model_path = os.path.join(self.models_dir, 'resnet18_face_shape.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: ResNet model not found at {model_path}")
            print("   Run train_resnet.py first to train the model")
            return None
        
        # Create model architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)  # 5 face shapes
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ ResNet18 model loaded from {model_path}")
        return model
    
    def detect_face(self, image):
        """
        Detect face in image using Haar Cascade
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            face_box: Tuple (x, y, w, h) or None if no face detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        face_box = max(faces, key=lambda f: f[2] * f[3])
        return face_box
    
    def classify_face_shape(self, image, face_box):
        """
        Classify face shape from cropped face region
        
        Args:
            image: OpenCV image (BGR format)
            face_box: Tuple (x, y, w, h) from Haar Cascade detection
            
        Returns:
            face_shape: String of detected face shape (one of FACE_SHAPES)
            confidence: Float confidence score (0.0 to 1.0)
            probabilities: Dict of {face_shape: probability} for all shapes
        """
        if self.resnet_model is None:
            print("⚠️  ResNet model not available, returning 'Oval' as default")
            return 'Oval', 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}
        
        try:
            x, y, w, h = face_box
            
            # Add padding to include more context
            padding = int(h * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = w + 2 * padding
            h = h + 2 * padding
            
            # Ensure bounds don't exceed image
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            x = max(0, min(x, x2 - 1))
            y = max(0, min(y, y2 - 1))
            
            # Crop face region
            face_crop = image[y:y2, x:x2]
            
            if face_crop.size == 0:
                print("⚠️  Invalid face crop, returning 'Oval' as default")
                return 'Oval', 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}
            
            # Convert BGR to RGB
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL and apply transforms
            face_pil = Image.fromarray(face_crop_rgb)
            face_tensor = self.transform(face_pil).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.resnet_model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_idx = predicted_idx.item()
                confidence = confidence.item()
            
            # Convert probabilities to dict
            prob_dict = {
                self.FACE_SHAPES[i]: float(probabilities[i].item())
                for i in range(len(self.FACE_SHAPES))
            }
            
            face_shape = self.FACE_SHAPES[predicted_idx]
            
            return face_shape, confidence, prob_dict
        
        except Exception as e:
            print(f"⚠️  Error in face shape classification: {e}")
            print("   Returning 'Oval' as default")
            return 'Oval', 0.5, {shape: 0.2 for shape in self.FACE_SHAPES}
    
    def detect_and_classify(self, image):
        """
        Complete pipeline: detect face and classify its shape
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            result: Dict with keys:
                - 'success': bool
                - 'face_shape': str or None
                - 'confidence': float or None
                - 'probabilities': dict or None
                - 'face_box': tuple (x, y, w, h) or None
                - 'error': str or None
        """
        try:
            # Detect face
            face_box = self.detect_face(image)
            if face_box is None:
                return {
                    'success': False,
                    'face_shape': None,
                    'confidence': None,
                    'probabilities': None,
                    'face_box': None,
                    'error': 'No face detected in image'
                }
            
            # Classify face shape
            face_shape, confidence, probabilities = self.classify_face_shape(image, face_box)
            
            return {
                'success': True,
                'face_shape': face_shape,
                'confidence': confidence,
                'probabilities': probabilities,
                'face_box': face_box,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'face_shape': None,
                'confidence': None,
                'probabilities': None,
                'face_box': None,
                'error': str(e)
            }


# Singleton instance for use in app.py
_vision_service = None

def get_vision_service(models_dir):
    """
    Get or create Vision Service instance (singleton pattern)
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        VisionService instance
    """
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService(models_dir)
    return _vision_service


def initialize_vision_service(models_dir):
    """
    Explicitly initialize vision service
    Call this from app.py at startup
    """
    global _vision_service
    _vision_service = VisionService(models_dir)
    return _vision_service