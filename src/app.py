"""
BarberBot Flask Backend
Hairstyle recommendation system using SVM and image overlay
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os
import base64
from PIL import Image
import io
from vision_service import initialize_vision_service, get_vision_service

app = Flask(__name__)
CORS(app)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
HAIRSTYLES_DIR = os.path.join(BASE_DIR, 'hairstyles')

# Load models at startup
print("Loading SVM model...")
with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)

print("Loading scaler...")
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Initialize Vision Service for face shape classification using ResNet18
print("Initializing Vision Service (ResNet18 Face Shape Classifier)...")
vision_service = initialize_vision_service(MODELS_DIR)

# Hairstyle configuration
HAIRSTYLES = {
    0: {'name': 'Fringe', 'file': 'fringe.png', 'description': 'Textured Fringe - Adds volume and modern style'},
    1: {'name': 'Fade', 'file': 'fade.png', 'description': 'Classic Fade - Clean sides, timeless look'}
}

# Feature encoding
FEATURE_ENCODING = {
    'hairType': {'Straight': 0, 'Wavy': 1, 'Curly': 2},
    'hairLength': {'Short': 0, 'Medium': 1, 'Long': 2},
    'stylePref': {'Low Maintenance': 0, 'Trendy': 1, 'Professional': 2},
    'facialHair': {'No': 0, 'Yes': 1},
    'faceShape': {'Oval': 0, 'Square': 1, 'Round': 2, 'Heart': 3, 'Diamond': 4}
}


def encode_answers(hair_type, hair_length, style_pref, facial_hair, face_shape):
    """Convert questionnaire answers + detected face shape to SVM feature vector (5 features)"""
    features = np.array([
        FEATURE_ENCODING['hairType'][hair_type],
        FEATURE_ENCODING['hairLength'][hair_length],
        FEATURE_ENCODING['stylePref'][style_pref],
        FEATURE_ENCODING['facialHair'][facial_hair],
        FEATURE_ENCODING['faceShape'][face_shape]
    ]).reshape(1, -1)
    return features


def detect_face(image):
    """Detect face using vision_service - returns face_box for hairstyle overlay"""
    vision_result = vision_service.detect_and_classify(image)
    if vision_result['success']:
        return vision_result['face_box']
    return None


def overlay_hairstyle(image, face_box, hairstyle_path):
    """Overlay hairstyle PNG on the detected face region"""
    try:
        x, y, w, h = face_box
        
        # Load hairstyle with transparency
        hairstyle = Image.open(hairstyle_path).convert('RGBA')
        
        # Resize hairstyle to match face width, extend height for proper coverage
        hairstyle_height = int(h * 1.2)
        hairstyle = hairstyle.resize((w, hairstyle_height), Image.Resampling.LANCZOS)
        
        # Position hairstyle above the face (slightly overlapping)
        paste_y = max(0, y - int(h * 0.3))
        
        # Convert image to PIL for pasting
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Paste hairstyle with alpha blending
        image_pil.paste(hairstyle, (x, paste_y), hairstyle)
        
        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return result
    except Exception as e:
        print(f"Error overlaying hairstyle: {e}")
        return image


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode()


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    Processes questionnaire + image and returns recommended hairstyle with overlay
    Now includes face shape detection via ResNet18 as a feature input to SVM
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image file
        image_file = request.files['image']
        img_bytes = image_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get questionnaire answers
        hair_type = request.form.get('hairType')
        hair_length = request.form.get('hairLength')
        style_pref = request.form.get('stylePref')
        facial_hair = request.form.get('facialHair')
        
        # Validate all answers provided
        if not all([hair_type, hair_length, style_pref, facial_hair]):
            return jsonify({'error': 'Missing questionnaire answers'}), 400
        
        # Detect face and classify face shape using ResNet18
        vision_result = vision_service.detect_and_classify(image)
        if not vision_result['success']:
            return jsonify({'error': f'Face detection failed: {vision_result["error"]}'}), 400
        
        detected_face_shape = vision_result['face_shape']
        face_shape_confidence = vision_result['confidence']
        face_shape_probabilities = vision_result['probabilities']
        face_box = vision_result['face_box']
        
        # Encode answers including detected face shape and predict with SVM
        try:
            features = encode_answers(hair_type, hair_length, style_pref, facial_hair, detected_face_shape)
            features_scaled = scaler.transform(features)
            
            prediction = svm_model.predict(features_scaled)[0]
            probabilities = svm_model.predict_proba(features_scaled)[0]
            
            recommended_style_id = int(prediction)
            fringe_confidence = float(probabilities[0])
            fade_confidence = float(probabilities[1])
        except Exception as e:
            return jsonify({'error': f'SVM prediction failed: {str(e)}'}), 500
        
        # Overlay hairstyle
        hairstyle_file = HAIRSTYLES[recommended_style_id]['file']
        hairstyle_path = os.path.join(HAIRSTYLES_DIR, hairstyle_file)
        
        if not os.path.exists(hairstyle_path):
            return jsonify({'error': f'Hairstyle file not found: {hairstyle_file}'}), 500
        
        result_image = overlay_hairstyle(image, face_box, hairstyle_path)
        
        # Encode result to base64
        result_base64 = image_to_base64(result_image)
        
        # Build response
        response = {
            'success': True,
            'recommended_style': HAIRSTYLES[recommended_style_id]['name'],
            'description': HAIRSTYLES[recommended_style_id]['description'],
            'confidence': float(max(fringe_confidence, fade_confidence)),
            'svm_scores': {
                'Fringe': fringe_confidence,
                'Fade': fade_confidence
            },
            'user_profile': {
                'hairType': hair_type,
                'hairLength': hair_length,
                'stylePref': style_pref,
                'facialHair': facial_hair
            },
            'face_shape_analysis': {
                'detected_shape': detected_face_shape,
                'confidence': face_shape_confidence,
                'probabilities': face_shape_probabilities
            },
            'result_image': f'data:image/png;base64,{result_base64}'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'OK', 'message': 'BarberBot backend is running'}), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        'name': 'BarberBot Backend',
        'version': '1.0',
        'endpoints': {
            '/api/health': 'GET - Health check',
            '/api/analyze': 'POST - Analyze questionnaire + image, return hairstyle recommendation'
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*50)
    print("BarberBot Backend Starting...")
    print("="*50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Hairstyles directory: {HAIRSTYLES_DIR}")
    print("="*50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)