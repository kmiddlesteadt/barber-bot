"""
Generate synthetic training data and train SVM model for hairstyle recommendation.
Run this once to train and save the model.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os

STYLE_NAMES = [
    'Fringe',
    'Fade',
]


def score_styles(hair_type, hair_length, style_pref, facial_hair, face_shape):
    """
    Score the two original haircut classes from the five profile variables.

    Features:
    - hair_type: 0=Straight, 1=Wavy, 2=Curly
    - hair_length: 0=Short, 1=Medium, 2=Long
    - style_pref: 0=Low Maintenance, 1=Trendy, 2=Professional
    - facial_hair: 0=No, 1=Yes
    - face_shape: 0=Oval, 1=Square, 2=Round, 3=Heart, 4=Diamond
    """
    is_wavy = hair_type == 1
    is_curly = hair_type == 2
    is_short = hair_length == 0
    is_medium = hair_length == 1
    wants_easy = style_pref == 0
    wants_trendy = style_pref == 1
    is_oval = face_shape == 0
    is_square = face_shape == 1
    is_round = face_shape == 2
    is_heart = face_shape == 3

    return np.array([
        1.6 * wants_trendy + 1.2 * is_medium + 0.8 * is_wavy + 0.5 * (is_oval or is_heart),
        1.5 * wants_easy + 1.2 * is_short + 0.8 * is_curly + 0.5 * (is_square or is_round),
    ])


def generate_synthetic_data(n_samples=200):
    """
    Generate synthetic training data.
    
    Features: [hair_type, hair_length, style_pref, facial_hair, face_shape]
    - hair_type: 0=Straight, 1=Wavy, 2=Curly
    - hair_length: 0=Short, 1=Medium, 2=Long
    - style_pref: 0=Low Maintenance, 1=Trendy, 2=Professional
    - facial_hair: 0=No, 1=Yes
    - face_shape: 0=Oval, 1=Square, 2=Round, 3=Heart, 4=Diamond
    
    Labels: 0=Fringe, 1=Fade.
    """
    np.random.seed(42)
    
    # Generate random features (5 features now)
    X = np.random.randint(0, 3, size=(n_samples, 4))
    X[:, 3] = np.random.randint(0, 2, size=n_samples)  # facial_hair: 0 or 1
    face_shapes = np.random.randint(0, 5, size=n_samples)  # face_shape: 0-4
    X = np.column_stack([X, face_shapes])
    
    # Create labels based on heuristic style scores with small deterministic noise.
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        hair_type, hair_length, style_pref, facial_hair, face_shape = X[i]
        
        scores = score_styles(hair_type, hair_length, style_pref, facial_hair, face_shape)
        scores += np.random.normal(0, 0.08, size=len(STYLE_NAMES))
        y[i] = int(np.argmax(scores))
    
    return X, y

def train_svm_model(X, y):
    """Train and return SVM model"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM with probability estimates
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_scaled, y)
    
    return svm, scaler

def save_model(svm, scaler, model_dir):
    """Save trained model and scaler"""
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm, f)
    
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_dir}")

def main():
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=200)
    print(f"  Generated {len(X)} samples")
    for idx, name in enumerate(STYLE_NAMES):
        print(f"  {name} ({idx}): {np.sum(y == idx)} samples")
    
    print("\nTraining SVM model...")
    svm, scaler = train_svm_model(X, y)
    print(f"  SVM trained with {len(svm.support_vectors_)} support vectors")
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    print("\nSaving model...")
    save_model(svm, scaler, model_dir)
    
    # Test with example
    print("\n--- Test Prediction ---")
    test_sample = np.array([[1, 1, 1, 0, 1]])  # Wavy, Medium, Trendy, No beard, Square face
    test_scaled = scaler.transform(test_sample)
    pred = svm.predict(test_scaled)[0]
    proba = svm.predict_proba(test_scaled)[0]
    
    print(f"Test input (hair_type, hair_length, style_pref, facial_hair, face_shape): {test_sample[0]}")
    print(f"Prediction: {STYLE_NAMES[pred]}")
    for idx, score in enumerate(proba):
        print(f"  {STYLE_NAMES[idx]}={score:.2f}")

if __name__ == '__main__':
    main()
