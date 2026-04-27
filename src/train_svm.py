"""
Generate synthetic training data and train SVM model for hairstyle recommendation.
Run this once to train and save the model.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os

def generate_synthetic_data(n_samples=200):
    """
    Generate synthetic training data.
    
    Features: [hair_type, hair_length, style_pref, facial_hair, face_shape]
    - hair_type: 0=Straight, 1=Wavy, 2=Curly
    - hair_length: 0=Short, 1=Medium, 2=Long
    - style_pref: 0=Low Maintenance, 1=Trendy, 2=Professional
    - facial_hair: 0=No, 1=Yes
    - face_shape: 0=Oval, 1=Square, 2=Round, 3=Heart, 4=Diamond
    
    Labels: 0=Fringe, 1=Fade
    """
    np.random.seed(42)
    
    # Generate random features (5 features now)
    X = np.random.randint(0, 3, size=(n_samples, 4))
    X[:, 3] = np.random.randint(0, 2, size=n_samples)  # facial_hair: 0 or 1
    face_shapes = np.random.randint(0, 5, size=n_samples)  # face_shape: 0-4
    X = np.column_stack([X, face_shapes])
    
    # Create labels based on heuristic rules
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        hair_type, hair_length, style_pref, facial_hair, face_shape = X[i]
        
        # Heuristic with face shape consideration:
        # Fringe (0): Better for Straight/Wavy hair, Trendy preference, longer hair, Oval/Heart faces
        # Fade (1): Better for all types, but especially Curly, Low maintenance, shorter, Square/Round faces
        
        fringe_score = (hair_type * 0.3 + (2 - hair_length) * 0.2 + 
                       style_pref * 0.4 + (face_shape < 2) * 0.3)
        fade_score = ((2 - hair_length) * 0.5 + (1 - style_pref) * 0.3 + 
                     hair_type * 0.2 + (face_shape >= 3) * 0.3)
        
        y[i] = 0 if fringe_score >= fade_score else 1
    
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
    
    print(f"✓ Model saved to {model_dir}")

def main():
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=200)
    print(f"  Generated {len(X)} samples")
    print(f"  Fringe (0): {np.sum(y == 0)} samples")
    print(f"  Fade (1): {np.sum(y == 1)} samples")
    
    print("\nTraining SVM model...")
    svm, scaler = train_svm_model(X, y)
    print(f"  SVM trained with {len(svm.support_vectors_)} support vectors")
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    print("\nSaving model...")
    save_model(svm, scaler, model_dir)
    
    # Test with example
    print("\n--- Test Prediction ---")
    test_sample = np.array([[0, 0, 1, 0, 0]])  # Straight, Short, Trendy, No beard, Oval face
    test_scaled = scaler.transform(test_sample)
    pred = svm.predict(test_scaled)[0]
    proba = svm.predict_proba(test_scaled)[0]
    
    styles = ['Fringe', 'Fade']
    print(f"Test input (hair_type, hair_length, style_pref, facial_hair, face_shape): {test_sample[0]}")
    print(f"Prediction: {styles[pred]}")
    print(f"Confidence: Fringe={proba[0]:.2f}, Fade={proba[1]:.2f}")

if __name__ == '__main__':
    main()