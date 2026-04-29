# BarberBot

Machine learning haircut recommender with a four-phase demo flow:

1. Frontend collects four style answers plus a face photo.
2. FastAPI sends the photo through a ResNet50 face-shape classifier.
3. A Scikit-Learn SVM recommends a haircut from the complete profile.
4. The browser shows the style, score, detected face shape, and saved image.

## Assets

Add presentation images to `src/images` using the filenames listed in
`src/images/README.md`.

## Optional Training

```powershell
python src/train_svm.py
python src/train_resnet.py
```

`train_resnet.py` expects face-shape folders under `src/training/face_shapes`.
