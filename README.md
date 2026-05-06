# BarberBot

BarberBot is an AI-powered haircut recommendation demo. The frontend asks for
basic hair preferences and a face photo, then the FastAPI backend analyzes the
photo, predicts a face shape, and recommends a haircut style.

The demo flow is:

1. The browser collects four style answers plus a face photo.
2. FastAPI sends the photo through a ResNet50 face-shape classifier.
3. A Scikit-Learn SVM recommends a haircut from the complete profile.
4. The browser shows the recommended style, score, detected face shape, and
   haircut image.

## Project Layout

- `src/app.py` - FastAPI backend and API endpoints
- `src/html/index.html` - landing page for the browser demo
- `src/html/questionnaire.html` - questionnaire and camera upload flow
- `src/models/` - trained model files used by the backend
- `src/images/` - haircut and demo images served by the backend
- `src/training/face_shapes/` - training images grouped by face shape

## Set Up Python

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run The Backend

Start the FastAPI server from the project root:

```powershell
uvicorn src.app:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Useful backend URLs:

- `http://127.0.0.1:8000/api/health` - checks that the API is running
- `http://127.0.0.1:8000/docs` - interactive FastAPI API docs
- `http://127.0.0.1:8000/images/{style}.jpg` - served haircut images

## Run The Frontend With VS Code Live Server

1. Open this project folder in VS Code.
2. Install the "Live Server" extension if it is not already installed.
3. Right-click `src/html/index.html`.
4. Choose "Open with Live Server".
5. Use the page that opens in your browser to start the BarberBot demo.

Keep the `uvicorn` backend running while using Live Server. The HTML pages call
the backend API at `http://127.0.0.1:8000`.

## Optional Training

The project already includes model files in `src/models/`. To retrain them, run
these commands from the project root after activating the virtual environment:

```powershell
python src/train_svm.py
python src/train_resnet.py
```

`train_resnet.py` expects face-shape folders under
`src/training/face_shapes`.
