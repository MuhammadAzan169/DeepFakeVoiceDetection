# Deepfake Voice Detection

A full-stack deepfake voice detection project with:

- FastAPI backend for model inference
- React + Vite frontend for upload/record and result visualization
- local training/inference utility scripts in the project root

The app predicts whether an input voice sample is likely `Real` or `Fake`, and returns transcript and explanation signals.

## Repository Structure

```text
backend/                 # FastAPI API server
frontend/                # React + Vite web app
app.py                   # local script entry
extract_features.py      # feature extraction utility
prepare_dataset.py       # dataset preparation utility
train_xgboost.py         # model training utility
vectorize_features.py    # text/audio vectorization utility
artifacts/               # local model artifacts (ignored in git)
features/                # generated feature files (ignored in git)
plots/                   # generated plots (ignored in git)
```

## Tech Stack

- Backend: FastAPI, Uvicorn, scikit-learn/joblib, Whisper
- Frontend: React, Vite, Axios
- Models: XGBoost-based classifiers and preprocessing artifacts loaded from `artifacts/`

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+
- Git

## 1) Backend Setup and Run (PowerShell)

```powershell
cd D:/Deep Fake Voice Detection/backend
python -m pip install -r requirements.txt
python run.py
```

Backend runs at:

- http://localhost:8000

## 2) Frontend Setup and Run (PowerShell)

```powershell
cd D:/Deep Fake Voice Detection/frontend
npm install
npm run dev
```

Frontend runs at:

- http://localhost:5173

## API Endpoints

- `GET /api/v1/health` - service health and artifact state
- `GET /api/v1/ready` - readiness probe
- `GET /api/v1/models/info` - model info
- `POST /api/v1/predict` - single file prediction
- `POST /api/v1/predict/batch` - batch prediction

## Quick API Test

```powershell
curl.exe -X GET "http://localhost:8000/api/v1/health"
curl.exe -X POST "http://localhost:8000/api/v1/predict" -F "file=@D:/path/to/sample.wav"
```

## Frontend Notes

The UI supports:

- audio file upload (MP3/WAV/WEBM/OGG/M4A)
- microphone recording in browser
- mobile-optimized layout (improved touch targets, compact small-screen layout, safe-area support)

## Important: Artifacts Are Required

Inference will only work when required model artifacts are present in `artifacts/`:

- `xgb_audio.pkl`
- `xgb_text.pkl`
- `xgb_combined.pkl`
- `tfidf_vectorizer.pkl`
- `audio_scaler.pkl`
- `expected_feature_count.pkl`

If these files are missing, backend health/readiness will report not ready and prediction endpoints will not process requests.

## Privacy and Git Hygiene

This repository is configured to avoid pushing local/private generated content:

- `artifacts/`
- `plots/`
- `frontend/node_modules/`
- `frontend/dist/`
- `frontend/.vite/`
- generated feature CSVs
- metadata CSV files containing local absolute paths
- caches, temp files, and environment files

If you share data files, sanitize local machine paths and personal identifiers first.

## Pulling and Running on Another PC

Yes, you can clone this repository on another PC and run the project.

What will run immediately:

- frontend app startup
- backend server startup
- health routes

What needs extra setup for full predictions:

- install backend/frontend dependencies
- provide required files in `artifacts/`
- ensure model-compatible environment (Python packages and Whisper dependencies)

Without artifacts, prediction endpoints will return service-not-ready behavior by design.

## Troubleshooting

- Frontend cannot connect: confirm backend is running on `http://localhost:8000`
- Backend not ready: check missing files in `artifacts/`
- Prediction timeout: try smaller audio file (limit is 25MB)

## License

Add a LICENSE file if you plan to open-source this repository publicly.
