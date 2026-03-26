import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router
from app.services.artifact_loader import ArtifactLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deepfake Voice Detection API",
    description="API for detecting deepfake voices using audio and text analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        loader = ArtifactLoader()
        app.state.artifact_loader = loader
        app.state.artifacts_ready = True
        app.state.prediction_service = None
        logger.info("All artifacts loaded successfully")
    except Exception as e:
        app.state.artifacts_ready = False
        app.state.prediction_service = None
        logger.error("Failed to load artifacts: %s", str(e))


app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Deepfake Voice Detection API",
        "version": "1.0.0",
        "status": "operational",
        "artifacts_ready": getattr(app.state, "artifacts_ready", False),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if getattr(app.state, "artifacts_ready", False) else "degraded",
        "artifacts_ready": getattr(app.state, "artifacts_ready", False),
        "models": ["audio", "text", "combined"] if getattr(app.state, "artifacts_ready", False) else [],
    }


@app.get("/ready")
async def readiness_check():
    if getattr(app.state, "artifacts_ready", False):
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "not ready", "reason": "artifacts not loaded"})
