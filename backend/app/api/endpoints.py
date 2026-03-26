import logging
import time
import traceback
from typing import List

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.models import ErrorDetail, HealthResponse, PredictionResponse
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".webm", ".ogg", ".m4a")


def _is_supported_audio_upload(file: UploadFile) -> bool:
    filename_ok = bool(file.filename) and file.filename.lower().endswith(ALLOWED_AUDIO_EXTENSIONS)
    content_type_ok = bool(file.content_type) and file.content_type.lower().startswith("audio/")
    return filename_ok or content_type_ok


def _get_prediction_service(request: Request) -> PredictionService:
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        svc = PredictionService(request.app.state.artifact_loader)
        request.app.state.prediction_service = svc
    return svc


@router.get("/health", response_model=HealthResponse)
async def health_v1(request: Request):
    ready = getattr(request.app.state, "artifacts_ready", False)
    return HealthResponse(
        status="healthy" if ready else "degraded",
        artifacts_ready=ready,
        models=["audio", "text", "combined"] if ready else [],
    )


@router.get("/ready")
async def ready_v1(request: Request):
    ready = getattr(request.app.state, "artifacts_ready", False)
    if ready:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not ready", "reason": "artifacts not loaded"})


@router.post("/predict", response_model=PredictionResponse)
async def predict_audio(request: Request, file: UploadFile = File(...)):
    start_time = time.time()

    if not getattr(request.app.state, "artifacts_ready", False):
        return PredictionResponse(
            success=False,
            error=ErrorDetail(code="SERVICE_UNAVAILABLE", message="Models are not loaded. Please try again later."),
        )

    if not _is_supported_audio_upload(file):
        return PredictionResponse(
            success=False,
            error=ErrorDetail(code="INVALID_FILE_TYPE", message="Invalid file type. Supported formats: MP3, WAV, WEBM, OGG, M4A."),
        )

    try:
        content = await file.read()

        if len(content) > 25 * 1024 * 1024:
            return PredictionResponse(
                success=False,
                error=ErrorDetail(code="FILE_TOO_LARGE", message="File too large. Maximum size is 25MB."),
            )

        prediction_service = _get_prediction_service(request)
        transcript, label, reasons = prediction_service.predict(content, file.filename)

        processing_time = time.time() - start_time

        return PredictionResponse(
            success=True,
            data={
                "transcript": transcript,
                "label": label,
                "reasons": reasons,
                "processing_time": round(processing_time, 2),
            },
        )

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        logger.error(traceback.format_exc())
        raw_error = str(e).strip()
        detail = raw_error if raw_error else e.__class__.__name__
        return PredictionResponse(
            success=False,
            error=ErrorDetail(code="PREDICTION_ERROR", message=f"Error processing audio: {detail}"),
        )


@router.post("/predict/batch")
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    if not getattr(request.app.state, "artifacts_ready", False):
        return JSONResponse(status_code=503, content={"status": "not ready", "reason": "artifacts not loaded"})

    prediction_service = _get_prediction_service(request)
    results = []

    for file in files:
        try:
            content = await file.read()
            if not _is_supported_audio_upload(file):
                results.append({"filename": file.filename, "status": "failed", "error": "Invalid file type"})
                continue
            if len(content) > 25 * 1024 * 1024:
                results.append({"filename": file.filename, "status": "failed", "error": "File too large"})
                continue

            transcript, label, reasons = prediction_service.predict(content, file.filename)
            results.append(
                {
                    "filename": file.filename,
                    "status": "processed",
                    "transcript": transcript,
                    "label": label,
                    "reasons": reasons,
                }
            )
        except Exception as e:
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})

    return {"results": results}


@router.get("/models/info")
async def get_models_info(request: Request):
    if not getattr(request.app.state, "artifacts_ready", False):
        return JSONResponse(status_code=503, content={"status": "unavailable", "message": "Models not loaded"})

    try:
        info = _get_prediction_service(request).get_models_info()
        return JSONResponse(content=info)
    except Exception as e:
        logger.error("Error getting models info: %s", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
