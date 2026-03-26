from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    file_name: str
    file_size: int


class PredictionResponse(BaseModel):
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "transcript": "Hello world",
                    "label": "Real",
                    "reasons": [
                        "Low contrast in audio spectrum.",
                        "Smooth transitions in audio detected.",
                        "Combined analysis of audio and text features.",
                    ],
                    "processing_time": 2.34,
                },
            }
        }
    )


class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
    models: List[str]
