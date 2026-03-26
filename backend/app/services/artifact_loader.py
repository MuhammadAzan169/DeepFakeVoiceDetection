import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """Load all required artifacts from the artifacts folder."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.artifacts_dir = self.project_root / "artifacts"
        self.artifacts = {}
        self._load_all_artifacts()

    def _load_all_artifacts(self):
        artifacts_to_load = {
            "model_audio": "xgb_audio.pkl",
            "model_text": "xgb_text.pkl",
            "model_combined": "xgb_combined.pkl",
            "tfidf": "tfidf_vectorizer.pkl",
            "scaler": "audio_scaler.pkl",
            "expected_dim": "expected_feature_count.pkl",
        }

        missing_artifacts = []

        for key, filename in artifacts_to_load.items():
            path = self.artifacts_dir / filename
            if path.exists():
                try:
                    self.artifacts[key] = joblib.load(path)
                    logger.info("Loaded %s from %s", key, path)
                except Exception as e:
                    logger.error("Failed to load %s: %s", key, str(e))
                    missing_artifacts.append(filename)
            else:
                logger.error("Artifact not found: %s", path)
                missing_artifacts.append(filename)

        if "expected_dim" in self.artifacts:
            self.artifacts["expected_dim"] = int(self.artifacts["expected_dim"])

        if missing_artifacts:
            raise FileNotFoundError(f"Missing artifacts: {', '.join(missing_artifacts)}")

        try:
            import whisper

            self.artifacts["whisper_model"] = whisper.load_model("base")
            logger.info("Loaded Whisper model")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", str(e))
            raise

    def get(self, key):
        return self.artifacts.get(key)

    def get_all(self):
        return self.artifacts
