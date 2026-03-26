import os
import tempfile
import logging

import librosa
import numpy as np
import whisper
from scipy.sparse import hstack

from app.services.artifact_loader import ArtifactLoader

logger = logging.getLogger(__name__)


class PredictionService:
    """Service that handles the prediction logic using existing app.py behavior."""

    def __init__(self, artifact_loader=None):
        if artifact_loader:
            self.artifacts = artifact_loader.get_all()
        else:
            self.artifacts_loader = ArtifactLoader()
            self.artifacts = self.artifacts_loader.get_all()

    def explain_audio(self, feats):
        bullets = []
        if np.std(feats[:5]) > 15:
            bullets.append("🔊 High variation in audio frequencies.")
        if feats[40] > 0.02:
            bullets.append("🎤 Unusual pitch variation detected.")
        if feats[41] < 15:
            bullets.append("🔍 Low contrast in audio spectrum.")
        if feats[42] < 100:
            bullets.append("⚡ Smooth transitions in audio detected.")
        return bullets

    def explain_text(self, txt):
        bullets = []
        if len(txt.split()) < 5:
            bullets.append("✂️ Very short transcript.")
        if txt.isupper():
            bullets.append("🔠 All-caps text detected.")
        if len(txt.split()) > 20:
            bullets.append("📜 Transcript is long and coherent.")
        if sum(1 for c in txt if c in ".,?") < 2:
            bullets.append("❓ Limited punctuation in text.")
        return bullets

    def explain_combined(self):
        return ["🔀 Combined analysis of audio and text features."]

    def predict(self, audio_content, filename):
        suffix = os.path.splitext(filename)[1] if os.path.splitext(filename)[1] else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_content)
            tmp_path = tmp.name

        try:
            whisper_model = self.artifacts["whisper_model"]

            y, sr = librosa.load(tmp_path, sr=16000)
            audio = whisper.pad_or_trim(y)
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
            # Force English output regardless of spoken language.
            decode_options = whisper.DecodingOptions(task="translate", language="en")
            result = whisper.decode(whisper_model, mel, decode_options)
            txt = result.text.strip() or "[No speech]"

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
            dmfcc = librosa.feature.delta(mfccs).mean()
            try:
                pv = np.var(librosa.yin(y, 80, 400, sr=sr))
            except Exception:
                pv = 0.0
            sc = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
            sf = np.mean(np.diff(librosa.feature.spectral_centroid(y=y, sr=sr)))
            feats = np.hstack([mfccs, np.repeat(dmfcc, 20), pv, sc, sf]).reshape(1, -1)

            scaled = self.artifacts["scaler"].transform(feats)
            tv = self.artifacts["tfidf"].transform([txt])
            comb = hstack([tv, scaled])

            pa = self.artifacts["model_audio"].predict_proba(scaled)[0][1]
            pt = self.artifacts["model_text"].predict_proba(tv)[0][1]
            pc = self.artifacts["model_combined"].predict_proba(comb)[0][1]

            if pc > 0.5:
                final_label = "Fake"
            elif pc == 0.5:
                if pa > 0.515:
                    final_label = "Fake"
                elif pt > pa:
                    final_label = "Real"
                else:
                    final_label = "Fake" if pc > 0.5 else "Real"
            else:
                final_label = "Real"

            reasons = self.explain_audio(feats.flatten()) + self.explain_text(txt) + self.explain_combined()
            return txt, final_label, reasons

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def get_models_info(self):
        return {
            "models": {
                "audio": {"type": "XGBoost", "loaded": True},
                "text": {"type": "XGBoost", "loaded": True},
                "combined": {"type": "XGBoost", "loaded": True},
            },
            "artifacts": list(self.artifacts.keys()),
            "feature_dimension": self.artifacts.get("expected_dim"),
            "status": "ready",
        }
