# app.py

import streamlit as st

# 🌐 Must be at the top before any other Streamlit command
st.set_page_config(page_title="Deepfake Voice Detector", layout="centered")

import numpy as np
import joblib
import whisper
import librosa
from scipy.sparse import hstack
import tempfile
import os

# --- Load Models & Artifacts ---
@st.cache_resource
def load_artifacts():
    return {
        "model_audio": joblib.load("xgb_audio.pkl"),
        "model_text": joblib.load("xgb_text.pkl"),
        "model_combined": joblib.load("xgb_combined.pkl"),
        "tfidf": joblib.load("tfidf_vectorizer.pkl"),
        "scaler": joblib.load("audio_scaler.pkl"),
        "expected_dim": joblib.load("expected_feature_count.pkl"),
        "whisper_model": whisper.load_model("base")
    }

artifacts = load_artifacts()

# --- Explainability Functions ---
def explain_audio(feats):
    bullets = []
    if np.std(feats[:5]) > 15: bullets.append("🔊 High variation in audio frequencies.")
    if feats[40] > 0.02: bullets.append("🎤 Unusual pitch variation detected.")
    if feats[41] < 15: bullets.append("🔍 Low contrast in audio spectrum.")
    if feats[42] < 100: bullets.append("⚡ Smooth transitions in audio detected.")
    return bullets

def explain_text(txt):
    bullets = []
    if len(txt.split()) < 5: bullets.append("✂️ Very short transcript.")
    if txt.isupper(): bullets.append("🔠 All-caps text detected.")
    if len(txt.split()) > 20: bullets.append("📜 Transcript is long and coherent.")
    if sum(1 for c in txt if c in ".,?") < 2: bullets.append("❓ Limited punctuation in text.")
    return bullets

def explain_combined():
    return ["🔀 Combined analysis of audio and text features."]

# --- Prediction Pipeline ---
def predict_audio(fp):
    whisper_model = artifacts["whisper_model"]

    # 1. Load and preprocess audio
    y, sr = librosa.load(fp, sr=16000)
    audio = whisper.pad_or_trim(y)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    result = whisper.decode(whisper_model, mel, whisper.DecodingOptions())
    txt = result.text.strip() or "[No speech]"

    # 2. Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    dmfcc = librosa.feature.delta(mfccs).mean()
    try: pv = np.var(librosa.yin(y, 80, 400, sr=sr))
    except: pv = 0.0
    sc = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    sf = np.mean(np.diff(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats = np.hstack([mfccs, np.repeat(dmfcc, 20), pv, sc, sf]).reshape(1, -1)

    scaled = artifacts["scaler"].transform(feats)
    tv = artifacts["tfidf"].transform([txt])
    comb = hstack([tv, scaled])

    # 3. Predict
    pa = artifacts["model_audio"].predict_proba(scaled)[0][1]
    pt = artifacts["model_text"].predict_proba(tv)[0][1]
    pc = artifacts["model_combined"].predict_proba(comb)[0][1]

    # 4. Decision
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

    reasons = explain_audio(feats.flatten()) + explain_text(txt) + explain_combined()
    return txt, final_label, reasons

# --- Streamlit Interface ---
st.title("🎙️ Deepfake Voice Detection")
st.write("Upload a `.mp3` or `.wav` file to detect if it's real or fake.")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Processing file...")
    transcript, label, reasons = predict_audio(tmp_path)

    st.subheader("📝 Transcript")
    st.write(transcript)

    st.subheader("✅ Final Prediction")
    st.success(f"This voice is **{label}**")

    st.subheader("📌 Reasoning")
    for r in reasons:
        st.markdown(f"- {r}")

    os.remove(tmp_path)
