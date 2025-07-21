# 🎙️ Deepfake Voice Detection

An end-to-end AI system to detect deepfake voice recordings using both **audio features** and **transcribed text** from the audio. 
This project uses **XGBoost classifiers** trained on engineered features and **OpenAI Whisper** for transcription.

## 🚀 Key Features

- 🎧 **Extracts 43 audio features**:
  - MFCCs
  - Delta MFCCs
  - Pitch variance
  - Spectral contrast
  - Spectral flux
- 📝 **Transcribes audio using OpenAI Whisper**
- 🧠 **Trains three XGBoost models**:
  - Audio-only
  - Text-only
  - Combined (audio + text)
- 📊 **Evaluates all models**:
  - Confusion matrix
  - ROC-AUC
  - Classification report
- 💬 **Interpretable predictions**:
  - Custom logic explains why a sample is classified as fake or real
- 🌐 **Streamlit Web App**:
  - Upload `.wav` or `.mp3` files
  - Get instant predictions with reasoning
- 💾 **Preprocessed CSVs, saved models, and visualizations** included

---

## 🧠 Tech Stack

- **Python 3.11**
- **Librosa** – Audio feature extraction
- **OpenAI Whisper** – Transcription
- **XGBoost** – Classification
- **Scikit-learn** – Preprocessing, evaluation
- **Streamlit** – Frontend interface
- **Pandas**, **Joblib**, **TQDM**

---

## 📂 Project Structure

📦 Deepfake Voice Detection
├── data_preparation.py
├── feature_extraction.py
├── vectorize_features.py
├── train_xgboost.py
├── app.py


## 📈 Model Results

| Model        | Accuracy | AUC     |
|--------------|----------|---------|
| Audio-only   | ~84%     | ~0.91   |
| Text-only    | ~56%     | ~0.58   |
| Combined     | **~80%** | **0.87** |


## 🔍 How It Works

1. **Audio Feature Extraction**  
   43 custom audio features from each `.wav`/`.mp3` file using `librosa`

2. **Text Transcription**  
   Uses OpenAI Whisper to convert speech to text

3. **Vectorization**  
   - Audio → Scaled with `StandardScaler`  
   - Text → Transformed via `TfidfVectorizer` (1000 features)

4. **Model Training**  
   Trains 3 models:
   - Audio-only
   - Text-only
   - Combined (audio + transcript)

5. **Prediction Logic**  
   Uses probabilities from all models to make a final prediction with custom fallback logic:

6. **Explainability**

Every prediction includes:

- 🔊 Audio patterns (e.g., pitch, spectral variation)
- 📝 Text traits (length, punctuation, case)
- 🔀 Combined logic reasoning

## 🔮 Sample Output

✅ Final Prediction: Fake
📄 Transcript: "Hi there, I'm an AI-generated voice used for testing."

📌 Reasoning:
- 🔊 High variation in audio frequencies.
- 🎤 Unusual pitch variation detected.
- 📜 Transcript is long and coherent.
- 🔀 Combined analysis of audio and text features.

## 🏆 Achievements

- ✅ End-to-end audio+NLP pipeline
- ✅ Handles edge cases & borderline decisions
- ✅ Real-time usable with GUI
- ✅ Combines ML, audio signal processing, and LLM transcription

**⭐ If you liked this project, consider starring the repo!**
