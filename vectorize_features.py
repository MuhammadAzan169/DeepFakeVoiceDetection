import os 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

# ✅ Set path to features
FEATURES_DIR = "features"
TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_features.csv")
VAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "val_features.csv")
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "test_features.csv")

# ✅ Load feature CSVs
train_df = pd.read_csv(TRAIN_FEATURES_PATH)
val_df = pd.read_csv(VAL_FEATURES_PATH)
test_df = pd.read_csv(TEST_FEATURES_PATH)

# ✅ TF-IDF Leakage Check
print("🔍 Top 10 most common transcripts in training data:")
print(train_df['transcript'].value_counts().head(10))

# ✅ Fit TF-IDF only on training transcripts to avoid leakage
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf.fit(train_df['transcript'].fillna(""))

# ✅ Transform transcripts to TF-IDF vectors
X_train_text = tfidf.transform(train_df['transcript'].fillna(""))
X_val_text = tfidf.transform(val_df['transcript'].fillna(""))
X_test_text = tfidf.transform(test_df['transcript'].fillna(""))

# ✅ Use exactly the first 43 audio features
feature_cols = [f'feature_{i}' for i in range(43)]

# ✅ Validate all dataframes contain these 43 features
for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"❌ {name}_features.csv is missing columns: {missing}")

# ✅ Extract and scale audio features
X_train_audio = train_df[feature_cols].values
X_val_audio = val_df[feature_cols].values
X_test_audio = test_df[feature_cols].values

scaler = StandardScaler()
X_train_audio = scaler.fit_transform(X_train_audio)
X_val_audio = scaler.transform(X_val_audio)
X_test_audio = scaler.transform(X_test_audio)

# ✅ Combine TF-IDF + audio features
X_train = hstack([X_train_text, X_train_audio])
X_val = hstack([X_val_text, X_val_audio])
X_test = hstack([X_test_text, X_test_audio])

# ✅ Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['label'])
y_val = label_encoder.transform(val_df['label'])
y_test = label_encoder.transform(test_df['label'])

# ✅ Save vectorized data
joblib.dump((X_train, y_train), "train_vectorized.pkl")
joblib.dump((X_val, y_val), "val_vectorized.pkl")
joblib.dump((X_test, y_test), "test_vectorized.pkl")

# ✅ Save supporting encoders
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "audio_scaler.pkl")

# ✅ Save raw DataFrames for downstream modeling (audio-only, text-only, etc.)
joblib.dump(train_df, "train_raw_df.pkl")
joblib.dump(val_df, "val_raw_df.pkl")
joblib.dump(test_df, "test_raw_df.pkl")

print("✅ All features scaled, vectorized, and saved successfully!")
print("📦 Raw DataFrames also saved for backup and model variants.")
