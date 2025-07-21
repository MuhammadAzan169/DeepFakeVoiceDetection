import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

# 🔧 Configuration
sns.set_theme(style="darkgrid")
sns.set_palette("husl")
RANDOM_STATE = 42
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data():
    """Load processed datasets"""
    print("🚀 Loading vectorized datasets...")

    try:
        # Load combined vectorized datasets
        X_train_comb, y_train = joblib.load("train_vectorized.pkl")
        X_val_comb, y_val = joblib.load("val_vectorized.pkl")
        X_test_comb, y_test = joblib.load("test_vectorized.pkl")

        # Load raw DataFrames
        train_df = joblib.load("train_raw_df.pkl")
        val_df = joblib.load("val_raw_df.pkl")
        test_df = joblib.load("test_raw_df.pkl")

        # Load TF-IDF vectorizer and audio scaler
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        scaler = joblib.load("audio_scaler.pkl")

        # Extract audio features
        audio_cols = [f"feature_{i}" for i in range(43)]
        X_train_audio = scaler.transform(train_df[audio_cols].values)
        X_val_audio = scaler.transform(val_df[audio_cols].values)
        X_test_audio = scaler.transform(test_df[audio_cols].values)

        # Extract text features (TF-IDF)
        X_train_text = tfidf.transform(train_df['transcript'].fillna("")).toarray()
        X_val_text = tfidf.transform(val_df['transcript'].fillna("")).toarray()
        X_test_text = tfidf.transform(test_df['transcript'].fillna("")).toarray()

        # Convert combined matrices to dense
        X_train_comb = X_train_comb.toarray()
        X_val_comb = X_val_comb.toarray()
        X_test_comb = X_test_comb.toarray()

        print("✅ All data loaded and separated\n")

        return {
            'combined': (X_train_comb, y_train, X_val_comb, y_val, X_test_comb, y_test),
            'audio':    (X_train_audio, y_train, X_val_audio, y_val, X_test_audio, y_test),
            'text':     (X_train_text,  y_train, X_val_text,  y_val, X_test_text,  y_test)
        }

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit(1)


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier"""
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=20,
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    return model


def evaluate_model(model, X, y, set_name="Test", prefix="combined"):
    """Print metrics and generate plots"""
    print(f"\n📊 {prefix.upper()} - {set_name} Set Evaluation:")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print(classification_report(y, y_pred, target_names=['Real', 'Fake']))
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f"Confusion Matrix ({prefix.upper()} - {set_name})")
    plt.savefig(os.path.join(PLOT_DIR, f'cm_{prefix}_{set_name.lower()}.png'))
    plt.close()

    roc_auc = roc_auc_score(y, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"ROC Curve ({prefix.upper()} - {set_name})")
    plt.savefig(os.path.join(PLOT_DIR, f'roc_{prefix}_{set_name.lower()}.png'))
    plt.close()


def train_and_save(name, data_tuple):
    """Train and evaluate for each feature type"""
    print("\n" + "=" * 60)
    print(f"🧠 Training and Evaluating Model: {name.upper()}")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = data_tuple
    model = train_model(X_train, y_train, X_val, y_val)

    # Save model
    model_path = f"xgb_{name}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved as {model_path}")

    # Evaluate
    evaluate_model(model, X_train, y_train, "Training", prefix=name)
    evaluate_model(model, X_val, y_val, "Validation", prefix=name)
    evaluate_model(model, X_test, y_test, "Test", prefix=name)


def main():
    # ✅ Load all datasets
    all_data = load_data()

    # ✅ Train and evaluate models for each type
    for i, key in enumerate(['audio', 'text', 'combined'], 1):
        print(f"\n🚀 Running model {i}/3 → [{key.upper()}]")
        train_and_save(key, all_data[key])

    # ✅ Save expected feature count for predictor
    joblib.dump(all_data['combined'][0].shape[1], "expected_feature_count.pkl")

    print("\n🎉 All models trained, evaluated, and saved successfully!")
    print("📊 Check the 'plots/' folder for confusion matrices and ROC curves.")


if __name__ == "__main__":
    main()
