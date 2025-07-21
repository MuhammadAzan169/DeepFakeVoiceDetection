import os
import librosa
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import torch
import shutil

# ⚙️ Parameters
SR = 16000
N_MFCC = 20
WHISPER_MODEL = "medium"
SAVE_EVERY = 50
MIN_DURATION = 0.5
EXPECTED_FEATURES = 43
REQUIRED_COLUMNS = EXPECTED_FEATURES + 3

# ✅ Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"🚀 Using device: {device}")

# ✅ Load Whisper
print(f"🔊 Loading Whisper model: {WHISPER_MODEL}")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL).to(device)
    if device == "cuda":
        whisper_model = whisper_model.half()
except Exception as e:
    print(f"❌ Failed to load Whisper: {str(e)}")
    exit(1)

# ✅ Load metadata with path validation
def load_metadata():
    splits = {}
    for split in ['train', 'val', 'test']:
        try:
            df = pd.read_csv(f'{split}_metadata.csv')
            # Verify files exist before processing
            df['exists'] = df['file_path'].apply(os.path.exists)
            print(f"✅ {split}: {df['exists'].sum()}/{len(df)} files exist")
            splits[split] = df[df['exists']].copy()
        except Exception as e:
            print(f"❌ Failed to load {split}_metadata.csv: {str(e)}")
            exit(1)
    return splits

splits = load_metadata()
os.makedirs("features", exist_ok=True)

def extract_features(file_path):
    """Enhanced feature extraction with better error handling"""
    try:
        # Double-check file existence
        if not os.path.exists(file_path):
            return None, None

        y, sr = librosa.load(file_path, sr=SR)
        
        if librosa.get_duration(y=y, sr=sr) < MIN_DURATION:
            return None, None

        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        pitch = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        valid_pitch = pitch[~np.isnan(pitch)]
        pitch_var = np.std(valid_pitch) if len(valid_pitch) > 0 else 0.0
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spectral_contrast)
        spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            [pitch_var, spec_contrast_mean, spectral_flux]
        ])

        # Whisper transcription
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            result = whisper_model.transcribe(file_path, language='en')
        transcript = result.get('text', '').strip()

        return features, transcript

    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(file_path)}: {str(e)}")
        return None, None

# ✅ Process splits with atomic writes
for split_name, df in splits.items():
    output_path = f"features/{split_name}_features.csv"
    temp_path = f"features/{split_name}_features_temp.csv"
    backup_path = f"features/{split_name}_features_backup.csv"
    
    # Load existing data if available
    existing_data = []
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if len(existing_df.columns) == REQUIRED_COLUMNS:
                existing_data = existing_df.to_dict('records')
                print(f"📝 Resuming {split_name} with {len(existing_data)} existing entries")
        except:
            print(f"⚠️ Corrupted {output_path}, starting fresh")

    processed_files = {x['file_path'] for x in existing_data}
    features = existing_data.copy()
    
    print(f"\n🔄 Processing {split_name} ({len(df)} files)")
    
    try:
        progress_bar = tqdm(df.iterrows(), total=len(df))
        for _, row in progress_bar:
            file_path = row['file_path']
            
            if file_path in processed_files:
                continue
                
            feature_vec, transcript = extract_features(file_path)
            
            if feature_vec is not None and transcript is not None:
                if len(feature_vec) != EXPECTED_FEATURES:
                    continue
                    
                features.append({
                    'file_path': file_path,
                    'label': row['label'],
                    'transcript': transcript,
                    **{f'feature_{i}': float(x) for i, x in enumerate(feature_vec)}
                })
                processed_files.add(file_path)
                
                # Update progress
                progress_bar.set_postfix({
                    'processed': len(features),
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1e6:.1f}MB" if device == 'cuda' else 'cpu'
                })
                
                # Periodic save
                if len(features) % SAVE_EVERY == 0:
                    pd.DataFrame(features).to_csv(temp_path, index=False)
                    shutil.copy2(temp_path, backup_path)  # Create backup
                    os.replace(temp_path, output_path)
                    
    except Exception as e:
        print(f"❌ Error in {split_name}: {str(e)}")
        traceback.print_exc()
    finally:
        if features:
            pd.DataFrame(features).to_csv(temp_path, index=False)
            shutil.copy2(temp_path, backup_path)
            os.replace(temp_path, output_path)
            print(f"💾 Saved {len(features)} entries to {output_path}")

print("\n🎉 Feature extraction complete!")