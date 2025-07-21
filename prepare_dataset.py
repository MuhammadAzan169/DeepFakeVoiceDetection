import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the base directory
BASE_DIR = r"C:\Users\muham\OneDrive\Desktop\Deep Fake Voice Detection\for-norm"

# Define the subfolders for fake and real audio
FAKE_FOLDERS = ['fake1', 'fake2', 'fake3']
REAL_FOLDERS = ['real1', 'real2', 'real3']

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Process fake audio files
for folder in FAKE_FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            file_paths.append(os.path.join(folder_path, file_name))
            labels.append(1)  # Label 1 for fake

# Process real audio files
for folder in REAL_FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            file_paths.append(os.path.join(folder_path, file_name))
            labels.append(0)  # Label 0 for real

# Create a DataFrame
data = pd.DataFrame({
    'file_path': file_paths,
    'label': labels
})

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training (80%), validation (10%), and test (10%) sets
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Save the splits to CSV files
train_data.to_csv('train_metadata.csv', index=False)
val_data.to_csv('val_metadata.csv', index=False)
test_data.to_csv('test_metadata.csv', index=False)

print(f"Total samples: {len(data)}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}") 