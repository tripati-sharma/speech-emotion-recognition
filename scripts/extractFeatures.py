import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
labels_csv = '../features/ravdessLabels.csv'
output_csv = '../features/ravdessFeatures.csv'

# Function to extract features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        return np.hstack((mfcc, chroma, mel))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Read labels CSV
df_labels = pd.read_csv(labels_csv)
features_list = []

# Extract features for each file
for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
    file_path = row['path']
    emotion = row['emotion']
    features = extract_features(file_path)
    if features is not None:
        features_list.append([*features, emotion])

# Convert to DataFrame
num_features = len(features_list[0]) - 1
columns = [f'feat_{i}' for i in range(num_features)] + ['emotion']
df_features = pd.DataFrame(features_list, columns=columns)

# Save features CSV
df_features.to_csv(output_csv, index=False)
print(f"Feature extraction complete. Saved to {output_csv}")
