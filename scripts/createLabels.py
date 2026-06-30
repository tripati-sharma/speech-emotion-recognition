import os
import pandas as pd

# Map filename codes to emotion labels
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

ravdess_dir = '../data/ravdess'
data_list = []

# Walk through all files and extract emotion from filename
for root, _, files in os.walk(ravdess_dir):
    for file in files:
        if file.endswith('.wav'):
            parts = file.split('-')
            emotion = emotion_map.get(parts[2], 'unknown')
            path = os.path.join(root, file)
            data_list.append([path, emotion])

# Save to CSV
df = pd.DataFrame(data_list, columns=['path', 'emotion'])
df.to_csv('../features/ravdessLabels.csv', index=False)
print(f"CSV created with {len(df)} entries.")
