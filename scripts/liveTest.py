import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import joblib
import librosa

# Paths
model_path = '../models/ravdessRfModel.pkl'

# Load trained Random Forest model
rf_model = joblib.load(model_path)
print("Random Forest model loaded!")

# Recording parameters
fs = 44100       # Sample rate
seconds = 4      # Duration of recording
filename = 'test.wav'

# 1️⃣ Record audio
print("Recording... Speak now!")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write(filename, fs, recording)  # Save as WAV file
print(f"Recording saved as {filename}")

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
# 2️⃣ Extract features
features = extract_features(filename)
features = features.reshape(1, -1)  # reshape for prediction

# 3️⃣ Predict emotion
predicted_emotion = rf_model.predict(features)
print("Predicted Emotion:", predicted_emotion[0])
