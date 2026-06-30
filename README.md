# Speech Emotion Recognition

A Python project that uses the RAVDESS speech emotion dataset to extract audio features, train a Random Forest classifier, and perform a live emotion prediction from microphone input.

## Project overview

This repository includes:

- `data/` - raw audio dataset files (RAVDESS)
- `features/` - generated CSV files for labels and extracted audio features
- `models/` - saved machine learning model files
- `scripts/` - Python scripts for preparing data, training, and testing

## What it does

1. `scripts/createLabels.py` scans the RAVDESS audio files and creates `features/ravdessLabels.csv` with file paths and emotion labels.
2. `scripts/extractFeatures.py` loads each audio file, extracts MFCC, chroma, and Mel spectrogram features, and saves the result to `features/ravdessFeatures.csv`.
3. `scripts/train.py` trains a Random Forest classifier on the extracted features, evaluates accuracy, and saves the trained model to `models/ravdessRfModel.pkl`.
4. `scripts/liveTest.py` records a short microphone sample, extracts the same features, and predicts the emotion using the saved model.

## Requirements

Install the required Python packages before running the scripts. Example:

```bash
pip install numpy pandas scikit-learn joblib librosa sounddevice scipy tqdm
```

> If you prefer, create a virtual environment first and install packages inside it.

## Data setup

The project expects the RAVDESS audio files to be available under:

```text
data/ravdess/
```

The script `scripts/createLabels.py` will search this folder for `.wav` files and generate the label CSV.

## Usage

### 1. Create labels

```bash
python scripts/createLabels.py
```

This produces `features/ravdessLabels.csv` containing file paths and emotion labels.

### 2. Extract features

```bash
python scripts/extractFeatures.py
```

This reads `features/ravdessLabels.csv` and saves feature vectors to `features/ravdessFeatures.csv`.

### 3. Train the model

```bash
python scripts/train.py
```

This trains a Random Forest classifier, prints the evaluation report, and saves the model to `models/ravdessRfModel.pkl`.

### 4. Run live prediction

```bash
python scripts/liveTest.py
```

The script records audio from the default microphone, extracts features, and prints the predicted emotion.

## File structure

- `scripts/createLabels.py` - build label CSV from RAVDESS filenames
- `scripts/extractFeatures.py` - extract audio features for each sample
- `scripts/train.py` - train and evaluate the Random Forest model
- `scripts/liveTest.py` - record audio and infer emotion with the trained model

## Notes

- `features/ravdessFeatures.csv` contains the feature vectors and the target emotion label.
- `models/ravdessRfModel.pkl` is the serialized scikit-learn model used by `liveTest.py`.
- If you want different audio input or dataset locations, update the path variables in the scripts.

## Recommended next steps

- Add a `requirements.txt` file for reproducible installation.
- Add error handling for missing model or data files.
- Try other classifiers or neural network models for better accuracy.
