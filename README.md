# Speech Emotion Recognition

This project extracts audio features from the RAVDESS dataset and trains models to classify speech emotions.

## Structure

- `data/` — dataset folders
- `features/` — extracted feature CSVs
- `models/` — trained model files
- `scripts/` — feature extraction, training, and live test scripts

## GitHub setup

1. Make sure the remote is set to your GitHub repo:
   ```powershell
   git remote set-url origin https://github.com/tripati-sharma/speech-emotion-recognition.git
   ```
2. Add and commit your files:
   ```powershell
   git add .
   git commit -m "Initial commit"
   ```
3. Push to GitHub:
   ```powershell
   git push -u origin main
   ```

## Notes

- `data/` is ignored in `.gitignore` because raw datasets are typically too large for GitHub.
- If you want to include a smaller dataset or specific files, remove them from `.gitignore` before committing.
