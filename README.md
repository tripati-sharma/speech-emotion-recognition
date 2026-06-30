# Speech Emotion Recognition

This project extracts audio features from the RAVDESS dataset and trains models to classify speech emotions.

## Structure

- `data/` — dataset folders
- `features/` — extracted feature CSVs
- `models/` — trained model files
- `scripts/` — feature extraction, training, and live test scripts

## GitHub setup

1. Create a new repo on GitHub.
2. In this folder:
   ```bash
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```

## Notes

- `data/` is ignored in `.gitignore` because raw datasets are typically too large for GitHub.
- If you want to include a smaller dataset or specific files, remove them from `.gitignore` before committing.
