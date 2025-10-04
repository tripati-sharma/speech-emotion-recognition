import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
features_csv = '../features/ravdessFeatures.csv'
model_path = '../models/ravdessRfModel.pkl'

# 1️⃣ Load features
df = pd.read_csv(features_csv)
X = df.drop('emotion', axis=1)
y = df['emotion']

# 2️⃣ Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3️⃣ Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=None,     # let trees grow fully
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
rf_model.fit(X_train, y_train)

# 4️⃣ Predict on test set
y_pred = rf_model.predict(X_test)

# 5️⃣ Evaluate
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6️⃣ Save trained model
joblib.dump(rf_model, model_path)
print(f"\nRandom Forest model saved to {model_path}")
