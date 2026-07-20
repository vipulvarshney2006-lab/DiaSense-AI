import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

df = pd.read_csv("dataset/clean_diabetes_dataset.csv")

# Load the encoders already fit on raw text in preprocessing.py
gender_encoder = joblib.load("models/gender_encoder.pkl")
smoking_encoder = joblib.load("models/smoking_encoder.pkl")

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # now actually used/consistent

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
# gender_encoder / smoking_encoder are already saved by preprocessing.py — no need to re-dump

print("✅ Model saved successfully!")