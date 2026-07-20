import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split

# ----------------------------
# Create images folder
# ----------------------------
os.makedirs("images", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

# ----------------------------
# Encoding
# ----------------------------
gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_map = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

df["gender"] = df["gender"].map(gender_map)
df["smoking_history"] = df["smoking_history"].map(smoking_map)

# ----------------------------
# Features
# ----------------------------
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/random_forest_model.pkl")

# ----------------------------
# Confusion Matrix
# ----------------------------
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.close()

# ----------------------------
# ROC Curve
# ----------------------------
RocCurveDisplay.from_estimator(model, X_test, y_test)

plt.title("ROC Curve")
plt.savefig("images/roc_curve.png")
plt.close()

# ----------------------------
# Precision Recall Curve
# ----------------------------
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)

plt.title("Precision Recall Curve")
plt.savefig("images/precision_recall_curve.png")
plt.close()

# ----------------------------
# Feature Importance
# ----------------------------
importance = model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(8,5))

plt.barh(feature_names, importance)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.close()

print("✅ All graphs generated successfully!")