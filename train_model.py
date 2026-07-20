import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Load Clean Dataset
# -----------------------------
df = pd.read_csv("dataset/clean_diabetes_dataset.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Results
# -----------------------------
print("=" * 50)
print("Accuracy")

print(accuracy_score(y_test, y_pred))

print("\n")

print("=" * 50)
print("Confusion Matrix")

print(confusion_matrix(y_test, y_pred))

print("\n")

print("=" * 50)
print("Classification Report")

print(classification_report(y_test, y_pred))