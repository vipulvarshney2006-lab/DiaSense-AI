import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/clean_diabetes_dataset.csv")

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# -----------------------------
# Cross Validation
# -----------------------------
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("="*50)

print("Cross Validation Accuracy")

print(scores)

print()

print("Average Accuracy : ", scores.mean())

print("Standard Deviation :", scores.std())

print("="*50)