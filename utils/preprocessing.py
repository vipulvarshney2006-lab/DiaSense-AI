import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

print("Original Shape :", df.shape)
df = df.drop_duplicates()
print("After Removing Duplicates :", df.shape)

# Encode Gender
gender_encoder = LabelEncoder()
df["gender"] = gender_encoder.fit_transform(df["gender"])

# Encode Smoking History
smoking_encoder = LabelEncoder()
df["smoking_history"] = smoking_encoder.fit_transform(df["smoking_history"])

# Save the encoders — fit on the ORIGINAL text labels
joblib.dump(gender_encoder, "models/gender_encoder.pkl")
joblib.dump(smoking_encoder, "models/smoking_encoder.pkl")

print("\nFirst Five Rows:\n")
print(df.head())

df.to_csv("dataset/clean_diabetes_dataset.csv", index=False)
print("\nClean dataset saved successfully!")
print("Encoders saved successfully!")