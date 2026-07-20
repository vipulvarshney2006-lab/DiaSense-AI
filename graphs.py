import os
import pandas as pd
import matplotlib.pyplot as plt

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# Graph 1: Diabetes Distribution
# -----------------------------
plt.figure(figsize=(6,5))
df["diabetes"].value_counts().plot(kind="bar")
plt.title("Diabetes Distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.xticks([0,1],["No","Yes"])
plt.tight_layout()
plt.savefig("images/01_diabetes_distribution.png")
plt.show()

# -----------------------------
# Graph 2: Age Distribution
# -----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("images/02_age_distribution.png")
plt.show()

# -----------------------------
# Graph 3: BMI Distribution
# -----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["bmi"], bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/03_bmi_distribution.png")
plt.show()

# -----------------------------
# Graph 4: Blood Glucose Distribution
# -----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["blood_glucose_level"], bins=20)
plt.title("Blood Glucose Distribution")
plt.xlabel("Blood Glucose")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/04_glucose_distribution.png")
plt.show()

# -----------------------------
# Graph 5: HbA1c Distribution
# -----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["HbA1c_level"], bins=20)
plt.title("HbA1c Level Distribution")
plt.xlabel("HbA1c")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/05_hba1c_distribution.png")
plt.show()