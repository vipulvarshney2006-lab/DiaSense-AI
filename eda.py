import pandas as pd

# Load dataset
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

# Basic information
print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\n" + "=" * 50)
print("Columns")
print(df.columns)

print("\n" + "=" * 50)
print("Data Types")
print(df.dtypes)

print("\n" + "=" * 50)
print("Missing Values")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Duplicate Rows")
print(df.duplicated().sum())

print("\n" + "=" * 50)
print("Target Variable Distribution")
print(df["diabetes"].value_counts())

print("\n" + "=" * 50)
print("Statistical Summary")
print(df.describe())