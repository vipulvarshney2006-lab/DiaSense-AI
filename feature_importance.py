import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Load dataset
df = pd.read_csv("dataset/clean_diabetes_dataset.csv")

X = df.drop("diabetes", axis=1)

# Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# Plot
plt.figure(figsize=(10,6))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()