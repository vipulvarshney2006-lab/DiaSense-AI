import joblib
import shap

model = joblib.load("models/random_forest_model.pkl")

explainer = shap.TreeExplainer(model)

joblib.dump(explainer, "models/shap_explainer.pkl")

print("SHAP Explainer Saved Successfully!")