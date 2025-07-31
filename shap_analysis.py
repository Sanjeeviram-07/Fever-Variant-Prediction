# shap_analysis.py
import shap
import pandas as pd
import pickle

# Load model
with open("model/ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load sample dataset
df = pd.read_csv("dataset.csv")
X = df.drop("fever_type", axis=1)

# Use SHAP for model explainability
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X)
