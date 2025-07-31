# train_model.py

import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load the data
df = pd.read_csv("dataset.csv")

# Map target labels
df['fever_type'] = df['fever_type'].map({
    'Normal': 0,
    'Dengue': 1,
    'Malaria': 2,
    'Typhoid': 3
})

X = df.drop('fever_type', axis=1)
y = df['fever_type']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier()

lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    min_data_in_leaf=1,
    min_data_in_bin=1,
    force_col_wise=True,
    verbosity=-1
)

ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb),
], voting='soft')

# Train
ensemble_model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, ensemble_model.predict(X_test))
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)
