# app.py

from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Load the model
model_path = "model/ensemble_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Class labels
labels = {
    0: 'Normal',
    1: 'Dengue',
    2: 'Malaria',
    3: 'Typhoid'
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Fetch inputs
    try:
        data = [
            int(request.form['headache']),
            int(request.form['chills']),
            int(request.form['body_pain']),
            int(request.form['fatigue']),
            int(request.form['rash']),
            float(request.form['temperature'])
        ]

        input_data = np.array([data])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        result = labels[prediction]
        confidence = round(proba[prediction] * 100, 2)

        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
