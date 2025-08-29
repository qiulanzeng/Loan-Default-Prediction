import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
from prediction import Predictor
import subprocess

app = Flask(__name__)

# Instantiate Predictor
predictor = Predictor()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/train', methods=['GET'])
def training():
    try:
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True, check=True)
        return Response(f"Training successful!\n\n{result.stdout}", mimetype='text/plain')
    except subprocess.CalledProcessError as e:
        return Response(f"Training failed!\n\n{e.stderr}", mimetype='text/plain')
    
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input JSON:", data)

    # Convert JSON to DataFrame (column order must match training)
    df = pd.DataFrame([data])  # shape (1, n_features)
    print("Converted DataFrame:\n", df)

    output = predictor.predict_class(df)
    print("Prediction:", output[0])

    return jsonify({'prediction': int(output[0])})

@app.route('/predict', methods=['POST'])
def predict():
    raw_data = request.form.to_dict()

    # Convert empty strings to np.nan, cast known numeric fields
    numeric_fields = ['rate_of_interest', 'Interest_rate_spread', 'Credit_Score', 'dtir1', 'loan_amount', 'Upfront_charges', 'term', 'property_value', 'income', 'LTV']
    cleaned_data = {}
    for key, value in raw_data.items():
        if value == '':
            cleaned_data[key] = np.nan
        elif key in numeric_fields:
            try:
                cleaned_data[key] = float(value)
            except ValueError:
                cleaned_data[key] = np.nan  # fallback if conversion fails
        else:
            cleaned_data[key] = value

    df = pd.DataFrame([cleaned_data])

    output = predictor.predict_class(df)[0]
    return render_template("home.html", prediction_text=f"Predict results: The default status is {output}")
if __name__ == "__main__":
    app.run(debug=True)
