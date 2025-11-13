from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os
import math
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# === Funzione helper per convertire i dati utente nelle feature del modello ===
def preprocess_input(data):
    """
    Converte i dati dell'utente nelle feature usate dal modello.
    """
    date = pd.to_datetime(data["Date"])
    year_start = datetime(date.year, 1, 1)
    days_since_start = (date - year_start).days
    years_since_start = (date.year - 2010) + days_since_start / 365.0

    type_a = 1 if data["Type"] == "A" else 0
    type_b = 1 if data["Type"] == "B" else 0
    type_c = 1 if data["Type"] == "C" else 0

    month_sin = math.sin(2 * math.pi * date.month / 12)
    month_cos = math.cos(2 * math.pi * date.month / 12)
    day_sin = math.sin(2 * math.pi * date.day / 31)
    day_cos = math.cos(2 * math.pi * date.day / 31)

    features = [
        data["Store"],
        data["Dept"],
        data["IsHoliday"],
        data["Temperature"],
        data["Size"],
        years_since_start,
        month_sin,
        month_cos,
        day_sin,
        day_cos,
        type_a,
        type_b,
        type_c,
    ]

    return np.array(features).reshape(1, -1)



# === Endpoint Flask ===
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    model_path = os.path.join("models", f"{model_name}.pkl")

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    model = joblib.load(model_path)
    data = request.get_json(force=True)

    try:
        input_data = preprocess_input(data)
    except Exception as e:
        return jsonify({"error": f"Errore nel preprocessing: {str(e)}"}), 400

    # Previsione
    try:
        prediction = model.predict(input_data)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": f"Errore nella previsione: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
