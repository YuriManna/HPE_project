from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os
import math
from datetime import datetime
import tensorflow as tf
from keras.models import load_model

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
    data = request.get_json(force=True)

    try:
        input_data = preprocess_input(data)
    except Exception as e:
        return jsonify({"error": f"Errore nel preprocessing: {str(e)}"}), 400

    if model_name == 'ffn_model':
        try:
            # Carica scalers
            x_scaler = joblib.load("./models/NN_x_scaler.pkl")
            y_scaler = joblib.load("./models/NN_y_scaler.pkl")

            # Carica modello keras
            model = load_model("./models/ffn_model.keras")

            # Standardizza input
            mask = [0,1,3,4,5,6,7,8,9]
            input_data[:, mask] = x_scaler.transform(input_data[:, mask])

            # Predici
            y_scaled = model.predict(input_data)
            print(y_scaled)
            # De-standardizza output
            y_pred = y_scaler.inverse_transform(y_scaled)[0][0]
            #y_pred = y_scaled * 693099.36

            return jsonify({"prediction": float(y_pred)})

        except Exception as e:
            return jsonify({"error": f"Errore FFN: {str(e)}"}), 500

    else:
        try:
            model_path = os.path.join("models", f"{model_name}.pkl")
            scaler_path = os.path.join("models", "linear_regression_scaler.pkl")

            print(f"Model path: {model_path}")
            if not os.path.exists(model_path):
                return jsonify({"error": "Model file not found"}), 404
            if not os.path.exists(scaler_path):
                return jsonify({"error": "Scaler file not found"}), 404


            # Caricamenti
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Standardizza solo le colonne corrette
            try:
                input_data[:, 3:5] = scaler.transform(input_data[:, 3:5])
            except Exception as e:
                return jsonify({"error": f"Errore nella standardizzazione: {str(e)}"}), 400

            # Predizione
            try:
                prediction = model.predict(input_data)
                return jsonify({"prediction": prediction.tolist()})
            except Exception as e:
                return jsonify({"error": f"Errore nella previsione: {str(e)}"}), 500


        except Exception as e:
            return jsonify({"error": f"Errore modello sklearn: {str(e)}"}), 500


import io
import base64
import matplotlib.pyplot as plt

@app.route('/forecast/<model_name>', methods=['POST'])
def forecast(model_name):
    data = request.get_json(force=True)
    n_weeks = int(data.get("weeks", 12))  # default: 12 weeks

    try:
        # === generate future dates ===
        start_date = pd.to_datetime(data["Date"])
        future_dates = pd.date_range(start_date, periods=n_weeks + 1, freq="W")[1:]

        predictions = []

        # === NN MODEL (FFN, LSTM, Transformer) ===
        if model_name == "ffn_model":
            try:
                # Load scalers
                x_scaler = joblib.load("./models/NN_x_scaler.pkl")
                y_scaler = joblib.load("./models/NN_y_scaler.pkl")

                # Load Keras model
                model = load_model("./models/ffn_model.keras")

                # Columns to scale
                mask = [0,1,3,4,5,6,7,8,9]

                # === loop over future weeks ===
                for date in future_dates:
                    fake = {
                        "Date": date,
                        "Store": data["Store"],
                        "Dept": data["Dept"],
                        "IsHoliday": 0,
                        "Temperature": data.get("Temperature", 20),
                        "Size": data.get("Size", 100000),
                        "Type": data.get("Type", "A")
                    }

                    X = preprocess_input(fake)
                    X[:, mask] = x_scaler.transform(X[:, mask])
                    y_scaled = model.predict(X)
                    y_pred = y_scaler.inverse_transform(y_scaled)[0][0]

                    predictions.append(float(y_pred))

                # --- CREA GRAFICO ---
                plt.figure(figsize=(8, 4))
                plt.plot(future_dates, predictions, marker='o')
                plt.title(f"Forecast {n_weeks} weeks")
                plt.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                return jsonify({"plot": plot_b64})

            except Exception as e:
                return jsonify({"error": f"Errore FFN Forecast: {str(e)}"}), 500


        # === SKLEARN MODELLI ===
        else:
            try:
                model_path = os.path.join("models", f"{model_name}.pkl")
                scaler_path = os.path.join("models", "linear_regression_scaler.pkl")

                if not os.path.exists(model_path):
                    return jsonify({"error": "Model file not found"}), 404
                if not os.path.exists(scaler_path):
                    return jsonify({"error": "Scaler file not found"}), 404

                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)

                for date in future_dates:
                    fake = {
                        "Date": date,
                        "Store": data["Store"],
                        "Dept": data["Dept"],
                        "IsHoliday": 0,
                        "Temperature": data.get("Temperature", 20),
                        "Size": data.get("Size", 100000),
                        "Type": data.get("Type", "A")
                    }

                    X = preprocess_input(fake)

                    # Apply scaler to same columns as in predict()
                    X[:, 3:5] = scaler.transform(X[:, 3:5])

                    y_pred = model.predict(X)[0]
                    predictions.append(float(y_pred))

                # --- CREA GRAFICO ---
                plt.figure(figsize=(8, 4))
                plt.plot(future_dates, predictions, marker='o')
                plt.title(f"Forecast {n_weeks} weeks")
                plt.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                return jsonify({"plot": plot_b64})

            except Exception as e:
                return jsonify({"error": f"Errore sklearn Forecast: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Errore Forecast: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
