from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained LSTM model
model = tf.keras.models.load_model("lstm_stock_model.h5")

# Load the scaler used for normalization
scaler = joblib.load("scaler.pkl")

def prepare_input(data, look_back=60):
    data = scaler.transform(data)
    sequence = []
    for i in range(len(data) - look_back, len(data)):
        sequence.append(data[i])
    return np.array(sequence).reshape(1, look_back, -1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        historical_data = pd.DataFrame(data['historical_prices'])
        input_sequence = prepare_input(historical_data)
        prediction = model.predict(input_sequence)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
