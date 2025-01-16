from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from config import Config

app = Flask(__name__)

# Load the model and scaler
model = load_model(Config.MODEL_FILE)
scaler = joblib.load(Config.SCALER_FILE)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions."""
    data = request.json['data']
    data = np.array(data).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.reshape((1, len(data_scaled), 1))
    
    prediction = model.predict(data_scaled)
    prediction = scaler.inverse_transform(prediction)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT)