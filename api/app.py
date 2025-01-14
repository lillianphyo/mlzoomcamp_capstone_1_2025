# api/app.py
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from config.config import config

app = Flask(__name__)

# Load model and scaler from config
model = load_model(config.MODEL_PATH)
with open(config.SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data = np.array(data).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.reshape((1, 10, 1))
    prediction = model.predict(data_scaled)
    prediction = scaler.inverse_transform(prediction)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.DEBUG)