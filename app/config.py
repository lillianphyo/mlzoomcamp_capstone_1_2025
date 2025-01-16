import os
from dotenv import load_dotenv

load_dotenv()

class Config:    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    MODEL_FILE = os.path.join(MODEL_PATH, "lstm_model/model.h5")
    SCALER_FILE = os.path.join(MODEL_PATH, "lstm_model/scaler.pkl")
    
    # Flask app
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))