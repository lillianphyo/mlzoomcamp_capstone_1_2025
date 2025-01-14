import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Data paths
    DATA_PATH = os.getenv("DATA_PATH", "data/")
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw/p2-east-1b.csv")
    PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed/processed_data.csv")
    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    MODEL_FILE = os.path.join(MODEL_PATH, "lstm_model/model.h5")
    SCALER_FILE = os.path.join(MODEL_PATH, "lstm_model/scaler.pkl")

    SEQUENCE_LENGTH = 10
    
    # Plot paths
    PLOTS_PATH = os.getenv("PLOTS_PATH", "plots/")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/pipeline.log")
    
    # Flask app
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))