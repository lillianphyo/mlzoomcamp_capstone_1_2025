# config/config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    # General settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/lstm_model.h5")
    SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/models/scaler.pkl")

    # Flask settings
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

    # BentoML settings
    BENTOML_MODEL_NAME = os.getenv("BENTOML_MODEL_NAME", "ec2_spot_forecast")
    BENTOML_SERVICE_NAME = os.getenv("BENTOML_SERVICE_NAME", "EC2SpotForecastService")

    # Saturn Cloud settings
    SATURN_CLOUD_API_KEY = os.getenv("SATURN_CLOUD_API_KEY", "")
    SATURN_CLOUD_PROJECT_ID = os.getenv("SATURN_CLOUD_PROJECT_ID", "")

# Load environment-specific configurations
config = Config()