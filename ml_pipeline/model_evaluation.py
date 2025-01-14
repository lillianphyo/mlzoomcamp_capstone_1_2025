from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
from ml_pipeline.config import Config

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model, X_val, y_val, scaler):
    """Evaluate the LSTM model on the validation set."""
    logger.info("Evaluating the model")
    
    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    
    # Inverse transform the predictions and actual values
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    logger.info("Evaluation Metrics:")
    logger.info(f"MAE: {mae}")
    logger.info(f"MSE: {mse}")
    logger.info(f"RMSE: {rmse}")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }