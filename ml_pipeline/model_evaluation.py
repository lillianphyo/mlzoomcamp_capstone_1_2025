# import numpy as np
# import logging
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from ml_pipeline.config import Config

# # Set up logging
# logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def evaluate_model(model, data, scaler):
#     """
#     Evaluate the model using the provided data.

#     Args:
#         model: Trained model to evaluate.
#         data: Tuple containing (X, y), where:
#             - X: Input features (numpy array or compatible).
#             - y: True target values (numpy array or compatible).
#         scaler: The scaler object used to normalize the data during training.

#     Returns:
#         dict: Dictionary containing evaluation metrics (MAE, MSE, RMSE).
#     """
#     logger.info("Evaluating model")

#     try:
#         # Unpack data
#         X, y = data

#         # Drop non-numeric columns from X (if X is a DataFrame)
#         if hasattr(X, 'select_dtypes'):
#             X = X.select_dtypes(include='number')  # Keep only numeric columns

#         # Ensure X and y are numpy arrays with numeric data types
#         X = np.array(X, dtype=np.float32)
#         y = np.array(y, dtype=np.float32)

#         # Validate input shapes
#         if X.size == 0:
#             raise ValueError("Input X is empty after dropping non-numeric columns.")
#         if len(X.shape) != 3:
#             raise ValueError(f"Input X must have shape (num_samples, seq_length, num_features). Got {X.shape}.")
#         if len(y.shape) != 1:
#             raise ValueError(f"Target y must have shape (num_samples,). Got {y.shape}.")

#         # Make predictions
#         y_pred = model.predict(X)

#         # Flatten y_pred if necessary
#         if len(y_pred.shape) > 1:
#             y_pred = y_pred.flatten()

#         # Inverse transform the predictions and true values to the original scale
#         y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
#         y_original = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(y_original, y_pred_original)
#         mse = mean_squared_error(y_original, y_pred_original)
#         rmse = np.sqrt(mse)

#         # Log metrics
#         logger.info(f"MAE: {mae}")
#         logger.info(f"MSE: {mse}")
#         logger.info(f"RMSE: {rmse}")

#         # Return metrics as a dictionary
#         return {
#             "MAE": mae,
#             "MSE": mse,
#             "RMSE": rmse
#         }

#     except Exception as e:
#         logger.error(f"Error during model evaluation: {e}")
#         raise

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