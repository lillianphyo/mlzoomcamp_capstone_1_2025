# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# import joblib
# import logging
# from ml_pipeline.config import Config

# # Set up logging
# logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def create_sequences(data, seq_length):
#     xs, ys = [], []
#     for i in range(len(data) - seq_length):
#         x = data[i:i + seq_length]
#         y = data[i + seq_length]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# def train_model(df):
#     """Train the LSTM model and save it."""
#     logger.info("Starting model training")
    
#     # Normalize the data
#     scaler = MinMaxScaler()
#     df['Cost'] = scaler.fit_transform(df[['Cost']])
    
#     # Create sequences
#     seq_length = 10
#     X, y = create_sequences(df['Cost'].values, seq_length)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
    
#     # Split into training and validation sets
#     split_index = int(len(X) * 0.8)
#     X_train, X_val = X[:split_index], X[split_index:]
#     y_train, y_val = y[:split_index], y[split_index:]
    
#     # Build the LSTM model
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(25))
#     model.add(Dense(1))
    
#     # Compile the model
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     # Train the model
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=100, verbose=1)
    
#     # Save the model and scaler
#     os.makedirs(os.path.dirname(Config.MODEL_FILE), exist_ok=True)
#     model.save(Config.MODEL_FILE.replace('.h5', '.keras'))
#     joblib.dump(scaler, Config.SCALER_FILE)
#     logger.info(f"Model saved to {Config.MODEL_FILE}")
#     logger.info(f"Scaler saved to {Config.SCALER_FILE}")
    
#     return model, scaler

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import logging
from ml_pipeline.config import Config

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model(processed_data):
    """Train the LSTM model and save it."""
    logger.info("Starting model training")
    
    # Extract processed data
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    y_train = processed_data['y_train']
    y_val = processed_data['y_val']
    scaler = processed_data['scaler']
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=100, verbose=1)
    
    # Save the model and scaler
    os.makedirs(os.path.dirname(Config.MODEL_FILE), exist_ok=True)
    model.save(Config.MODEL_FILE.replace('.h5', '.keras'))
    joblib.dump(scaler, Config.SCALER_FILE)
    logger.info(f"Model saved to {Config.MODEL_FILE}")
    logger.info(f"Scaler saved to {Config.SCALER_FILE}")
    
    return model, scaler, X_train, X_val, y_train, y_val