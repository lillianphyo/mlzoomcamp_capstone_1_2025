import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from ml_pipeline.config import Config

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    """Load raw data."""
    logger.info("Loading raw data from %s", Config.RAW_DATA_PATH)
    df = pd.read_csv(Config.RAW_DATA_PATH)
    return df

def preprocess_data(df):
    """Preprocess the data for LSTM modeling."""
    logger.info("Preprocessing data")
    
    # Drop missing values
    df = df.dropna()
    logger.info("Dropped missing values")

    # Convert datetime column to timestamp (if applicable)
    if 'datetime_column' in df.columns:  # Replace 'datetime_column' with the actual column name
        logger.info("Converting datetime column to timestamp")
        df['timestamp'] = pd.to_datetime(df['datetime_column']).astype(int) / 10**9  # Convert to Unix time
        df = df.drop(columns=['datetime_column'])  # Drop the original datetime column
    
    # Normalize the 'Cost' column
    scaler = MinMaxScaler()
    df['Cost'] = scaler.fit_transform(df[['Cost']])
    logger.info("Normalized the 'Cost' column using MinMaxScaler")
    
    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(df['Cost'].values, Config.SEQUENCE_LENGTH)
    logger.info(f"Created sequences with sequence length {Config.SEQUENCE_LENGTH}")
    
    # Reshape X to 3D array for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    logger.info("Reshaped X for LSTM input")
    
    # Split into training and validation sets
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    logger.info("Split data into training and validation sets")
    
    # Save processed data (optional)
    processed_data = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'scaler': scaler
    }
    logger.info(f"Processed data ready for model training")
    
    return processed_data