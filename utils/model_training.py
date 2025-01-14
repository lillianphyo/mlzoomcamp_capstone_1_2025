# utils/model_training.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Import the data preprocessing functions
from utils.data_preprocessing import load_data, preprocess_data

def build_model(seq_length):
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=100):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Main script
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = "path/to/your/data.csv"  # Replace with the path to your dataset
    target_column = "your_target_column"  # Replace with the name of your target column
    seq_length = 10  # Sequence length for the LSTM model

    df = load_data(file_path)
    X_train, X_val, y_train, y_val, scaler = preprocess_data(df, target_column, seq_length)

    # Build and train the model
    model = build_model(seq_length)
    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=100)

    # Evaluate the model
    mae, mse, rmse = evaluate_model(model, X_val, y_val)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")