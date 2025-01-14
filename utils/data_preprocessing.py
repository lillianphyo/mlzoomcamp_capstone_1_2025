# utils/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column, seq_length=10):
    scaler = MinMaxScaler()
    df[target_column] = scaler.fit_transform(df[[target_column]])
    
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(df[target_column].values, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    return X_train, X_val, y_train, y_val, scaler