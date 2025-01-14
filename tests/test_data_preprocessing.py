# tests/test_data_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data

def test_load_data():
    # Test loading a sample CSV file
    df = load_data("artifacts/data/p2-east-1b.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data():
    # Create a sample DataFrame
    data = {"0.25": [0.1, 0.2, 0.3, 0.4, 0.5]}
    df = pd.DataFrame(data)

    # Test preprocessing
    X_train, X_val, y_train, y_val, scaler = preprocess_data(df, "0.25", seq_length=2)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape == (3, 2, 1)  # Check shape after sequence creation
    assert y_train.shape == (3,)