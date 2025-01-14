# tests/test_model_training.py
import pytest
import numpy as np
from utils.model_training import build_model, train_model, evaluate_model

def test_build_model():
    # Test building the LSTM model
    model = build_model(seq_length=10)
    assert model is not None
    assert len(model.layers) == 5  # Check the number of layers

def test_train_model():
    # Create dummy data
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.rand(20)

    # Test model training
    model = build_model(seq_length=10)
    trained_model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=1, batch_size=10)
    assert trained_model is not None
    assert "loss" in history.history

def test_evaluate_model():
    # Create dummy data
    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.rand(20)

    # Test model evaluation
    model = build_model(seq_length=10)
    mae, mse, rmse = evaluate_model(model, X_val, y_val)
    assert isinstance(mae, float)
    assert isinstance(mse, float)
    assert isinstance(rmse, float)