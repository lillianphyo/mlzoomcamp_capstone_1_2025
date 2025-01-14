import pytest
from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.model_training import train_model

def test_train_model():
    df = load_data()
    processed_df = preprocess_data(df)
    model, scaler = train_model(processed_df)
    assert model is not None
    assert scaler is not None