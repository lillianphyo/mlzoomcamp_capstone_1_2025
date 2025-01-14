import pytest
from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.model_training import train_model
from ml_pipeline.model_evaluation import evaluate_model

def test_evaluate_model():
    df = load_data()
    processed_df = preprocess_data(df)
    model, scaler = train_model(processed_df)
    evaluate_model(model, (processed_df, scaler))