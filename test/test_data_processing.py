import pytest
from ml_pipeline.data_processing import load_data, preprocess_data

def test_load_data():
    df = load_data()
    assert df is not None

def test_preprocess_data():
    df = load_data()
    processed_df = preprocess_data(df)
    assert processed_df is not None