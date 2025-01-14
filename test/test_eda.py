import pytest
from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.eda import perform_eda

def test_perform_eda():
    df = load_data()
    processed_df = preprocess_data(df)
    perform_eda(processed_df)