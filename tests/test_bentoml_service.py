# tests/test_bentoml_service.py
import pytest
import numpy as np
from services.bentoml_service import EC2SpotForecastService

def test_bentoml_service():
    # Create a dummy BentoML service instance
    svc = EC2SpotForecastService()

    # Test the predict method
    data = {"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    prediction = svc.predict(data)
    assert isinstance(prediction, list)