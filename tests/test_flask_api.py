# tests/test_flask_api.py
import pytest
import requests
from api.app import app

@pytest.fixture
def client():
    # Create a test client for the Flask app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    # Test the /predict endpoint
    data = {"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json