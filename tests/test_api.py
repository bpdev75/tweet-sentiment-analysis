import pytest
from fastapi.testclient import TestClient
from src.api.LogisticRegression.api import app

client = TestClient(app)

def get_sentiment(tweet):
    response = client.get(f"/predict?tweet={tweet}")
    assert response.status_code == 200
    return response.json().get("predicted class")

def test_predict_sentiment():
    assert get_sentiment("I love this product!") == "Positif"
    assert get_sentiment("I hate this product!") == "Négatif"