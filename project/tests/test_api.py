from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.service import app


MODEL_PATH = Path("artifacts/ticket_classifier.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run python -m src.train before this test")
def test_api_predict():
    client = TestClient(app)
    response = client.post("/predict", json={"text": "Не могу войти в аккаунт"})

    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["predicted_category"] == "account_access"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run python -m src.train before this test")
def test_api_health():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()
