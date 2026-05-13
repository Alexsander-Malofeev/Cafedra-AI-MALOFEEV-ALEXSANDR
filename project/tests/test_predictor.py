from pathlib import Path

import pytest

from src.predictor import TicketClassifier


MODEL_PATH = Path("artifacts/ticket_classifier.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run python -m src.train before this test")
def test_predictor_returns_payment_category():
    predictor = TicketClassifier(model_path=MODEL_PATH)
    result = predictor.predict("Списали деньги два раза")

    assert result["predicted_category"] == "payment_issue"
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run python -m src.train before this test")
def test_predictor_asks_clarification_for_short_text():
    predictor = TicketClassifier(model_path=MODEL_PATH)
    result = predictor.predict("Плохо")

    assert result["category"] == "needs_clarification"
    assert result["needs_clarification"] is True
