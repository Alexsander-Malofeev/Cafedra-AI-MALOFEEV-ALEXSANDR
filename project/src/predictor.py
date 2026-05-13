from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import get_model_path, load_config, resolve_project_path
from src.preprocessing import clean_text, is_too_short

CLARIFICATION_CATEGORY = "needs_clarification"

CLARIFICATION_MESSAGES = {
    "payment_issue": "Уточните, что именно произошло с оплатой: платеж не проходит, деньги списались или чек не пришел?",
    "refund_request": "Уточните, пожалуйста, что вы хотите вернуть: деньги, товар или отменить заказ?",
    "delivery_issue": "Уточните, пожалуйста, что именно не пришло: заказ, товар, письмо или код?",
    "account_access": "Уточните, пожалуйста, проблема со входом, паролем, кодом подтверждения или блокировкой аккаунта?",
    "technical_bug": "Уточните, пожалуйста, где возникла ошибка: на сайте, в приложении, при оплате или в личном кабинете?",
    "general_question": "Пожалуйста, опишите вопрос подробнее, чтобы его можно было правильно направить.",
}

DEFAULT_CLARIFICATION_MESSAGE = "Пожалуйста, опишите проблему подробнее."


class TicketClassifier:
    """Load trained model and make predictions for support ticket text."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config = load_config(config_path)
        configured_model_path = get_model_path(self.config)
        self.model_path = resolve_project_path(model_path) if model_path else configured_model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Run: python -m src.generate_dataset && python -m src.train"
            )

        self.model = joblib.load(self.model_path)
        model_config = self.config["model"]
        self.min_confidence = float(model_config["min_confidence"])
        self.short_text_min_words = int(model_config["short_text_min_words"])
        self.short_text_min_chars = int(model_config["short_text_min_chars"])

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "category": CLARIFICATION_CATEGORY,
                "predicted_category": None,
                "confidence": 0.0,
                "needs_clarification": True,
                "message": DEFAULT_CLARIFICATION_MESSAGE,
            }

        probabilities = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        best_index = int(np.argmax(probabilities))
        predicted_category = classes[best_index]
        confidence = float(probabilities[best_index])

        too_short = is_too_short(
            text,
            min_words=self.short_text_min_words,
            min_chars=self.short_text_min_chars,
        )
        low_confidence = confidence < self.min_confidence
        needs_clarification = bool(too_short or low_confidence)

        if needs_clarification:
            category = CLARIFICATION_CATEGORY
            message = CLARIFICATION_MESSAGES.get(predicted_category, DEFAULT_CLARIFICATION_MESSAGE)
        else:
            category = predicted_category
            message = "Категория определена."

        return {
            "category": category,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "needs_clarification": needs_clarification,
            "message": message,
        }
