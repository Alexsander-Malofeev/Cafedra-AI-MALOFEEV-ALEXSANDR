from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.predictor import TicketClassifier
from src.schemas import TicketRequest, TicketResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_COUNTER = Counter("ticket_predict_requests_total", "Total /predict requests")
ERROR_COUNTER = Counter("ticket_predict_errors_total", "Total /predict errors")
LATENCY_HISTOGRAM = Histogram("ticket_predict_latency_seconds", "Prediction request latency")

app = FastAPI(
    title="Ticket Category Classifier",
    description="Demo API for classifying support tickets by category.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> TicketClassifier:
    return TicketClassifier()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ticket-category-classifier",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, object]:
    try:
        predictor = get_predictor()
        return {"status": "ok", "model_loaded": True, "model_path": str(predictor.model_path)}
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "model_loaded": False, "error": str(exc)}


@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest) -> TicketResponse:
    REQUEST_COUNTER.inc()
    start_time = time.perf_counter()

    try:
        predictor = get_predictor()
        result = predictor.predict(request.text)
        logger.info(
            "Prediction: category=%s predicted=%s confidence=%.4f clarification=%s",
            result["category"],
            result["predicted_category"],
            result["confidence"],
            result["needs_clarification"],
        )
        return TicketResponse(**result)
    except FileNotFoundError as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        ERROR_COUNTER.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        LATENCY_HISTOGRAM.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
