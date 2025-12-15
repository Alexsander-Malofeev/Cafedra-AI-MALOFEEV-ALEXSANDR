from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    compute_quality_flags_simple,  # HW03: наша функция
    missing_table,
    summarize_dataset,
)

app = FastAPI(
    title="AIE Dataset Quality API - HW04",
    version="0.2.0",
    description=(
        "HTTP-сервис для оценки качества датасетов. "
        "HW04: Добавлен новый эндпоинт /quality-flags-from-csv с эвристиками из HW03."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------

class HealthResponse(BaseModel):
    """Модель ответа для /health"""
    status: str
    timestamp: str
    service: str
    version: str

class QualityRequest(BaseModel):
    """Модель запроса для /quality"""
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )

class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""
    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


# ============================================================================
# HW04: НОВЫЕ МОДЕЛИ ДЛЯ НОВОГО ЭНДПОИНТА
# ============================================================================

class QualityFlagsResponse(BaseModel):
    """Ответ для нового эндпоинта /quality-flags-from-csv (HW04)"""
    request_id: str = Field(..., description="Уникальный идентификатор запроса")
    timestamp: str = Field(..., description="Время обработки запроса")
    latency_ms: float = Field(..., description="Время обработки в миллисекундах")
    dataset_info: dict = Field(..., description="Информация о датасете")
    quality_score: float = Field(..., description="Оценка качества 0-100")
    flags: dict = Field(..., description="Все флаги качества (включая HW03)")
    hw03_heuristics: dict = Field(..., description="Специфические флаги из HW03")


# ---------- Системный эндпоинт ----------

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality-hw04",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """Эндпоинт-заглушка, который принимает агрегированные признаки датасета."""

    start = time.time()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (time.time() - start) * 1000.0

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------

@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """Эндпоинт, который принимает CSV-файл и возвращает оценку качества данных."""

    start = time.time()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (time.time() - start) * 1000.0

    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool) and not key.startswith("has_") and not key.endswith("_list")
    }

    # Добавляем HW03 флаги
    flags_bool["has_constant_columns"] = flags_all.get("has_constant_columns", False)
    flags_bool["has_high_cardinality_categoricals"] = flags_all.get("has_high_cardinality_categoricals", False)

    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ============================================================================
# HW04: НОВЫЙ ЭНДПОИНТ /quality-flags-from-csv
# ============================================================================

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["hw04"],
    summary="Полный набор флагов качества из CSV (HW04)",
    description="""Возвращает полный набор флагов качества, включая новые эвристики из HW03:
    - Константные колонки (has_constant_columns)
    - Высокая кардинальность категориальных признаков (has_high_cardinality_categoricals)
    
    Использует доработки из HW03."""
)
async def quality_flags_from_csv(
    file: UploadFile = File(...),
    min_missing_share: float = 0.3
) -> QualityFlagsResponse:
    """
    НОВЫЙ ЭНДПОИНТ ДЛЯ HW04.
    
    Принимает CSV-файл и возвращает полный набор флагов качества,
    включая эвристики, добавленные в HW03.
    """
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    print(f"[HW04] Новый запрос к /quality-flags-from-csv: {request_id}")
    print(f"[HW04] Параметр min_missing_share: {min_missing_share}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть в формате CSV"
        )

    try:
        # Сохраняем временный файл
        temp_path = f"temp_{request_id}.csv"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Загружаем данные
        df = pd.read_csv(temp_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV-файл пуст")
        
        # Используем нашу функцию из HW03
        quality_flags = compute_quality_flags_simple(df, min_missing_share=min_missing_share)
        
        # Формируем детальную информацию
        dataset_info = {
            "filename": file.filename,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Собираем флаги
        all_flags = {
            "has_missing_values": quality_flags["has_missing_values"],
            "missing_percentage": quality_flags["missing_percentage"],
            "has_duplicates": quality_flags["has_duplicates"],
            "quality_score": quality_flags["quality_score"],
        }
        
        # HW03: Специфические флаги
        hw03_heuristics = {
            "has_constant_columns": quality_flags["has_constant_columns"],
            "constant_columns_list": quality_flags["constant_columns_list"],
            "has_high_cardinality_categoricals": quality_flags["has_high_cardinality_categoricals"],
            "high_cardinality_columns": quality_flags["high_cardinality_columns"],
            "min_missing_share_used": min_missing_share,
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Удаляем временный файл
        import os
        os.remove(temp_path)
        
        print(f"[HW04] Успешно обработан запрос {request_id}, latency: {latency_ms:.2f}ms")
        
        return QualityFlagsResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            latency_ms=round(latency_ms, 2),
            dataset_info=dataset_info,
            quality_score=quality_flags["quality_score"],
            flags=all_flags,
            hw03_heuristics=hw03_heuristics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[HW04] Ошибка обработки CSV: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV файла: {str(e)}"
        )


# ---------- Эндпоинт для информации о датасете ----------

@app.post("/dataset-info", tags=["info"])
async def dataset_info(file: UploadFile = File(...)):
    """Информация о датасете (размеры, типы данных)"""
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        df = pd.read_csv(file.file)
        
        info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "dataset": {
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            },
        }
        
        return info
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV файла: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)