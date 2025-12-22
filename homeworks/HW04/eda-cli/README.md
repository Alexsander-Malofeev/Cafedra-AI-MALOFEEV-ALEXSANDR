# eda-cli — HW04

Мини‑утилита для простого EDA + HTTP API (FastAPI). Проект — домашнее задание HW04: поверх HW03 добавлен HTTP‑сервис и новый эндпоинт, использующий эвристики из HW03.

---

## Что важно для проверки (чеклист)

* В README явно указано, как запустить uvicorn (строка с `uvicorn` и `eda_cli.api:app` есть).
* В README явно указано и показано как использовать дополнительный эндпоинт: **`/quality-flags-from-csv`**.
* CLI, тесты и HTTP‑сервис можно запустить стандартными командами (см. раздел «Запуск»).

---

## Требования

Python >= 3.8
Проект использует `uv` (uvtool / hatch+uv wrapper), зависимости указаны в `pyproject.toml`:

* pandas
* matplotlib
* typer
* fastapi
* uvicorn[standard]
* python-multipart
* pytest

---

## Структура репозитория (важные файлы)

```
homeworks/
  HW04/
    eda-cli/
      pyproject.toml
      README.md        <- этот файл
      src/eda_cli/
        api.py         <- FastAPI приложение (eda_cli.api:app)
        cli.py
        core.py
        viz.py
      data/
        example.csv
      tests/
        test_core.py
```

---

## Быстрый старт (локально)

1. Перейдите в папку проекта:

```bash
cd homeworks/HW04/eda-cli
```

2. Синхронизируйте окружение и установите зависимости (через `uv` wrapper):

```bash
uv sync
```

3. Проверка CLI (пример):

```bash
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_example
```

4. Запуск тестов:

```bash
uv run pytest -q
```

5. Запуск HTTP‑сервиса (обратите внимание: **в команде явно указан `uvicorn` и модуль `eda_cli.api:app`**):

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

> Альтернатива без `uv` (если у вас установлен uvicorn в окружении):
>
> ```bash
> uvicorn eda_cli.api:app --reload --port 8000
> ```

---

## Документация API

После запуска доступна автодокументация: `http://127.0.0.1:8000/docs`

### 1) GET `/health`

Простейшая проверка доступности сервиса.

**Пример:**

```bash
curl -s http://127.0.0.1:8000/health
```

**Ожидаемый ответ (пример):**

```json
{ "status": "ok", "service": "dataset-quality-hw04", "version": "0.2.0" }
```

---

### 2) POST `/quality` (по агрегированным признакам)

Принимает JSON с агрегированными метриками (см. модель `QualityRequest`). Используется как заглушка для быстрой оценки.

**Пример:**

```bash
curl -s -X POST http://127.0.0.1:8000/quality \
  -H 'Content-Type: application/json' \
  -d '{"n_rows":1000, "n_cols":10, "max_missing_share":0.05, "numeric_cols":8, "categorical_cols":2}'
```

---

### 3) POST `/quality-from-csv` (EDA + качество)

Принимает CSV (multipart/form-data, поле `file`), использует `summarize_dataset`, `missing_table` и `compute_quality_flags` и возвращает `QualityResponse`.

**Пример (curl):**

```bash
curl -s -F "file=@data/example.csv" http://127.0.0.1:8000/quality-from-csv
```

---

### 4) POST **`/quality-flags-from-csv`** — *НОВЫЙ эндпоинт HW04* (обязательно)

**Путь:** `/quality-flags-from-csv` (см. `src/eda_cli/api.py`).

Что делает:

* Принимает CSV (multipart/form-data, поле `file`).
* Опциональный параметр запроса `min_missing_share` (float, по умолчанию `0.3`).
* Внутри использует функцию из HW03: `compute_quality_flags_simple(df, min_missing_share=...)`.
* Возвращает детальный набор флагов качества, включая специфические эвристики из HW03: `has_constant_columns`, `has_high_cardinality_categoricals`, списки колонок и т.п.

**Пример (curl):**

```bash
curl -s -F "file=@data/example.csv" -F "min_missing_share=0.25" http://127.0.0.1:8000/quality-flags-from-csv
```

**Ключевые поля в ответе:**

* `request_id` — uuid запроса
* `timestamp` — время обработки
* `latency_ms` — задержка
* `dataset_info` — basic info (filename, n_rows, n_cols, columns, dtypes)
* `quality_score` — оценка (0–100)
* `flags` — основные флаги
* `hw03_heuristics` — флаги, добавленные в HW03 (например `has_constant_columns`, `constant_columns_list`, `has_high_cardinality_categoricals`, `high_cardinality_columns`)

> В README обязательно упоминается этот путь `/quality-flags-from-csv` — это то замечание проверяющего, которое нужно закрыть.

---

## Примеры использования из Python (requests)

```python
import requests

url = "http://127.0.0.1:8000/quality-flags-from-csv"
with open("data/example.csv", "rb") as f:
    files = {"file": ("example.csv", f)}
    r = requests.post(url, files=files, data={"min_missing_share": 0.25})
    print(r.json())
```

---

## Полезные команды для проверки (резюме)

```bash
# синхронизация окружения
uv sync

# CLI: быстрый обзор
uv run eda-cli overview data/example.csv

# CLI: генерация отчёта
uv run eda-cli report data/example.csv --out-dir reports_example

# тесты
uv run pytest -q

# запуск сервера (обязательно: 'uvicorn' и 'eda_cli.api:app' присутствуют в команде)
uv run uvicorn eda_cli.api:app --reload --port 8000
```

---

## Что можно улучшить дополнительно (опционально)

* Добавить клиент‑скрипт `scripts/client.py`, который делает набор вызовов и суммирует latencies.
* Реализовать простой `/metrics` endpoint для статистики по запросам.
* Включить структурированное логирование (JSON) в `api.py`.

---

Если надо — могу подправить текст README под конкретный стиль проверяющего или добавить примеры ожидаемых ответов в полном виде.
