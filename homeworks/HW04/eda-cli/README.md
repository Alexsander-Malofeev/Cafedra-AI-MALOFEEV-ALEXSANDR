# eda-cli (HW04)

Проект для домашнего задания HW04. Представляет собой небольшую утилиту для EDA (exploratory data analysis) с CLI‑интерфейсом и HTTP API на FastAPI.

В HW04 поверх функциональности HW03 был добавлен HTTP‑сервис и дополнительный эндпоинт, использующий эвристики качества данных из HW03.

---

## Требования

* Python >= 3.8
* Менеджер окружений `uv`

Основные зависимости указаны в `pyproject.toml` (pandas, fastapi, typer, uvicorn, pytest и др.).

---

## Структура проекта

```
HW04/eda-cli/
  pyproject.toml
  README.md
  src/eda_cli/
    api.py        # FastAPI приложение
    cli.py        # CLI интерфейс
    core.py       # EDA и расчёт метрик
    viz.py        # Визуализация
  data/
    example.csv
  tests/
    test_core.py
```

---

## Установка и запуск

Перейдите в директорию проекта:

```bash
cd homeworks/HW04/eda-cli
```

Установите зависимости:

```bash
uv sync
```

### CLI

Пример быстрого обзора датасета:

```bash
uv run eda-cli overview data/example.csv
```

Генерация отчёта:

```bash
uv run eda-cli report data/example.csv --out-dir reports_example
```

### Тесты

```bash
uv run pytest -q
```

---

## HTTP API

### Запуск сервера

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска доступна документация Swagger:

```
http://127.0.0.1:8000/docs
```

---

### GET `/health`

Проверка доступности сервиса.

```bash
curl http://127.0.0.1:8000/health
```

---

### POST `/quality-from-csv`

Принимает CSV‑файл и возвращает базовую информацию о датасете и агрегированную оценку качества.

Пример:

```bash
curl -F "file=@data/example.csv" http://127.0.0.1:8000/quality-from-csv
```

---

### POST `/quality-flags-from-csv`

Дополнительный эндпоинт HW04.

* Принимает CSV‑файл (multipart/form-data, поле `file`).
* Использует эвристики качества данных, реализованные в HW03.
* Поддерживает параметр `min_missing_share` (по умолчанию 0.3).

Пример вызова:

```bash
curl -F "file=@data/example.csv" -F "min_missing_share=0.25" \
  http://127.0.0.1:8000/quality-flags-from-csv
```

В ответе возвращаются:

* базовая информация о датасете;
* числовая оценка качества;
* флаги качества данных;
* результаты эвристик HW03 (например, константные колонки, высокая кардинальность категориальных признаков).

---

## Примечания

* CLI и HTTP‑сервис используют общий код анализа данных.
* Проект предназначен для учебных целей и демонстрации простого ML/EDA пайплайна.
