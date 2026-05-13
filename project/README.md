# Сервис классификации пользовательских обращений

Проект для курса «Инженерия Искусственного Интеллекта».

Идея: сделать простой demo-сервис, который получает текст обращения пользователя и относит его к одной из фиксированных категорий. Это упрощённая BASIC-версия темы 5.1 «Классификация обращений/тикетов».

## 1. Паспорт проекта

- Название проекта: Сервис классификации пользовательских обращений
- Автор: Александр Малофеев
- Уровень: BASIC
- Домен: NLP, классификация текста
- Пользователь сервиса: сотрудник службы поддержки или helpdesk-система

Сервис принимает текст обращения, например:

```text
Списали деньги два раза
```

И возвращает категорию:

```json
{
  "category": "payment_issue",
  "predicted_category": "payment_issue",
  "confidence": 0.91,
  "needs_clarification": false,
  "message": "Категория определена."
}
```

Если сообщение слишком короткое или модель не уверена, сервис возвращает `needs_clarification`.

## 2. Категории

- `payment_issue` — проблемы с оплатой;
- `refund_request` — возврат денег или товара;
- `delivery_issue` — проблемы с доставкой;
- `account_access` — доступ к аккаунту;
- `technical_bug` — техническая ошибка;
- `general_question` — общий вопрос.

`needs_clarification` — это не обучающая категория, а решение сервиса: нужно попросить пользователя уточнить обращение.

## 3. Структура проекта

```text
project/
├── artifacts/                 # модель и метрики после обучения
├── configs/
│   ├── .env.example            # пример переменных окружения
│   └── config.yaml             # настройки проекта
├── data/
│   ├── README.md
│   └── tickets.csv             # синтетический датасет после генерации
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── generate_dataset.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── schemas.py
│   ├── service.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_dataset.py
│   ├── test_predictor.py
│   └── test_preprocessing.py
├── Dockerfile
├── README.md
├── report.md
├── requirements.txt
└── self-checklist.md
```

## 4. Установка

Перейти в папку проекта:

```bash
cd project
```

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Быстрый запуск

Сгенерировать синтетический датасет:

```bash
python -m src.generate_dataset
```

Посмотреть краткую сводку по данным:

```bash
python -m src.eda
```

Обучить модель:

```bash
python -m src.train
```

Запустить API:

```bash
uvicorn src.service:app --reload
```

После запуска открыть Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## 6. Пример запроса

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Не могу войти в аккаунт\"}"
```

Пример ответа:

```json
{
  "category": "account_access",
  "predicted_category": "account_access",
  "confidence": 0.93,
  "needs_clarification": false,
  "message": "Категория определена."
}
```

Пример короткого сообщения:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Плохо\"}"
```

В этом случае сервис может попросить уточнение:

```json
{
  "category": "needs_clarification",
  "predicted_category": "general_question",
  "confidence": 0.34,
  "needs_clarification": true,
  "message": "Пожалуйста, опишите вопрос подробнее, чтобы его можно было правильно направить."
}
```

## 7. API endpoints

- `GET /` — информация о сервисе;
- `GET /health` — проверка состояния сервиса и модели;
- `POST /predict` — классификация обращения;
- `GET /metrics` — базовые метрики Prometheus.

## 8. Тесты

```bash
pytest tests
```

Если модель ещё не обучена, тесты API и predictor будут пропущены. Чтобы проверить всё полностью, сначала выполните:

```bash
python -m src.generate_dataset
python -m src.train
pytest tests
```

## 9. Docker

Собрать образ:

```bash
docker build -t ticket-classifier .
```

Запустить контейнер:

```bash
docker run -p 8000:8000 ticket-classifier
```

После этого API будет доступен по адресу:

```text
http://127.0.0.1:8000/docs
```

## 10. Ограничения

- Датасет синтетический, поэтому качество на реальных обращениях может быть ниже.
- Сервис не пытается понимать все возможные пользовательские формулировки.
- Короткие сообщения вроде «Плохо» или «Ошибка» обрабатываются осторожно: сервис просит уточнить тему.
- Обработка опечаток реализована базово через символьные признаки и несколько примеров с ошибками в датасете.
