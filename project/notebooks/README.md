# Ноутбуки

В эту папку можно добавить ноутбук с EDA и экспериментами, если преподаватель попросит.

Минимальный сценарий без ноутбука уже есть в коде:

```bash
python -m src.generate_dataset
python -m src.eda
python -m src.train
```

Скрипт `src/eda.py` сохраняет краткую сводку по данным в `artifacts/data_summary.json`.
