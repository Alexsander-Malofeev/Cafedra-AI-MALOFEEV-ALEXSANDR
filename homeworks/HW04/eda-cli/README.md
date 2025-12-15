# eda-cli — домашнее задание HW04

EDA утилита с HTTP API на FastAPI. Основана на коде преподавателя (вашеим из Гита) с интеграцией изменений из HW03.

## Возможности

### CLI (с изменениями из HW03)
```bash
# Общий обзор
uv run eda-cli overview data/example.csv

# Полный отчёт с новыми параметрами HW03
uv run eda-cli report data/example.csv \
  --out-dir reports \
  --max-hist-columns 3 \
  --min-missing-share 0.2