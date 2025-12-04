# eda-cli — HW03 (Александр Малофеев)

Мини-приложение для быстрого EDA.

## Команды
uv run eda-cli overview data/example.csv

uv run eda-cli report data/example.csv --out-dir reports --title "Мой анализ данных" --max-hist-columns 5

## Что нового (для HW03)
- Две новые эвристики качества: константные колонки и вроде высокая кардинальность
- Новые параметры CLI: `--title`, `--max-hist-columns`
- Тесты на новые эвристикиcd homeworks/HW03/eda-cli
