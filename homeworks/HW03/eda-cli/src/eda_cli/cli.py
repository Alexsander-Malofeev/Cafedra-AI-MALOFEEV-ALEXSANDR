import tyro
import os
from jinja2 import Template
from .core import load_data, compute_quality_flags, get_basic_stats
from .viz import save_histograms

def overview(path: str):
    df = load_data(path)
    flags = compute_quality_flags(df)
    print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"Quality score: {flags['quality_score']}/100")
    print("Проблемы:", {k: v for k, v in flags.items() if not k.endswith('_list')})

def generate_report(
    path: str,
    out_dir: str = "reports",
    title: str = "EDA Отчёт по датасету",
    max_hist_columns: int = 4,
):
    df = load_data(path)
    stats = get_basic_stats(df)
    flags = compute_quality_flags(df)
    
    os.makedirs(out_dir, exist_ok=True)
    save_histograms(df, out_dir, max_cols=max_hist_columns)
    
    numeric_cols = df.select_dtypes(include="number").columns[:max_hist_columns].tolist()

    template = Template("""# {{ title }}

## Основные характеристики
- Строк: {{ stats.rows }}
- Столбцов: {{ stats.cols }}
- Quality score: {{ flags.quality_score }}/100

## Обнаруженные проблемы
{% if flags.has_missing_values %}• Есть пропуски ({{ "%.2f"|format(flags.missing_percentage) }}% от всех ячеек){% endif %}
{% if flags.has_duplicates %}• Есть дубли строк{% endif %}
{% if flags.has_constant_columns %}• Константы колонки: {{ flags.constant_columns_list }}{% endif %}
{% if flags.has_high_cardinality_categoricals %}• Слишком много уникальных значений: {{ flags.high_cardinality_columns }}{% endif %}

## Гистограммы (показано {{ max_hist_columns }} числовых колонок)
{% for col in numeric_cols %}
![{{ col }}]({{ out_dir }}/hist_{{ col }}.png)
{% endfor %}
""")

    report = template.render(
        title=title,
        stats=stats,
        flags=flags,
        out_dir=out_dir,
        max_hist_columns=max_hist_columns,
        numeric_cols=numeric_cols,
    )

    with open(f"{out_dir}/report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Отчёт успешно создан → {out_dir}/report.md")

def main():
    tyro.cli({
        "overview": overview,
        "report": generate_report,
    })

if __name__ == "__main__":
    main()