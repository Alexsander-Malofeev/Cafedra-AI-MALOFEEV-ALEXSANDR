from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags_simple,  # HW03: наша функция
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    # ============================================================================
    # HW03: НОВЫЕ ПАРАМЕТРЫ CLI
    # ============================================================================
    max_hist_columns: int = typer.Option(
        6, 
        help="Максимум числовых колонок для гистограмм. [HW03_NEW]"
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков для определения проблемных колонок. [HW03_NEW]"
    ),
    # ============================================================================
) -> None:
    """
    Сгенерировать полный EDA-отчёт.
    HW03: Добавлены параметры --max-hist-columns и --min-missing-share.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df)

    # 2. Качество в целом - HW03: используем нашу функцию с параметром min_missing_share
    quality_flags = compute_quality_flags_simple(df, min_missing_share=min_missing_share)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# EDA-отчёт\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        # ========================================================================
        # HW03: Показываем использованные параметры
        # ========================================================================
        f.write("## Настройки отчёта (HW03)\n")
        f.write(f"- Макс. гистограмм: **{max_hist_columns}**\n")
        f.write(f"- Порог пропусков: **{min_missing_share:.0%}**\n\n")
        # ========================================================================

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}/100**\n")
        f.write(f"- Доля пропусков: **{quality_flags['missing_percentage']:.1f}%**\n")
        
        # HW03: Добавляем информацию о новых эвристиках
        f.write("\n### Обнаруженные проблемы (HW03)\n")
        if quality_flags['has_missing_values']:
            f.write(f"- Есть пропуски ({quality_flags['missing_percentage']:.1f}%)\n")
        
        if quality_flags['has_constant_columns']:
            f.write(f"- Константные колонки: {quality_flags['constant_columns_list']}\n")
        
        if quality_flags['has_high_cardinality_categoricals']:
            f.write(f"- Высокая кардинальность: {quality_flags['high_cardinality_columns']}\n")
        
        f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"Построено гистограмм: {max_hist_columns} (ограничение HW03)\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки - HW03: используем max_hist_columns
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    
    # HW03: Выводим информацию о новых параметрах
    typer.echo(f"\n[HW03] Использованы новые параметры:")
    typer.echo(f"  --max-hist-columns={max_hist_columns}")
    typer.echo(f"  --min-missing-share={min_missing_share}")


# ============================================================================
# HW04: ИСПРАВЛЕНИЕ - добавляем функцию main()
# ============================================================================
def main():
    """Точка входа для CLI команды (указана в pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()