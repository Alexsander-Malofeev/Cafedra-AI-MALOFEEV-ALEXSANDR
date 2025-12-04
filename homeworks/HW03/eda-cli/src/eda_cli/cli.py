import sys
import os
from pathlib import Path
from .core import load_data, compute_quality_flags, get_basic_stats
from .viz import save_histograms

def overview(path: str):
    """Показать общую информацию о датасете."""
    df = load_data(path)
    flags = compute_quality_flags(df)
    print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"Качество данных: {flags['quality_score']}/100")
    
    print("Проблемы:")
    if flags['has_missing_values']:
        print("  ✗ Есть пропуски")
    if flags['has_duplicates']:
        print("  ✗ Есть дубликаты строк")
    if flags['has_constant_columns']:
        print(f"  ✗ Константные колонки: {flags['constant_columns_list']}")
    if flags['has_high_cardinality_categoricals']:
        print(f"  ✗ Высокая кардинальность: {flags['high_cardinality_columns']}")

def generate_report(
    path: str,
    out_dir: str = "reports",
    max_hist_columns: int = 4,
    min_missing_share: float = 0.3,
):
    """Сгенерировать полный отчёт по анализу данных."""
    df = load_data(path)
    stats = get_basic_stats(df)
    
    # Используем min_missing_share
    flags = compute_quality_flags(df, min_missing_share=min_missing_share)
    
    # Создаем папку для отчета
    os.makedirs(out_dir, exist_ok=True)
    
    # Сохраняем гистограммы с параметром max_hist_columns
    save_histograms(df, out_dir, max_cols=max_hist_columns)
    
    # Генерируем отчет
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчёт по анализу данных\n\n")
        
        f.write("## Основные характеристики\n")
        f.write(f"- Строк: {stats['rows']}\n")
        f.write(f"- Столбцов: {stats['cols']}\n")
        f.write(f"- Качество данных: {flags['quality_score']}/100\n")
        f.write(f"- Порог пропусков: {min_missing_share}\n")
        f.write(f"- Макс. гистограмм: {max_hist_columns}\n\n")
        
        f.write("## Обнаруженные проблемы\n")
        if flags['has_missing_values']:
            f.write(f"- Есть пропуски ({flags['missing_percentage']:.1f}%)\n")
        if flags['has_duplicates']:
            f.write("- Есть дубликаты строк\n")
        if flags['has_constant_columns']:
            f.write(f"- Константные колонки: {flags['constant_columns_list']}\n")
        if flags['has_high_cardinality_categoricals']:
            f.write(f"- Высокая кардинальность: {flags['high_cardinality_columns']}\n")
    
    print(f"Отчёт создан: {report_path}")

def main():
    """Основная функция CLI."""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  eda-cli overview <файл.csv>")
        print("  eda-cli report <файл.csv> [--out-dir reports] [--max-hist-columns 4] [--min-missing-share 0.3]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "overview":
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к файлу")
            sys.exit(1)
        overview(sys.argv[2])
    
    elif command == "report":
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к файлу")
            sys.exit(1)
        
        # Простой парсинг аргументов
        args = sys.argv[2:]
        input_csv = None
        out_dir = "reports"
        max_hist_columns = 4
        min_missing_share = 0.3
        
        i = 0
        while i < len(args):
            if args[i] == "--out-dir" and i + 1 < len(args):
                out_dir = args[i + 1]
                i += 2
            elif args[i] == "--max-hist-columns" and i + 1 < len(args):
                max_hist_columns = int(args[i + 1])
                i += 2
            elif args[i] == "--min-missing-share" and i + 1 < len(args):
                min_missing_share = float(args[i + 1])
                i += 2
            elif not args[i].startswith("--"):
                input_csv = args[i]
                i += 1
            else:
                i += 1
        
        if not input_csv:
            print("Ошибка: укажите путь к CSV файлу")
            sys.exit(1)
        
        generate_report(
            path=input_csv,
            out_dir=out_dir,
            max_hist_columns=max_hist_columns,
            min_missing_share=min_missing_share,
        )
    
    else:
        print(f"Неизвестная команда: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()