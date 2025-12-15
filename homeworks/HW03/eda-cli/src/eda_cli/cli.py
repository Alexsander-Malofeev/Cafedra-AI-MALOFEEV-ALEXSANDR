import sys
import os
from .core import load_data, compute_quality_flags, get_basic_stats
from .viz import save_histograms

def overview(path: str):
    df = load_data(path)
    flags = compute_quality_flags(df)
    stats = get_basic_stats(df)
    
    print(f"Файл: {path}")
    print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"Качество данных: {flags['quality_score']}/100")
    
    print("\nПроблемы:")
    if flags['has_missing_values']:
        print(f"  ✗ Есть пропуски ({flags['missing_percentage']:.1f}%)")
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
    title: str = "Отчёт по анализу данных",
    top_k_categories: int = 10
):
    """Сгенерировать отчёт с новыми параметрами"""
    df = load_data(path)
    flags = compute_quality_flags(df, min_missing_share=min_missing_share)
    stats = get_basic_stats(df)
    
    os.makedirs(out_dir, exist_ok=True)
    save_histograms(df, out_dir, max_cols=max_hist_columns)
    
    # Собираем информацию о категориальных колонках (топ-K)
    categorical_info = {}
    for col in df.select_dtypes(include=["object"]).columns:
        top_values = df[col].value_counts().head(top_k_categories)
        if not top_values.empty:
            categorical_info[col] = top_values.to_dict()
    
    # Проблемные колонки по пропускам
    problem_columns = []
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > min_missing_share:
            problem_columns.append((col, missing_ratio))
    
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        # Используем title из параметров
        f.write(f"# {title}\n\n")
        
        f.write("## Основные характеристики\n")
        f.write(f"- Файл: {path}\n")
        f.write(f"- Строк: {df.shape[0]}\n")
        f.write(f"- Столбцов: {df.shape[1]}\n")
        f.write(f"- Качество данных: {flags['quality_score']}/100\n\n")
        
        f.write("## Настройки отчёта\n")
        f.write(f"- Макс. гистограмм: {max_hist_columns}\n")
        f.write(f"- Порог пропусков: {min_missing_share}\n")
        f.write(f"- Топ-K категорий: {top_k_categories}\n")
        f.write(f"- Заголовок: {title}\n\n")
        
        f.write("## Обнаруженные проблемы\n")
        if flags['has_missing_values']:
            f.write(f"- Есть пропуски ({flags['missing_percentage']:.1f}%)\n")
        if flags['has_duplicates']:
            f.write("- Есть дубликаты строк\n")
        if flags['has_constant_columns']:
            f.write(f"- Константные колонки: {flags['constant_columns_list']}\n")
        if flags['has_high_cardinality_categoricals']:
            f.write(f"- Высокая кардинальность: {flags['high_cardinality_columns']}\n")
        
        if problem_columns:
            f.write(f"\n## Колонки с пропусками > {min_missing_share}\n")
            for col, ratio in problem_columns:
                f.write(f"- {col}: {ratio:.1%}\n")
        
        if categorical_info:
            f.write(f"\n## Топ-{top_k_categories} категорий\n")
            for col, values in categorical_info.items():
                f.write(f"### {col}\n")
                for value, count in values.items():
                    f.write(f"- '{value}': {count}\n")
    
    print(f"✓ Отчёт создан: {report_path}")
    print(f"✓ Использованы параметры: max-hist-columns={max_hist_columns}, "
          f"min-missing-share={min_missing_share}, title='{title}', top-k-categories={top_k_categories}")

def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  eda-cli overview <файл.csv>")
        print("  eda-cli report <файл.csv> [опции]")
        print("\nОпции для report:")
        print("  --out-dir <папка>           Папка для отчёта (по умолчанию: reports)")
        print("  --max-hist-columns <N>      Макс. число гистограмм (по умолчанию: 4)")
        print("  --min-missing-share <N>     Порог пропусков (по умолчанию: 0.3)")
        print("  --title <текст>             Заголовок отчёта (по умолчанию: Отчёт по анализу данных)")
        print("  --top-k-categories <N>      Топ-K категорий (по умолчанию: 10)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "overview":
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к файлу")
            print("Использование: eda-cli overview <файл.csv>")
            sys.exit(1)
        overview(sys.argv[2])
    
    elif command == "report":
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к файлу")
            print("Использование: eda-cli report <файл.csv> [опции]")
            sys.exit(1)
        
        args = sys.argv[2:]
        input_csv = None
        out_dir = "reports"
        max_hist_columns = 4
        min_missing_share = 0.3
        title = "Отчёт по анализу данных"
        top_k_categories = 10
        
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
            elif args[i] == "--title" and i + 1 < len(args):
                title = args[i + 1]
                i += 2
            elif args[i] == "--top-k-categories" and i + 1 < len(args):
                top_k_categories = int(args[i + 1])
                i += 2
            elif not args[i].startswith("--"):
                input_csv = args[i]
                i += 1
            else:
                print(f"Неизвестная опция: {args[i]}")
                sys.exit(1)
        
        if not input_csv:
            print("Ошибка: укажите путь к CSV файлу")
            sys.exit(1)
        
        generate_report(
            path=input_csv,
            out_dir=out_dir,
            max_hist_columns=max_hist_columns,
            min_missing_share=min_missing_share,
            title=title,
            top_k_categories=top_k_categories
        )
    
    else:
        print(f"Неизвестная команда: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()