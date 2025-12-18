import sys
import os
from .core import load_data, compute_quality_flags
from .viz import save_histograms

def overview(path: str):
    df = load_data(path)
    flags = compute_quality_flags(df)
    print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"Качество данных: {flags['quality_score']}/100")
   
    print("Проблемы:")
    if flags['has_missing_values']:
        print(" ✗ Есть пропуски")
    if flags['has_duplicates']:
        print(" ✗ Есть дубликаты строк")
    if flags['has_constant_columns']:
        print(f" ✗ Константные колонки: {flags['constant_columns_list']}")
    if flags['has_high_cardinality_categoricals']:
        print(f" ✗ Высокая кардинальность: {flags['high_cardinality_columns']}")

def generate_report(
    path: str,
    out_dir: str = "reports",
    max_hist_columns: int = 4,
    min_missing_share: float = 0.3
):
    df = load_data(path)
    flags = compute_quality_flags(df, min_missing_share=min_missing_share)
   
    os.makedirs(out_dir, exist_ok=True)
    save_histograms(df, out_dir, max_cols=max_hist_columns)
   
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчёт по анализу данных\n\n")
       
        f.write("## Основные характеристики\n")
        f.write(f"- Строк: {df.shape[0]}\n")
        f.write(f"- Столбцов: {df.shape[1]}\n")
        f.write(f"- Качество данных: {flags['quality_score']}/100\n")
        f.write(f"- Макс. гистограмм: {max_hist_columns} (новая опция HW03)\n")
        f.write(f"- Порог пропусков: {min_missing_share} (новая опция HW03)\n\n")
       
        f.write("## Обнаруженные проблемы\n")
        if flags['has_missing_values']:
            f.write(f"- Есть пропуски ({flags['missing_percentage']:.1f}%)\n")
        if flags['has_duplicates']:
            f.write("- Есть дубликаты строк\n")
        if flags['has_constant_columns']:
            f.write(f"- Константные колонки: {flags['constant_columns_list']}\n")
        if flags['has_high_cardinality_categoricals']:
            f.write(f"- Высокая кардинальность: {flags['high_cardinality_columns']}\n")
        # Добавлена секция для строгого соответствия ДЗ: список проблемных колонок по порогу
        if flags['has_high_missing']:
            f.write(f"- Проблемные колонки по пропускам (>{min_missing_share*100:.1f}%): {flags['high_missing_columns']}\n")
   
    print(f"Отчёт создан: {report_path}")
    print(f"Использованы новые опции HW03: max-hist-columns={max_hist_columns}, min-missing-share={min_missing_share}")

def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print(" eda-cli overview <файл.csv>")
        print(" eda-cli report <файл.csv> [опции]")
        print("\n" + "="*50)
        print("НОВЫЕ ОПЦИИ ДЛЯ HW03 (добавлены в задании):")
        print("="*50)
        print(" --max-hist-columns <N> Макс. число гистограмм (по умолчанию: 4)")
        print(" --min-missing-share <X> Порог пропусков для проблемных колонок (по умолчанию: 0.3)")
        print("="*50)
        print("\nСуществующие опции:")
        print(" --out-dir <папка> Папка для отчёта (по умолчанию: reports)")
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
       
        args = sys.argv[2:]
        input_csv = None
        out_dir = "reports"
        max_hist_columns = 4
        min_missing_share = 0.3
       
        i = 0
        while i < len(args):
            # Добавлена обработка --help для строгой проверки опций
            if args[i] == "--help":
                print("Использование: eda-cli report <файл.csv> [опции]")
                print("Опции:")
                print("  --out-dir <папка>          Папка для отчёта (по умолчанию: reports)")
                print("  --max-hist-columns <N>     Макс. число гистограмм для числовых колонок (по умолчанию: 4)")
                print("  --min-missing-share <X>    Порог доли пропусков для проблемных колонок (по умолчанию: 0.3)")
                sys.exit(0)
            elif args[i] == "--out-dir" and i + 1 < len(args):
                out_dir = args[i + 1]
                i += 2
            elif args[i] == "--max-hist-columns" and i + 1 < len(args):
                max_hist_columns = int(args[i + 1])
                print(f"Применена новая опция HW03: max-hist-columns = {max_hist_columns}")
                i += 2
            elif args[i] == "--min-missing-share" and i + 1 < len(args):
                min_missing_share = float(args[i + 1])
                print(f"Применена новая опция HW03: min-missing-share = {min_missing_share}")
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
        )
   
    else:
        print(f"Неизвестная команда: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()