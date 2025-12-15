import matplotlib.pyplot as plt
import os
import pandas as pd

def save_histograms(df: pd.DataFrame, out_dir: str, max_cols: int = 4):
    """Сохраняет гистограммы с ограничением по количеству (новая опция HW03)"""
    os.makedirs(out_dir, exist_ok=True)
    numeric = df.select_dtypes(include="number")
    
    if numeric.empty:
        print("Нет числовых колонок для гистограмм")
        return
    
    # ============================================
    # ИСПОЛЬЗОВАНИЕ НОВОЙ ОПЦИИ HW03: max_hist_columns
    # ============================================
    cols = numeric.columns[:max_cols]
    print(f"Создаю гистограммы для {len(cols)} колонок (ограничение HW03: max_hist_columns={max_cols})")
    
    for col in cols:
        plt.figure(figsize=(8, 5))
        plt.hist(numeric[col].dropna(), bins=20, alpha=0.7)
        plt.title(f"Гистограмма {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/hist_{col}.png", dpi=100)
        plt.close()