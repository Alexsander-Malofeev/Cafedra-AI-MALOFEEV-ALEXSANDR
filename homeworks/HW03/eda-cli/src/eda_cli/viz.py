import matplotlib.pyplot as plt
import os

def save_histograms(df: pd.DataFrame, out_dir: str, max_cols: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    numeric = df.select_dtypes(include="number")
    cols = numeric.columns[:max_cols]
    
    for col in cols:
        plt.figure(figsize=(8, 5))
        plt.hist(numeric[col].dropna(), bins=30, alpha=0.7, color="#4e79a7")
        plt.title(f"Гистограмма {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/hist_{col}.png", dpi=200, bbox_inches="tight")
        plt.close()