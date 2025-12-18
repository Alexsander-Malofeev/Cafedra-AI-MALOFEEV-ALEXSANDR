import pandas as pd

def load_data(path: str):
    return pd.read_csv(path)

def compute_quality_flags(df, min_missing_share: float = 0.3):
    flags = {
        "has_missing_values": df.isna().any().any(),
        "missing_percentage": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "has_duplicates": df.duplicated().any(),
    }
    
    # Эвристика 1: константные колонки
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns_list"] = constant_cols
    
    # Эвристика 2: высокая кардинальность категориальных признаков
    high_card_cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() > 20:
            high_card_cols.append(col)
    
    flags["has_high_cardinality_categoricals"] = len(high_card_cols) > 0
    flags["high_cardinality_columns"] = high_card_cols
    
    # Логика для --min-missing-share: проблемные колонки по пропускам
    missing_shares = df.isna().mean()
    high_missing_cols = [col for col in df.columns if missing_shares[col] > min_missing_share]
    flags["has_high_missing"] = len(high_missing_cols) > 0
    flags["high_missing_columns"] = high_missing_cols
    
    # Расчёт quality_score
    score = 100
    if flags["has_missing_values"]:
        score -= 25
    if flags["has_duplicates"]:
        score -= 25
    if flags["has_constant_columns"]:
        score -= 25
    if flags["has_high_cardinality_categoricals"]:
        score -= 25
    if flags["has_high_missing"]:
        score -= 10  # Дополнительный штраф
    
    flags["quality_score"] = max(0, score)
    return flags