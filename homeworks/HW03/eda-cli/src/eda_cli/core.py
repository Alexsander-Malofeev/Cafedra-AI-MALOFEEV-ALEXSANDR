import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def compute_quality_flags(df: pd.DataFrame, min_missing_share: float = 0.3) -> dict:
    flags = {
        "has_missing_values": df.isna().any().any(),
        "missing_percentage": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "has_duplicates": df.duplicated().any(),
    }

    # НОВАЯ ЭВРИСТИКА 1: константные колонки
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns_list"] = constant_cols

    # НОВАЯ ЭВРИСТИКА 2: высокая кардинальность категориальных признаков
    high_card_cols = []
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() > 20:  # Порог: более 20 уникальных значений
            high_card_cols.append(col)
    
    flags["has_high_cardinality_categoricals"] = len(high_card_cols) > 0
    flags["high_cardinality_columns"] = high_card_cols

    # Качество от 0 до 100
    score = 100
    if flags["has_missing_values"]: 
        score -= 25
    if flags["has_duplicates"]: 
        score -= 25
    if flags["has_constant_columns"]: 
        score -= 25
    if flags["has_high_cardinality_categoricals"]: 
        score -= 25
    
    # Учитываем порог пропусков
    if flags["missing_percentage"] > min_missing_share * 100:
        score -= 10
    
    flags["quality_score"] = max(0, score)

    return flags

def get_basic_stats(df: pd.DataFrame):
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isna().sum().to_dict(),
    }