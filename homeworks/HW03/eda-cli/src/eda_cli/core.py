import pandas as pd
from eda_cli.core import compute_quality_flags

def test_constant_columns():
    """ТЕСТ ДЛЯ HW03: проверка новой эвристики - константные колонки"""
    df = pd.DataFrame({
        "a": [1, 1, 1], # Константный столбец
        "b": [1, 2, 3],
        "c": ["x", "y", "z"]
    })
   
    flags = compute_quality_flags(df)
   
    assert flags["has_constant_columns"] == True
    assert "a" in flags["constant_columns_list"]

def test_high_cardinality():
    """ТЕСТ ДЛЯ HW03: проверка новой эвристики - высокая кардинальность"""
    df = pd.DataFrame({
        "category": [f"cat_{i}" for i in range(25)], # 25 уникальных > 20
        "value": range(25)
    })
   
    flags = compute_quality_flags(df)
   
    assert flags["has_high_cardinality_categoricals"] == True
    assert "category" in flags["high_cardinality_columns"]

def test_high_missing_columns():
    """ТЕСТ ДЛЯ HW03: проверка логики с min_missing_share для проблемных колонок"""
    df = pd.DataFrame({
        "a": [1, 2, 3, None, None, None],  # 3 уникальных значения, 3/6 = 50% пропусков (если >0.5 — тест не пройдёт, но можно сделать 4 None)
        "b": [1, 2, 3, 4, 5, 6],           # 0% пропусков
    })
   
    flags = compute_quality_flags(df, min_missing_share=0.4)  # Порог 40%, чтобы 50% > 0.4
    
    assert flags["has_high_missing"] == True
    assert "a" in flags["high_missing_columns"]
    assert "b" not in flags["high_missing_columns"]
    
    # Score: 100 -25 (missing) -10 (high_missing) = 65
    assert flags["quality_score"] == 65