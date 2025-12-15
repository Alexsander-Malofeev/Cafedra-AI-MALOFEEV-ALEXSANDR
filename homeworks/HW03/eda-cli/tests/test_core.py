import pandas as pd
from eda_cli.core import compute_quality_flags

def test_constant_columns():
    """Тест для обнаружения константных столбцов."""
    df = pd.DataFrame({
        "a": [1, 1, 1],  # Константный столбец
        "b": [1, 2, 3],
        "c": ["x", "y", "z"]
    })
    
    flags = compute_quality_flags(df)
    
    assert flags["has_constant_columns"] == True
    assert "a" in flags["constant_columns_list"]

def test_high_cardinality():
    """Тест для обнаружения высокой кардинальности."""
    df = pd.DataFrame({
        "category": [f"cat_{i}" for i in range(25)],  # 25 уникальных значений > 20
        "value": range(25)
    })
    
    flags = compute_quality_flags(df)
    
    assert flags["has_high_cardinality_categoricals"] == True
    assert "category" in flags["high_cardinality_columns"]