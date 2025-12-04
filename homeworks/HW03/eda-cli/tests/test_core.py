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
    
    # Проверяем, что флаг правильно определяет константный столбец
    assert flags["has_constant_columns"] == True
    assert "a" in flags["constant_columns_list"]