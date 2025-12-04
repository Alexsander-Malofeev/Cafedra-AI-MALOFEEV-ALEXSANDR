import pandas as pd
from eda_cli.core import compute_quality_flags

def test_constant_columns():
    df = pd.DataFrame({"a": [5, 5, 5], "b": [1, 2, 3]})
    flags = compute_quality_flags(df)
    assert flags["has_constant_columns"] is True

def test_no_constant_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    flags = compute_quality_flags(df)
    assert flags["has_constant_columns"] is False