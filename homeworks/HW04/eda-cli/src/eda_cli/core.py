from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам.
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


# ============================================================================
# HW03: НОВЫЕ ЭВРИСТИКИ КАЧЕСТВА ДАННЫХ
# ============================================================================

def _has_constant_columns(df: pd.DataFrame) -> bool:
    """Проверяет, есть ли константные колонки (все значения одинаковые)."""
    for col in df.columns:
        if df[col].nunique() <= 1:
            return True
    return False


def _get_constant_columns_list(df: pd.DataFrame) -> List[str]:
    """Возвращает список константных колонок."""
    return [col for col in df.columns if df[col].nunique() <= 1]


def _has_high_cardinality_categoricals(df: pd.DataFrame, threshold: int = 20) -> bool:
    """Проверяет, есть ли категориальные колонки с высокой кардинальностью."""
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            if s.nunique() > threshold:
                return True
    return False


def _get_high_cardinality_columns(df: pd.DataFrame, threshold: int = 20) -> List[str]:
    """Возвращает список колонок с высокой кардинальностью."""
    high_card_cols = []
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            if s.nunique() > threshold:
                high_card_cols.append(name)
    return high_card_cols


def compute_quality_flags(
    summary: DatasetSummary, 
    missing_df: pd.DataFrame,
    df: pd.DataFrame = None,  # HW03: добавляем df для новых эвристик
    min_missing_share: float = 0.3  # HW03: добавляем параметр
) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных.
    HW03: Добавлены 2 новые эвристики и параметр min_missing_share.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    # HW03: Используем min_missing_share вместо фиксированного 0.5
    flags["too_many_missing"] = max_missing_share > min_missing_share

    # HW03: Добавляем новые эвристики, если передан df
    if df is not None:
        flags["has_constant_columns"] = _has_constant_columns(df)
        flags["constant_columns_list"] = _get_constant_columns_list(df)
        flags["has_high_cardinality_categoricals"] = _has_high_cardinality_categoricals(df, threshold=20)
        flags["high_cardinality_columns"] = _get_high_cardinality_columns(df, threshold=20)

    # Простейший «скор» качества
    score = 1.0
    score -= max_missing_share  # чем больше пропусков, тем хуже
    
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    
    # HW03: Штрафы за новые проблемы
    if df is not None:
        if _has_constant_columns(df):
            score -= 0.15
        if _has_high_cardinality_categoricals(df, threshold=20):
            score -= 0.1

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# HW03: Новая функция для совместимости с нашим CLI
# ============================================================================

def compute_quality_flags_simple(
    df: pd.DataFrame, 
    min_missing_share: float = 0.3
) -> Dict[str, Any]:
    """
    Упрощённая версия для нашего CLI из HW03.
    Возвращает флаги в формате, совместимом с нашим кодом.
    """
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    flags = compute_quality_flags(summary, missing_df, df, min_missing_share)
    
    # Преобразуем в формат, ожидаемый нашим CLI
    result = {
        "has_missing_values": flags.get("too_many_missing", False),
        "missing_percentage": flags.get("max_missing_share", 0.0) * 100,
        "has_duplicates": False,  # В этой версии не считаем дубликаты
        "has_constant_columns": flags.get("has_constant_columns", False),
        "constant_columns_list": flags.get("constant_columns_list", []),
        "has_high_cardinality_categoricals": flags.get("has_high_cardinality_categoricals", False),
        "high_cardinality_columns": flags.get("high_cardinality_columns", []),
        "quality_score": flags.get("quality_score", 0.0) * 100,  # Конвертируем в 0-100
    }
    
    return result