from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import load_config, resolve_project_path
from src.generate_dataset import save_dataset
from src.preprocessing import clean_text


def build_model(random_state: int = 42) -> Pipeline:
    """Build a small NLP pipeline.

    Word n-grams help with normal phrases.
    Character n-grams help with simple typos like "вирнуть" instead of "вернуть".
    """
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline([("features", features), ("classifier", classifier)])


def train_model(config: dict[str, Any], data_path: str | Path | None = None) -> dict[str, Any]:
    random_state = int(config["model"]["random_state"])
    test_size = float(config["model"]["test_size"])

    data_path = resolve_project_path(data_path or config["paths"]["data_path"])
    if not data_path.exists():
        print(f"Dataset not found: {data_path}. Generating synthetic dataset...")
        save_dataset(data_path)

    dataset = pd.read_csv(data_path)
    required_columns = {"text", "category"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["category"],
    )

    model = build_model(random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, zero_division=0)

    metrics = {
        "rows_total": int(len(dataset)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "classes": sorted(dataset["category"].unique().tolist()),
        "classification_report": report_dict,
    }

    model_path = resolve_project_path(config["paths"]["model_path"])
    metrics_path = resolve_project_path(config["paths"]["metrics_path"])
    report_path = resolve_project_path(config["paths"]["report_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")
    print(report_text)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ticket category classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--data", default=None, help="Optional dataset path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config=config, data_path=args.data)


if __name__ == "__main__":
    main()
