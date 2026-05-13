from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import resolve_project_path
from src.preprocessing import clean_text, count_words


def build_summary(data_path: str | Path) -> dict:
    path = resolve_project_path(data_path)
    dataset = pd.read_csv(path)
    dataset["clean_text"] = dataset["text"].apply(clean_text)
    dataset["word_count"] = dataset["text"].apply(count_words)

    summary = {
        "rows": int(len(dataset)),
        "categories": dataset["category"].value_counts().to_dict(),
        "min_words": int(dataset["word_count"].min()),
        "max_words": int(dataset["word_count"].max()),
        "mean_words": float(round(dataset["word_count"].mean(), 2)),
        "empty_texts": int((dataset["clean_text"] == "").sum()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple EDA summary for dataset")
    parser.add_argument("--data", default="data/tickets.csv")
    parser.add_argument("--output", default="artifacts/data_summary.json")
    args = parser.parse_args()

    summary = build_summary(args.data)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
