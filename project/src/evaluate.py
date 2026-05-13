from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report

from src.config import load_config, resolve_project_path
from src.predictor import TicketClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on a CSV dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = resolve_project_path(args.data or config["paths"]["data_path"])
    dataset = pd.read_csv(data_path)

    predictor = TicketClassifier(config_path=args.config)
    predicted = [predictor.model.predict([text])[0] for text in dataset["text"]]
    print(classification_report(dataset["category"], predicted, zero_division=0))


if __name__ == "__main__":
    main()
