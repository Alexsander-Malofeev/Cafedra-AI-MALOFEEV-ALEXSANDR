from src.generate_dataset import make_dataset


def test_dataset_has_required_columns_and_categories():
    dataset = make_dataset()
    assert {"text", "category"}.issubset(dataset.columns)
    assert len(dataset) >= 150
    assert dataset["category"].nunique() == 6
