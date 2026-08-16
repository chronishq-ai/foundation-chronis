from pathlib import Path

from chronis_ml.loaders.base import DatasetLoader, LoaderConfig
from chronis_ml.loaders.example import ExampleLoader


def test_example_loader_follows_contract() -> None:
    loader: DatasetLoader = ExampleLoader()

    assert loader.dataset_name == "example"

    dataset = loader.load(LoaderConfig(source_path=Path("example")))

    assert dataset.records == ()
