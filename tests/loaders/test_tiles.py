import pytest

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.loaders.tiles import TilesLoader


def test_tiles_missing_path_fails(tmp_path):
    loader = TilesLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(LoaderConfig(source_path=tmp_path / "missing"))


def test_tiles_does_not_guess_schema(tmp_path):
    loader = TilesLoader()

    with pytest.raises(NotImplementedError):
        loader.load(LoaderConfig(source_path=tmp_path))
