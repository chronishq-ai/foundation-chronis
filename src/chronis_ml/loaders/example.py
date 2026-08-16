"""Example implementation of the loader contract.

This is only a contract test/reference. It is not a TILES-2018 or GLOBEM
implementation.
"""

from __future__ import annotations

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.schema.models import ChronisDataset


class ExampleLoader:
    """Minimal loader skeleton."""

    @property
    def dataset_name(self) -> str:
        return "example"

    def load(self, config: LoaderConfig) -> ChronisDataset:
        return ChronisDataset.from_records(())
