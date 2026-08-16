"""TILES-2018 dataset loader."""

from __future__ import annotations

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.schema.models import ChronisDataset


class TilesLoader:
    """Load approved TILES-2018 data into ChronisDataset."""

    @property
    def dataset_name(self) -> str:
        return "tiles_2018"

    def load(self, config: LoaderConfig) -> ChronisDataset:
        """Load TILES-2018.

        The exact parsing implementation must follow the approved
        TILES-2018 source layout and field mapping.
        """

        if not config.source_path.exists():
            raise FileNotFoundError(
                f"TILES source path does not exist: "
                f"{config.source_path}"
            )

        raise NotImplementedError(
            "TILES-2018 field mapping has not yet been configured."
        )