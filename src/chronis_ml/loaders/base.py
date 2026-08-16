from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chronis_ml.schema.models import ChronisDataset


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    source_path: Path
    user_ids: tuple[str, ...] | None = None
    strict: bool = True


class DatasetLoader(Protocol):
    @property
    def dataset_name(self) -> str: ...

    def load(self, config: LoaderConfig) -> ChronisDataset: ...
