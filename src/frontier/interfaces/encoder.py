"""
frontier/interfaces/encoder.py

Visual encoder metadata descriptors and interface contracts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualEncoderMetadata:
    """Canonical descriptor for a visual encoder. Referenced by all retrieval results."""
    model_id: str
    version: int
    dimension: int
    dtype: str           # e.g. "float32"
    normalization: str   # e.g. "L2" or "none"
    similarity_metric: str  # e.g. "cosine"


# Default metadata for the real CLIP encoder
DEFAULT_ENCODER_METADATA = VisualEncoderMetadata(
    model_id="openai/clip-vit-base-patch32",
    version=1,
    dimension=512,
    dtype="float32",
    normalization="L2",
    similarity_metric="cosine",
)
