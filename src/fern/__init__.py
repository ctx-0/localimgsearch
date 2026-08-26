"""Fern - AI-powered local image search with CLIP."""

__version__ = "1.0.0"
__all__ = ["LocalImageSearch", "AVAILABLE_MODELS", "DEFAULT_MODEL"]

from fern.embed import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    LocalImageSearch,
)
