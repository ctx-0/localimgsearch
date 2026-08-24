"""LocalImg Search - AI-powered local image search with CLIP."""

__version__ = "1.0.0"
__all__ = ["LocalImageSearch", "AVAILABLE_MODELS", "DEFAULT_MODEL"]

from localimgsearch.embed import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    LocalImageSearch,
)
