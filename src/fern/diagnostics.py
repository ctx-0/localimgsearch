"""Control noisy dependency diagnostics without hiding application failures."""

from __future__ import annotations

import logging
import os
import sys
import warnings


_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEPENDENCY_LOGGERS = (
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
)


def debug_enabled() -> bool:
    """Return whether verbose dependency diagnostics were explicitly requested."""

    value = os.environ.get("FERN_DEBUG", "").strip().lower()
    return value in _TRUE_VALUES or "--debug" in sys.argv[1:]


def configure_library_output() -> None:
    """Silence dependency warnings and progress output outside debug mode."""

    if debug_enabled():
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"^(torch|torchvision|transformers)(?:\.|$)",
    )
    for name in _DEPENDENCY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_transformers_output() -> None:
    """Apply Transformers controls that are only available after import."""

    if debug_enabled():
        return

    from transformers.utils import logging as transformers_logging

    for name in _DEPENDENCY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
