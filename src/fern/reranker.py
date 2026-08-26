"""Optional second-stage reranking for multimodal search results.

The Qwen backend is intentionally lazy: importing this module does not import
Sentence Transformers, initialize CUDA, or download model weights.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable


DEFAULT_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_INSTRUCTION = "Retrieve images or videos relevant to the user's query."


class RerankerError(RuntimeError):
    """Base error raised by a reranker backend."""


class RerankerUnavailableError(RerankerError):
    """The requested reranker backend cannot be loaded."""


class RerankerConfigurationError(RerankerError):
    """The reranker or a candidate has invalid configuration."""


class RerankerInferenceError(RerankerError):
    """The backend failed while scoring candidates."""


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """A first-stage result prepared for optional multimodal reranking.

    ``path`` is always the source asset. ``frame_paths`` represents an ordered
    video/GIF frame sequence and ``contact_sheet_path`` is its lower-cost image
    alternative. The adapter defensively samples frames to its configured cap.
    """

    id: str
    path: Path
    retrieval_score: float
    media_kind: Literal["image", "gif", "video"] = "image"
    timestamp_seconds: float | None = None
    frame_paths: tuple[Path, ...] = ()
    contact_sheet_path: Path | None = None
    text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("candidate id must not be empty")
        if not math.isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite")
        if self.media_kind not in ("image", "gif", "video"):
            raise ValueError("media_kind must be 'image', 'gif', or 'video'")
        if self.timestamp_seconds is not None and (
            not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be finite and non-negative")

        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if self.contact_sheet_path is not None and not isinstance(
            self.contact_sheet_path, Path
        ):
            object.__setattr__(self, "contact_sheet_path", Path(self.contact_sheet_path))
        object.__setattr__(
            self,
            "frame_paths",
            tuple(path if isinstance(path, Path) else Path(path) for path in self.frame_paths),
        )


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A candidate with its final score and original first-stage rank."""

    candidate: SearchCandidate
    score: float
    original_rank: int
    reranked: bool


@runtime_checkable
class Reranker(Protocol):
    """Interface for a second-stage search reranker."""

    def rerank(
        self, query: str, candidates: Sequence[SearchCandidate]
    ) -> list[RerankResult]: ...


class NoOpReranker:
    """Keep first-stage ordering and scores unchanged."""

    def rerank(
        self, query: str, candidates: Sequence[SearchCandidate]
    ) -> list[RerankResult]:
        del query
        return [
            RerankResult(candidate, candidate.retrieval_score, rank, reranked=False)
            for rank, candidate in enumerate(candidates)
        ]


class _CrossEncoder(Protocol):
    def predict(self, pairs: object, **kwargs: object) -> object: ...


FailurePolicy = Literal["passthrough", "raise"]


class Qwen3VLReranker:
    """Lazy adapter for the official Qwen Sentence Transformers interface.

    The model card documents ``CrossEncoder(model).predict(pairs, prompt=...)``
    for this checkpoint. ``failure_policy='passthrough'`` keeps search usable
    with first-stage order if the optional backend cannot load or infer;
    ``'raise'`` exposes a typed, actionable error instead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_candidates: int = 50,
        max_frames: int = 8,
        batch_size: int = 4,
        device: str | None = None,
        failure_policy: FailurePolicy = "passthrough",
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not instruction:
            raise ValueError("instruction must not be empty")
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if failure_policy not in ("passthrough", "raise"):
            raise ValueError("failure_policy must be 'passthrough' or 'raise'")

        self.model_name = model_name
        self.instruction = instruction
        self.max_candidates = max_candidates
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.device = device
        self.failure_policy = failure_policy
        self.last_error: RerankerError | None = None
        self._model: _CrossEncoder | None = None
        self._lock = threading.Lock()
        self._passthrough = NoOpReranker()
        self._warned = False

    @property
    def loaded(self) -> bool:
        """Whether model construction has completed successfully."""

        return self._model is not None

    def preload(self) -> None:
        """Load model weights now and raise if the backend is unavailable.

        Explicit startup loading is strict even when inference is configured to
        fall back to first-stage results. This keeps server readiness truthful.
        """

        try:
            with self._lock:
                self._load_model()
        except RerankerError as error:
            self.last_error = error
            raise
        self.last_error = None
        self._warned = False

    def rerank(
        self, query: str, candidates: Sequence[SearchCandidate]
    ) -> list[RerankResult]:
        baseline = self._passthrough.rerank(query, candidates)
        if not candidates:
            return baseline

        try:
            if not query.strip():
                raise RerankerConfigurationError("rerank query must not be empty")

            selected = candidates[: self.max_candidates]
            documents = [self._document(candidate) for candidate in selected]
            pairs = [(query, document) for document in documents]

            # Model initialization and inference are serialized. Most local GPU
            # runtimes do not benefit from concurrent calls into one model.
            with self._lock:
                model = self._load_model()
                raw_scores = model.predict(
                    pairs,
                    prompt=self.instruction,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            scores = self._scores(raw_scores, len(selected))
        except RerankerError as error:
            return self._on_error(error, baseline)
        except Exception as error:
            wrapped = RerankerInferenceError(
                f"Qwen reranking failed for {self.model_name!r}: {error}"
            )
            return self._on_error(wrapped, baseline, cause=error)

        self.last_error = None
        reranked = [
            RerankResult(candidate, score, rank, reranked=True)
            for rank, (candidate, score) in enumerate(zip(selected, scores))
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked + baseline[len(selected) :]

    def _load_model(self) -> _CrossEncoder:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerUnavailableError(
                "Qwen reranking requires the optional 'sentence-transformers' "
                "package; install a version supporting Qwen3-VL-Reranker"
            ) from error

        kwargs = {"device": self.device} if self.device else {}
        try:
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as error:
            raise RerankerUnavailableError(
                f"could not load Qwen reranker {self.model_name!r}: {error}"
            ) from error
        return self._model

    def _document(self, candidate: SearchCandidate) -> dict[str, object]:
        document: dict[str, object] = {}
        if candidate.frame_paths:
            document["video"] = [
                str(path.resolve()) for path in _sample_evenly(candidate.frame_paths, self.max_frames)
            ]
        elif candidate.contact_sheet_path is not None:
            document["image"] = str(candidate.contact_sheet_path.resolve())
        elif candidate.media_kind == "image":
            document["image"] = str(candidate.path.resolve())
        else:
            raise RerankerConfigurationError(
                f"{candidate.media_kind} candidate {candidate.id!r} needs bounded "
                "frame_paths or a contact_sheet_path"
            )
        if candidate.text:
            document["text"] = candidate.text
        if not document:
            raise RerankerConfigurationError(
                f"candidate {candidate.id!r} has no rerankable content"
            )
        return document

    @staticmethod
    def _scores(raw_scores: object, expected: int) -> list[float]:
        try:
            scores = [float(score) for score in raw_scores]  # type: ignore[union-attr]
        except (TypeError, ValueError) as error:
            raise RerankerInferenceError("reranker returned invalid scores") from error
        if len(scores) != expected or not all(math.isfinite(score) for score in scores):
            raise RerankerInferenceError(
                f"reranker returned {len(scores)} finite scores; expected {expected}"
            )
        return scores

    def _on_error(
        self,
        error: RerankerError,
        baseline: list[RerankResult],
        *,
        cause: Exception | None = None,
    ) -> list[RerankResult]:
        self.last_error = error
        if self.failure_policy == "raise":
            if cause is not None:
                raise error from cause
            raise error
        if not self._warned:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            self._warned = True
        return baseline


def _sample_evenly(paths: Sequence[Path], limit: int) -> tuple[Path, ...]:
    """Return at most ``limit`` paths spread across the full sequence."""

    if len(paths) <= limit:
        return tuple(paths)
    if limit == 1:
        return (paths[len(paths) // 2],)
    last = len(paths) - 1
    return tuple(paths[round(index * last / (limit - 1))] for index in range(limit))


__all__ = [
    "DEFAULT_MODEL",
    "NoOpReranker",
    "Qwen3VLReranker",
    "RerankResult",
    "Reranker",
    "RerankerConfigurationError",
    "RerankerError",
    "RerankerInferenceError",
    "RerankerUnavailableError",
    "SearchCandidate",
]
