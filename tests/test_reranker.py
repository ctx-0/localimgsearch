from pathlib import Path

import pytest

from localimgsearch.reranker import (
    NoOpReranker,
    Qwen3VLReranker,
    RerankerUnavailableError,
    SearchCandidate,
)


def candidate(id: str, score: float, *, frames: int = 0) -> SearchCandidate:
    return SearchCandidate(
        id=id,
        path=Path(f"{id}.jpg"),
        retrieval_score=score,
        media_kind="video" if frames else "image",
        frame_paths=tuple(Path(f"{id}-{index}.jpg") for index in range(frames)),
    )


def test_noop_preserves_order_and_scores() -> None:
    candidates = [candidate("a", 0.9), candidate("b", 0.7)]

    results = NoOpReranker().rerank("query", candidates)

    assert [result.candidate.id for result in results] == ["a", "b"]
    assert [result.score for result in results] == [0.9, 0.7]
    assert not any(result.reranked for result in results)


def test_qwen_bounds_candidates_and_frames_without_loading_weights() -> None:
    class FakeModel:
        documents: list[dict[str, object]]

        def predict(self, pairs: object, **kwargs: object) -> list[float]:
            del kwargs
            materialized = list(pairs)  # type: ignore[arg-type]
            self.documents = [pair[1] for pair in materialized]
            return [0.1, 0.9]

    reranker = Qwen3VLReranker(max_candidates=2, max_frames=3)
    fake = FakeModel()
    reranker._model = fake
    candidates = [candidate("a", 0.8, frames=8), candidate("b", 0.7), candidate("c", 0.6)]

    results = reranker.rerank("a dog", candidates)

    assert [result.candidate.id for result in results] == ["b", "a", "c"]
    assert len(fake.documents[0]["video"]) == 3  # type: ignore[arg-type]
    assert results[-1].reranked is False


def test_qwen_preload_initializes_once_and_clears_old_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    model = object()

    def load(self: Qwen3VLReranker) -> object:
        nonlocal calls
        if self._model is None:
            calls += 1
            self._model = model  # type: ignore[assignment]
        return self._model

    monkeypatch.setattr(Qwen3VLReranker, "_load_model", load)
    reranker = Qwen3VLReranker()
    reranker.last_error = RerankerUnavailableError("old failure")

    reranker.preload()
    reranker.preload()

    assert reranker.loaded
    assert reranker.last_error is None
    assert calls == 1


def test_qwen_preload_records_and_raises_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RerankerUnavailableError("not installed")

    def unavailable(self: Qwen3VLReranker) -> object:
        raise error

    monkeypatch.setattr(Qwen3VLReranker, "_load_model", unavailable)
    reranker = Qwen3VLReranker(failure_policy="passthrough")

    with pytest.raises(RerankerUnavailableError, match="not installed"):
        reranker.preload()

    assert reranker.last_error is error


def test_missing_optional_backend_can_raise_or_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(self: Qwen3VLReranker) -> object:
        raise RerankerUnavailableError("not installed")

    monkeypatch.setattr(Qwen3VLReranker, "_load_model", unavailable)
    candidates = [candidate("a", 0.8)]

    safe = Qwen3VLReranker(failure_policy="passthrough")
    with pytest.warns(RuntimeWarning, match="not installed"):
        assert safe.rerank("dog", candidates)[0].reranked is False
    assert isinstance(safe.last_error, RerankerUnavailableError)

    strict = Qwen3VLReranker(failure_policy="raise")
    with pytest.raises(RerankerUnavailableError, match="not installed"):
        strict.rerank("dog", candidates)
