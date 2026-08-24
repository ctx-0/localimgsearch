import asyncio
from pathlib import Path

import pytest
import torch
from fastapi import HTTPException

from localimgsearch.embed import LocalImageSearch
from localimgsearch.reranker import NoOpReranker, SearchCandidate
from localimgsearch import server


class FakeProcessor:
    def __call__(self, **kwargs):
        del kwargs
        return {"input_ids": torch.tensor([[1]])}


class FakeModel:
    def get_text_features(self, **kwargs):
        del kwargs
        return torch.tensor([[1.0, 0.0]])


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def query(self, *, n_results, **kwargs):
        del kwargs
        rows = self.rows[:n_results]
        return {
            "ids": [[row[0] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
            "distances": [[row[2] for row in rows]],
        }


def test_search_widens_and_returns_requested_unique_asset_count() -> None:
    rows = []
    for index in range(250):
        metadata = {"path": f"asset-{index}.jpg"}
        rows.append((f"unit-{index}", metadata, 0.1 + index / 10000))
        if index < 75:
            rows.append((f"duplicate-{index}", metadata, 0.2 + index / 10000))

    searcher = LocalImageSearch.__new__(LocalImageSearch)
    searcher.collection = FakeCollection(rows)
    searcher.processor = FakeProcessor()
    searcher.model = FakeModel()
    searcher.device = "cpu"
    searcher.reranker = NoOpReranker()

    results = searcher.search_candidates("query", top_k=200)

    assert len(results) == 200
    assert len({result.path for result in results}) == 200
    assert [result.metadata["retrieval_rank"] for result in results] == list(
        range(1, 201)
    )
    assert [result.metadata["final_rank"] for result in results] == list(
        range(1, 201)
    )


def test_static_gif_candidate_uses_indexed_preview(monkeypatch) -> None:
    class Searcher:
        def get_image_version(self, path):
            return "v1"

        def get_image_dimensions(self, path, version):
            return (16, 9)

    monkeypatch.setattr(server, "searcher", Searcher())
    candidate = SearchCandidate(
        id="vu:asset:generation:0000",
        path=Path("still.gif"),
        retrieval_score=0.8,
        media_kind="gif",
        metadata={"asset_id": "asset", "preview_path": "preview.jpg"},
    )

    result = server.format_search_results([candidate])[0]

    assert result["thumbnail_url"].startswith("/api/preview/")
    assert result["asset_url"] == "/api/asset/asset"


def test_formatter_exposes_reranker_rank_movement(monkeypatch) -> None:
    class Searcher:
        def get_image_version(self, path):
            return "v1"

        def get_image_dimensions(self, path, version):
            return (4, 3)

    monkeypatch.setattr(server, "searcher", Searcher())
    candidate = SearchCandidate(
        id="image",
        path=Path("image.jpg"),
        retrieval_score=0.72,
        metadata={
            "asset_id": "asset",
            "retrieval_rank": 12,
            "final_rank": 3,
            "reranked": True,
            "rerank_score": 0.88,
        },
    )

    result = server.format_search_results([candidate])[0]

    assert result["retrieval_rank"] == 12
    assert result["final_rank"] == 3
    assert result["rank_delta"] == 9
    assert result["retrieval_score"] == 0.72
    assert result["rerank_score"] == 0.88


def test_full_image_route_rejects_unindexed_files(tmp_path, monkeypatch) -> None:
    image = tmp_path / "private.jpg"
    image.write_bytes(b"not actually an image")

    class Collection:
        def get(self, **kwargs):
            del kwargs
            return {"ids": []}

    class Searcher:
        collection = Collection()

    monkeypatch.setattr(server, "searcher", Searcher())

    with pytest.raises(HTTPException, match="404"):
        asyncio.run(server.get_image(str(image)))
