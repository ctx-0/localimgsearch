"""LocalImageSearch - AI-powered local image search with CLIP and ChromaDB."""

import argparse
import base64
import hashlib
import math
import os
import re
import sys
import tempfile
import threading
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from fern.diagnostics import configure_transformers_output, debug_enabled
from fern.media import DEFAULT_POLICY, SamplingPolicy, extract_visual_units
from fern.reranker import NoOpReranker, Reranker, SearchCandidate


configure_transformers_output()

AVAILABLE_MODELS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
]

DEFAULT_MODEL = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

# ChromaDB settings
CHROMA_DB_PATH = "./chromadb"


def _sanitize_collection_name(model_name: str) -> str:
    """
    Convert model name to a valid ChromaDB collection name.

    ChromaDB requires collection names to match: ^[a-zA-Z][a-zA-Z0-9_-]*$
    Example: "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" -> "laion_clip_vit_l_14"
    """
    # Remove the org prefix (e.g., "laion/" or "openai/")
    name = model_name.split("/")[-1] if "/" in model_name else model_name

    # Replace hyphens with underscores, remove any non-alphanumeric chars except underscore
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Remove trailing underscores
    name = name.rstrip("_")

    # Ensure starts with letter (prepend 'img_' if needed)
    if name and not name[0].isalpha():
        name = "img_" + name

    # Limit length (ChromaDB has limits)
    name = name[:60]

    return name.lower()


class LocalImageSearch:
    """Local image search using CLIP embeddings stored in ChromaDB."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        db_path: str = CHROMA_DB_PATH,
        reranker: Optional[Reranker] = None,
        media_policy: SamplingPolicy = DEFAULT_POLICY,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = (
            model_name if model_name in AVAILABLE_MODELS else DEFAULT_MODEL
        )
        self.db_path = db_path
        self.collection_name = _sanitize_collection_name(self.model_name)
        self.reranker = reranker or NoOpReranker()
        self.media_policy = media_policy
        self._asset_count_cache: Optional[int] = None

        # Progress tracking
        self.is_indexing = False
        self.index_progress = {"current": 0, "total": 0, "status": "idle"}

        # Initialize ChromaDB
        self._init_chroma()

        # Load model
        self._load_model()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and collection for this model."""
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)

        # Get or create collection for this specific model
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "model_name": self.model_name,
                "description": f"Image embeddings for {self.model_name}",
            },
        )

    def _load_model(self) -> None:
        """Load CLIP model and processor using Auto classes."""
        if debug_enabled():
            print(f"Loading CLIP model ({self.model_name}) on {self.device}...")

        self.model = AutoModel.from_pretrained(
            self.model_name, low_cpu_mem_usage=True, dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        if debug_enabled():
            print(f"[OK] Model loaded on {self.device}")

    def _extract_tensor(
        self, model_output: Union[torch.Tensor, object]
    ) -> torch.Tensor:
        """Safely extract tensor from model output."""
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, "pooler_output"):
            return model_output.pooler_output
        elif hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state[:, 0, :]
        elif isinstance(model_output, (tuple, list)):
            return model_output[1]
        else:
            raise TypeError(f"Cannot extract features from {type(model_output)}")

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize embeddings."""
        return features / (features.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convert default squared-L2 distance for unit vectors to cosine."""
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    def _get_image_hash(self, image_path: str) -> str:
        """Generate a hash for an image file to detect changes."""
        try:
            stat = os.stat(image_path)
            content = f"{image_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except OSError:
            return hashlib.md5(image_path.encode()).hexdigest()

    def get_image_version(self, image_path: str) -> str:
        """Return the lightweight version token used by caches and indexing."""
        return self._get_image_hash(image_path)

    @lru_cache(maxsize=32768)
    def _get_image_dimensions_cached(
        self, image_path: str, version: str
    ) -> Tuple[int, int]:
        """Read display dimensions once per file version without decoding pixels."""
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                orientation = image.getexif().get(274, 1)
                if orientation in {5, 6, 7, 8}:
                    width, height = height, width
                return max(width, 1), max(height, 1)
        except Exception:
            return 4, 3

    def get_image_dimensions(
        self, image_path: str, version: Optional[str] = None
    ) -> Tuple[int, int]:
        """Return cached display dimensions for stable result layout."""
        return self._get_image_dimensions_cached(
            image_path, version or self.get_image_version(image_path)
        )

    def _get_indexed_images(self) -> Dict[str, str]:
        """Get map of already indexed image paths to their hashes."""
        try:
            result = self.collection.get(include=["metadatas"])
            if result and result["metadatas"]:
                return {
                    meta["path"]: meta.get("file_hash", "")
                    for meta in result["metadatas"]
                    if meta and "path" in meta
                }
        except Exception as e:
            print(f"Warning: Could not get indexed images: {e}")
        return {}

    def index_images(
        self,
        image_dir: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        batch_size: int = 16,
        resume: bool = True,
        include_media: bool = False,
    ) -> bool:
        """
        Index images and store in ChromaDB.

        Args:
            image_dir: Directory containing images
            extensions: Tuple of image extensions to index
            batch_size: Number of images to process at once
            resume: If True, skip already indexed images

        Returns:
            True if successful, False otherwise
        """
        self.is_indexing = True
        self.index_progress = {"current": 0, "total": 0, "status": "scanning"}

        try:
            image_dir = Path(image_dir).expanduser().resolve()
            if not image_dir.exists():
                print(f"Error: Directory {image_dir} does not exist")
                self.is_indexing = False
                return False

            # Collect images
            print(f"Scanning {image_dir}...")
            all_image_paths = []
            for ext in extensions:
                all_image_paths.extend(image_dir.rglob(f"*{ext}"))
                all_image_paths.extend(image_dir.rglob(f"*{ext.upper()}"))

            all_image_paths = sorted(list(set([str(p) for p in all_image_paths])))

            if not all_image_paths:
                print("No still images found")
                self.is_indexing = False
                if include_media:
                    return self.index_media(
                        image_dir, batch_size=batch_size, resume=resume
                    )
                return False

            # Check for already indexed images
            indexed_images = self._get_indexed_images() if resume else {}
            images_to_process = []

            for path in all_image_paths:
                current_hash = self._get_image_hash(path)
                if path in indexed_images and indexed_images[path] == current_hash:
                    continue  # Skip unchanged, already indexed images
                images_to_process.append(path)

            if resume and indexed_images:
                skipped = len(all_image_paths) - len(images_to_process)
                print(
                    f"Found {len(all_image_paths)} images ({skipped} already indexed)"
                )
            else:
                print(f"Found {len(images_to_process)} images")

            if not images_to_process:
                print("All images already indexed")
                self.index_progress = {
                    "current": len(all_image_paths),
                    "total": len(all_image_paths),
                    "status": "completed",
                }
                self.is_indexing = False
                if include_media:
                    return self.index_media(
                        image_dir, batch_size=batch_size, resume=resume
                    )
                return True

            self.index_progress["total"] = len(images_to_process)
            self.index_progress["status"] = "indexing"

            # Process images in batches
            pbar = tqdm(total=len(images_to_process), desc="Indexing")
            processed_count = 0
            failed_count = 0

            for i in range(0, len(images_to_process), batch_size):
                batch_paths = images_to_process[i : i + batch_size]
                images = []
                valid_paths = []
                file_hashes = []

                # Load images
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        valid_paths.append(path)
                        file_hashes.append(self._get_image_hash(path))
                    except Exception as e:
                        print(f"\n⚠ Cannot load {path}: {e}")
                        failed_count += 1

                if not images:
                    pbar.update(len(batch_paths))
                    continue

                # Generate embeddings
                try:
                    inputs = self.processor(
                        images=images, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                        features = self._extract_tensor(outputs)
                        features = self._normalize(features)
                        embeddings = features.cpu().numpy().astype("float32")

                    # Store in ChromaDB
                    ids = valid_paths
                    metadatas = [
                        {
                            "path": path,
                            "file_hash": file_hash,
                            "indexed_at": datetime.now().isoformat(),
                        }
                        for path, file_hash in zip(valid_paths, file_hashes)
                    ]

                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings.tolist(),
                        metadatas=metadatas,
                    )

                    processed_count += len(valid_paths)
                    self._asset_count_cache = None

                except Exception as e:
                    print(f"\n[ERROR] Error processing batch: {e}")
                    failed_count += len(valid_paths)

                self.index_progress["current"] = i + len(batch_paths)
                pbar.update(len(batch_paths))

            pbar.close()

            # Update collection metadata
            current_count = self.collection.count()
            self.collection.modify(
                metadata={
                    "model_name": self.model_name,
                    "description": f"Image embeddings for {self.model_name}",
                    "last_indexed": datetime.now().isoformat(),
                    "total_images": current_count,
                }
            )

            self.index_progress["status"] = "completed"
            self.is_indexing = False

            print(f"✓ Indexed {processed_count} images ({failed_count} failed)")
            print(f"  Total in database: {current_count} images")
            media_ok = True
            if include_media:
                media_ok = self.index_media(
                    image_dir, batch_size=batch_size, resume=resume
                )
            return failed_count == 0 and media_ok

        except Exception as e:
            print(f"Indexing error: {e}")
            self.index_progress["status"] = f"error: {str(e)}"
            self.is_indexing = False
            return False

    @staticmethod
    def _asset_id(path: str) -> str:
        normalized = os.path.normcase(str(Path(path).expanduser().resolve()))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def _media_generation(self, file_hash: str) -> str:
        return hashlib.sha256(
            f"{file_hash}:{self.media_policy.fingerprint}".encode("ascii")
        ).hexdigest()[:16]

    def _encode_images(
        self, images: List[Image.Image], batch_size: int
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                features = self._normalize(self._extract_tensor(outputs))
            embeddings.extend(features.cpu().numpy().astype("float32").tolist())
        return embeddings

    @staticmethod
    def _save_preview(image: Image.Image, destination: Path) -> None:
        """Atomically publish a generated JPEG preview."""
        handle = tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".jpg.tmp", delete=False
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            image.save(temporary, format="JPEG", quality=82, optimize=True)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def index_media(
        self,
        media_dir: Union[str, Path],
        *,
        batch_size: int = 16,
        resume: bool = True,
    ) -> bool:
        """Index sampled GIF/video frames while preserving the old asset on failure."""
        media_dir = Path(media_dir).expanduser().resolve()
        if not media_dir.is_dir():
            print(f"Error: Directory {media_dir} does not exist")
            return False
        if batch_size < 1:
            print("Error: Batch size must be at least 1")
            return False
        extensions = {
            ".gif",
            ".mp4",
            ".m4v",
            ".mov",
            ".mpeg",
            ".mpg",
            ".webm",
            ".mkv",
            ".avi",
            ".wmv",
        }
        paths = sorted(
            str(path.resolve())
            for path in media_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not paths:
            return True

        indexed: Dict[str, Tuple[str, str]] = {}
        if resume:
            records = self.collection.get(include=["metadatas"])
            by_path = {}
            for unit_id, meta in zip(
                records.get("ids") or [], records.get("metadatas") or []
            ):
                if meta and meta.get("path"):
                    by_path.setdefault(str(meta["path"]), []).append((unit_id, meta))
            for path in paths:
                file_hash = self._get_image_hash(path)
                generation = self._media_generation(file_hash)
                entries = by_path.get(path, [])
                current = [
                    entry
                    for entry in entries
                    if entry[1].get("generation") == generation
                ]
                expected_count = int(current[0][1].get("unit_count", 0)) if current else 0
                previews_ready = all(
                    entry[1].get("preview_path")
                    and Path(str(entry[1]["preview_path"])).is_file()
                    for entry in current
                )
                if not current or len(current) != expected_count or not previews_ready:
                    continue
                stale = [entry for entry in entries if entry not in current]
                try:
                    if stale:
                        self.collection.delete(ids=[unit_id for unit_id, _ in stale])
                    for _, meta in stale:
                        if meta.get("preview_path"):
                            Path(str(meta["preview_path"])).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    print(f"[WARN] Media cleanup retry failed for {path}: {cleanup_error}")
                    continue
                indexed[path] = (file_hash, self.media_policy.fingerprint)

        pending = [
            path
            for path in paths
            if indexed.get(path)
            != (self._get_image_hash(path), self.media_policy.fingerprint)
        ]
        if not pending:
            print("All GIF/video assets already indexed")
            return True

        self.is_indexing = True
        self.index_progress = {"current": 0, "total": len(pending), "status": "indexing media"}
        preview_dir = Path(self.db_path).expanduser().resolve() / "media_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        failed = 0

        for index, path in enumerate(tqdm(pending, desc="Indexing media"), start=1):
            unit_batch = []
            new_previews: List[Path] = []
            old_previews = set()
            committed = False
            try:
                old = self.collection.get(where={"path": path}, include=["metadatas"])
                old_ids = set(old.get("ids") or [])
                old_previews = {
                    str(meta["preview_path"])
                    for meta in (old.get("metadatas") or [])
                    if meta and meta.get("preview_path")
                }

                file_hash = self._get_image_hash(path)
                asset_id = self._asset_id(path)
                generation = self._media_generation(file_hash)
                ids: List[str] = []
                embeddings: List[List[float]] = []
                metadatas = []

                def flush_units() -> None:
                    if not unit_batch:
                        return
                    try:
                        embeddings.extend(
                            self._encode_images(
                                [unit.image for unit in unit_batch],
                                batch_size=batch_size,
                            )
                        )
                        for unit in unit_batch:
                            unit_id = (
                                f"vu:{asset_id}:{generation}:{unit.ordinal:04d}"
                            )
                            preview = preview_dir / (
                                f"{hashlib.sha256(unit_id.encode()).hexdigest()}.jpg"
                            )
                            preview_image = unit.image.copy()
                            try:
                                preview_image.thumbnail(
                                    (640, 640), Image.Resampling.LANCZOS
                                )
                                self._save_preview(preview_image, preview)
                            finally:
                                preview_image.close()
                            new_previews.append(preview)
                            ids.append(unit_id)
                            metadata = {
                                "schema_version": 2,
                                "asset_id": asset_id,
                                "path": path,
                                "media_kind": (
                                    "gif"
                                    if Path(path).suffix.lower() == ".gif"
                                    else unit.media_kind.value
                                ),
                                "unit_index": unit.ordinal,
                                "timestamp_ms": round(
                                    unit.timestamp_seconds * 1000
                                ),
                                "width": unit.image.width,
                                "height": unit.image.height,
                                "file_hash": file_hash,
                                "generation": generation,
                                "policy_fingerprint": unit.policy_fingerprint,
                                "preview_path": str(preview),
                                "indexed_at": datetime.now().isoformat(),
                            }
                            if unit.duration_seconds is not None:
                                metadata["duration_ms"] = round(
                                    unit.duration_seconds * 1000
                                )
                            metadatas.append(metadata)
                    finally:
                        for unit in unit_batch:
                            unit.image.close()
                        unit_batch.clear()

                for unit in extract_visual_units(path, policy=self.media_policy):
                    unit_batch.append(unit)
                    if len(unit_batch) >= batch_size:
                        flush_units()
                flush_units()
                if not ids:
                    raise RuntimeError("media produced no visual units")
                for metadata in metadatas:
                    metadata["unit_count"] = len(ids)

                self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
                committed = True
                self._asset_count_cache = None
                try:
                    stale_ids = list(old_ids.difference(ids))
                    if stale_ids:
                        self.collection.delete(ids=stale_ids)
                    for stale_preview in old_previews.difference(map(str, new_previews)):
                        Path(stale_preview).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    print(f"\n[WARN] Media cleanup deferred for {path}: {cleanup_error}")
                processed += 1
            except Exception as error:
                failed += 1
                if not committed:
                    for preview in new_previews:
                        if str(preview) not in old_previews:
                            preview.unlink(missing_ok=True)
                print(f"\n[WARN] Cannot index media {path}: {error}")
            finally:
                for unit in unit_batch:
                    unit.image.close()
                self.index_progress["current"] = index

        self.is_indexing = False
        self.index_progress["status"] = "completed"
        print(f"Indexed {processed} GIF/video assets ({failed} failed)")
        return failed == 0

    def search_candidates(self, query: str, top_k: int = 5) -> List[SearchCandidate]:
        """Search visual units, collapse them to assets, then optionally rerank."""
        total_units = self.collection.count()
        if total_units == 0:
            raise ValueError("No images indexed")
        if top_k < 1:
            return []

        try:
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                features = self._normalize(self._extract_tensor(outputs))
                query_embedding = features.cpu().numpy().astype("float32")
        except Exception as error:
            raise RuntimeError(f"Failed to encode query: {error}") from error

        query_limit = min(total_units, max(top_k, 64))
        candidates: List[SearchCandidate] = []
        while True:
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=query_limit,
                    include=["metadatas", "distances"],
                )
            except Exception as error:
                raise RuntimeError(f"Search failed: {error}") from error

            candidates = []
            seen_assets = set()
            ids = (results.get("ids") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            for unit_id, meta, distance in zip(ids, metadatas, distances):
                if not meta or "path" not in meta:
                    continue
                path = str(meta["path"])
                asset_id = str(meta.get("asset_id") or self._asset_id(path))
                if asset_id in seen_assets:
                    continue
                seen_assets.add(asset_id)
                media_kind = str(meta.get("media_kind", "image"))
                preview_value = meta.get("preview_path")
                preview = Path(str(preview_value)) if preview_value else None
                candidates.append(
                    SearchCandidate(
                        id=str(unit_id),
                        path=Path(path),
                        retrieval_score=self._distance_to_score(distance),
                        media_kind=media_kind,
                        timestamp_seconds=float(meta.get("timestamp_ms", 0)) / 1000,
                        contact_sheet_path=preview,
                        metadata={**meta, "asset_id": asset_id},
                    )
                )
                if len(candidates) >= top_k:
                    break
            if len(candidates) >= top_k or query_limit >= total_units:
                break
            query_limit = min(total_units, query_limit * 2)

        frames_by_asset = {}
        for meta in metadatas:
            if not meta or not meta.get("preview_path") or not meta.get("path"):
                continue
            asset_id = str(
                meta.get("asset_id") or self._asset_id(str(meta["path"]))
            )
            frames_by_asset.setdefault(asset_id, []).append(
                (int(meta.get("unit_index", 0)), Path(str(meta["preview_path"])))
            )
        candidates = [
            replace(
                candidate,
                frame_paths=tuple(
                    path
                    for _, path in sorted(
                        frames_by_asset.get(
                            str(candidate.metadata["asset_id"]), []
                        ),
                        key=lambda item: item[0],
                    )
                ),
            )
            if candidate.media_kind in {"gif", "video"}
            else candidate
            for candidate in candidates
        ]

        reranked = self.reranker.rerank(query, candidates[:top_k])
        ranked_candidates = []
        for final_rank, result in enumerate(reranked, start=1):
            metadata = {
                **result.candidate.metadata,
                "retrieval_rank": result.original_rank + 1,
                "final_rank": final_rank,
                "reranked": result.reranked,
            }
            if result.reranked:
                metadata["rerank_score"] = self._sigmoid(result.score)
            ranked_candidates.append(replace(result.candidate, metadata=metadata))
        return ranked_candidates

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search images by text query.

        Args:
            query: Text query string
            top_k: Number of results to return

        Returns:
            List of (path, score) tuples
        """
        return [
            (
                str(candidate.path),
                float(
                    candidate.metadata.get(
                        "rerank_score", candidate.retrieval_score
                    )
                ),
            )
            for candidate in self.search_candidates(query, top_k)
        ]

    def search_by_image(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Search images by image query (image-to-image search).

        Args:
            image_path: Path to query image (optional if image_data provided)
            image_data: Raw image bytes (optional if image_path provided)
            top_k: Number of results to return
            exclude_self: If True and image_path is in index, exclude it from results

        Returns:
            List of (path, score) tuples
        """
        if self.collection.count() == 0:
            raise ValueError("No images indexed")

        if image_path is None and image_data is None:
            raise ValueError("Either image_path or image_data must be provided")

        # Load and encode image
        try:
            if image_path:
                img = Image.open(image_path).convert("RGB")
            else:
                img = Image.open(BytesIO(image_data)).convert("RGB")

            inputs = self.processor(images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                img_features = self._extract_tensor(outputs)
                img_features = self._normalize(img_features)
                query_embedding = img_features.cpu().numpy().astype("float32")
        except Exception as e:
            raise RuntimeError(f"Failed to encode image: {e}")

        try:
            total_units = self.collection.count()
            query_k = min(total_units, max(top_k, 64))
            paths_and_scores = []
            while True:
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=query_k,
                    include=["metadatas", "distances"],
                )
                paths_and_scores = []
                seen_paths = set()
                if results["metadatas"] and results["distances"]:
                    for meta, distance in zip(
                        results["metadatas"][0], results["distances"][0]
                    ):
                        if not meta or "path" not in meta:
                            continue
                        path = str(meta["path"])
                        if path in seen_paths:
                            continue
                        seen_paths.add(path)
                        if exclude_self and image_path and path == image_path:
                            continue
                        paths_and_scores.append(
                            (path, self._distance_to_score(distance))
                        )
                        if len(paths_and_scores) >= top_k:
                            break
                if len(paths_and_scores) >= top_k or query_k >= total_units:
                    return paths_and_scores
                query_k = min(total_units, query_k * 2)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def find_duplicates(self, threshold: float = 0.97) -> List[Dict]:
        """
        Find duplicate or near-duplicate images.

        Args:
            threshold: Similarity threshold (0.0-1.0). Higher = stricter matching.

        Returns:
            List of duplicate groups
        """
        count = self.collection.count()
        if count < 2:
            return []

        print(f"Finding duplicates (threshold={threshold})...")

        # Get all embeddings
        result = self.collection.get(include=["embeddings", "metadatas"])

        if not result or not result["embeddings"]:
            return []

        stills = [
            (embedding, meta)
            for embedding, meta in zip(result["embeddings"], result["metadatas"])
            if meta
            and meta.get("media_kind", "image") == "image"
            and not meta.get("preview_path")
        ]
        if len(stills) < 2:
            return []
        embeddings = np.array([embedding for embedding, _ in stills], dtype="float32")
        paths = [str(meta["path"]) for _, meta in stills]

        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Find pairs above threshold (upper triangle only)
        rows, cols = np.where(np.triu(similarities >= threshold, k=1))

        if len(rows) == 0:
            print("No duplicates found")
            return []

        # Group into connected components using union-find
        parent = list(range(len(paths)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in zip(rows, cols):
            union(int(i), int(j))

        # Group by root
        groups: Dict[int, List[int]] = {}
        for i in range(len(paths)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Build result
        duplicate_groups = []
        for indices in groups.values():
            if len(indices) < 2:
                continue

            indices = sorted(indices)
            group_paths = [paths[i] for i in indices]
            group_scores = [
                float(similarities[i, j])
                for ii, i in enumerate(indices)
                for j in indices[ii + 1 :]
            ]

            duplicate_groups.append(
                {
                    "indices": indices,
                    "paths": group_paths,
                    "scores": group_scores,
                    "count": len(indices),
                    "representative": indices[0],
                    "avg_similarity": sum(group_scores) / len(group_scores)
                    if group_scores
                    else 1.0,
                }
            )

        duplicate_groups.sort(key=lambda x: (-x["count"], -x["avg_similarity"]))

        print(
            f"Found {len(duplicate_groups)} duplicate groups ({sum(g['count'] for g in duplicate_groups)} images)"
        )
        return duplicate_groups

    def remove_image(self, path: str) -> bool:
        """Remove every visual unit belonging to one source asset."""
        try:
            records = self.collection.get(where={"path": path}, include=["metadatas"])
            ids = records.get("ids") or []
            if not ids:
                return False
            self.collection.delete(ids=ids)
            self._asset_count_cache = None
            for meta in records.get("metadatas") or []:
                if meta and meta.get("preview_path"):
                    Path(str(meta["preview_path"])).unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"Warning: Could not remove {path}: {e}")
            return False

    def remove_images(self, paths: List[str]) -> Tuple[int, List[str]]:
        """Remove multiple images from the database."""
        if not paths:
            return 0, []

        failed = [path for path in paths if not self.remove_image(path)]

        return len(paths) - len(failed), failed

    def get_stats(self) -> Dict:
        """Get database statistics."""
        unit_count = self.collection.count()
        metadata = self.collection.metadata or {}
        if self._asset_count_cache is None:
            records = self.collection.get(include=["metadatas"])
            self._asset_count_cache = len(
                {
                    str(meta["path"])
                    for meta in (records.get("metadatas") or [])
                    if meta and meta.get("path")
                }
            )

        return {
            "total_images": self._asset_count_cache,
            "total_visual_units": unit_count,
            "model_name": self.model_name,
            "collection_name": self.collection_name,
            "last_indexed": metadata.get("last_indexed"),
        }

    def clear_database(self) -> bool:
        """Clear all images from this model's collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "model_name": self.model_name,
                    "description": f"Image embeddings for {self.model_name}",
                },
            )
            self._asset_count_cache = 0
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

    def get_image_base64(
        self, image_path: str, max_size: int = 320, quality: int = 85
    ) -> Optional[str]:
        """Convert image to base64 for web display."""
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((max_size, max_size))
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"⚠ Failed to generate thumbnail for {image_path}: {e}")
            return None

    def get_thumbnail_path(
        self, image_path: str, max_size: int = 640, quality: int = 90
    ) -> Optional[Path]:
        """Return a cached JPEG thumbnail path, generating it when needed."""
        version = self._get_image_hash(image_path)
        cache_key = hashlib.sha256(
            f"{version}:{max_size}:{quality}:jpeg".encode()
        ).hexdigest()
        cache_dir = Path(self.db_path) / "thumbnails"
        cache_path = cache_dir / f"{cache_key}.jpg"

        if cache_path.is_file():
            return cache_path

        temp_path = cache_path.with_name(
            f"{cache_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.thumbnail((max_size, max_size))
                image.save(temp_path, format="JPEG", quality=quality)
            os.replace(temp_path, cache_path)
            return cache_path
        except Exception as e:
            print(f"⚠ Failed to generate cached thumbnail for {image_path}: {e}")
            return None
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def display_results(self, results: List[Tuple[str, float]]) -> None:
        """Display search results in terminal."""
        if not results:
            print("No results found")
            return

        print(f"\n{'=' * 70}")
        for i, (path, score) in enumerate(results, 1):
            print(f"{i}. {score:.4f} | {os.path.basename(path)}")

        # Try to display images
        try:
            import matplotlib.pyplot as plt

            n = len(results)
            fig, axes = plt.subplots(1, min(n, 5), figsize=(15, 3))
            if n == 1:
                axes = [axes]

            for idx, (path, score) in enumerate(results[:5]):
                try:
                    img = Image.open(path)
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"{score:.3f}")
                    axes[idx].axis("off")
                except Exception:
                    pass
            plt.tight_layout()
            plt.show()
        except Exception:
            pass


def list_available_models():
    """Print available models and exit."""
    print("Available CLIP models:")
    print()
    print(f"  {'Model':<60} {'Type':<10}")
    print("  " + "-" * 70)
    for model in AVAILABLE_MODELS:
        model_type = "OpenAI" if model.startswith("openai/") else "LAION"
        print(f"  {model:<60} {model_type:<10}")
    print()
    print("Examples:")
    print("  fern /path/to/images                    # Index images")
    print("  fern /path/to/images 'red car'          # Search after indexing")
    print("  fern /path/to/images sunset --top-k 20  # Search with more results")


def main():
    parser = argparse.ArgumentParser(description="Local Image Search with CLIP")
    parser.add_argument(
        "embed", nargs="?", help="Directory containing images to embed/index"
    )
    parser.add_argument("search", nargs="?", help="Text query to search for")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of search results (default: 5)",
    )
    parser.add_argument(
        "--reindex",
        "-r",
        action="store_true",
        help="Clear existing database before indexing",
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--db-path", default=CHROMA_DB_PATH, help="ChromaDB path (default: ./chroma_db)"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show database stats and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    print(f"Loading model: {args.model}")

    try:
        searcher = LocalImageSearch(model_name=args.model, db_path=args.db_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Show stats
    if args.stats:
        stats = searcher.get_stats()
        print("\nDatabase Stats:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Model: {stats['model_name']}")
        print(f"  Total images: {stats['total_images']}")
        if stats["last_indexed"]:
            print(f"  Last indexed: {stats['last_indexed']}")
        return

    # Clear and reindex if requested
    if args.reindex and args.embed:
        print("Clearing existing database...")
        searcher.clear_database()

    # Index images
    if args.embed:
        success = searcher.index_images(args.embed, resume=not args.reindex)
        if not success:
            sys.exit(1)

    # Check if we have any images
    if searcher.get_stats()["total_images"] == 0:
        print("No images indexed. Specify a folder to embed: fern <folder>")
        sys.exit(1)

    # Search
    if args.search:
        results = searcher.search(args.search, args.top_k)
        searcher.display_results(results)
    else:
        print("\nInteractive mode. Type queries (or 'quit' to exit):")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if query:
                    results = searcher.search(query, args.top_k)
                    searcher.display_results(results)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
