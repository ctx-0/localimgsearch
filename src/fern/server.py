"""FastAPI server for Fern with a ChromaDB backend."""

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

# Disable tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

import fern
from fern.embed import (
    AVAILABLE_MODELS,
    CHROMA_DB_PATH,
    DEFAULT_MODEL,
    LocalImageSearch,
)
from fern.reranker import DEFAULT_MODEL as DEFAULT_RERANKER_MODEL
from fern.reranker import NoOpReranker, Qwen3VLReranker, SearchCandidate

app = FastAPI(title="Fern", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates - check package directory first, then current directory
PACKAGE_DIR = Path(fern.__file__).parent
TEMPLATE_DIRS = [
    PACKAGE_DIR / "templates",
    Path.cwd() / "templates",
]
templates_dir = next((d for d in TEMPLATE_DIRS if d.exists()), TEMPLATE_DIRS[1])
templates = Jinja2Templates(directory=str(templates_dir))

# Global instance
searcher: Optional[LocalImageSearch] = None
model_name: str = DEFAULT_MODEL


def format_search_results(results):
    """Format search results for API response."""
    formatted = []
    for result in results:
        if isinstance(result, SearchCandidate):
            path = str(result.path)
            metadata = dict(result.metadata)
            retrieval_score = result.retrieval_score
            rerank_score = metadata.get("rerank_score")
            score = float(rerank_score) if rerank_score is not None else retrieval_score
            reranked = bool(metadata.get("reranked", False))
            retrieval_rank = int(metadata.get("retrieval_rank", 0))
            final_rank = int(metadata.get("final_rank", 0))
            media_kind = result.media_kind
            width = int(metadata.get("width", 0))
            height = int(metadata.get("height", 0))
            asset_id = str(metadata.get("asset_id", ""))
            thumbnail_url = (
                f"/api/preview/{quote(result.id, safe='')}"
                if media_kind != "image"
                else None
            )
            asset_url = (
                f"/api/asset/{quote(asset_id, safe='')}"
                if media_kind != "image" and asset_id
                else f"/api/image?{urlencode({'path': path})}"
            )
            matched_at_ms = round((result.timestamp_seconds or 0) * 1000)
            unit_id = result.id
        else:
            path, score = result
            retrieval_score = score
            rerank_score = None
            reranked = False
            retrieval_rank = 0
            final_rank = 0
            media_kind = "image"
            width = height = 0
            asset_id = ""
            thumbnail_url = None
            asset_url = f"/api/image?{urlencode({'path': path})}"
            matched_at_ms = 0
            unit_id = ""

        version = searcher.get_image_version(path)
        if width <= 0 or height <= 0:
            width, height = searcher.get_image_dimensions(path, version)
        if thumbnail_url is None:
            thumbnail_url = f"/api/thumbnail?{urlencode({'path': path, 'v': version, 'size': 640, 'quality': 90})}"
        formatted.append(
            {
                "path": path,
                "score": score,
                "retrieval_score": retrieval_score,
                "rerank_score": rerank_score,
                "reranked": reranked,
                "retrieval_rank": retrieval_rank,
                "final_rank": final_rank,
                "rank_delta": retrieval_rank - final_rank,
                "width": width,
                "height": height,
                "thumbnail_url": thumbnail_url,
                "asset_url": asset_url,
                "asset_id": asset_id,
                "unit_id": unit_id,
                "media_kind": media_kind,
                "matched_at_ms": matched_at_ms,
            }
        )
    return formatted


def reranker_status(results):
    """Expose reranker activity and fallback errors without hiding search results."""
    reranker = searcher.reranker
    enabled = not isinstance(reranker, NoOpReranker)
    error = getattr(reranker, "last_error", None)
    return {
        "enabled": enabled,
        "model": getattr(reranker, "model_name", None),
        "loaded": bool(getattr(reranker, "loaded", False)),
        "limit": getattr(reranker, "max_candidates", 0),
        "reranked_count": sum(
            bool(result.metadata.get("reranked", False))
            for result in results
            if isinstance(result, SearchCandidate)
        ),
        "error": str(error) if error else None,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    current_model = searcher.model_name if searcher else model_name
    stats = searcher.get_stats() if searcher else {}

    template = templates.env.get_template("index.html")
    html = template.render(
        request=request,
        model_name=str(current_model),
        model_type="CLIP",
        total_images=stats.get("total_images", 0),
    )
    return HTMLResponse(content=html)


@app.get("/api/info")
async def api_info():
    """Return current model and database info."""
    stats = searcher.get_stats() if searcher else {}
    return {
        "model_name": stats.get("model_name", model_name),
        "collection_name": stats.get("collection_name", ""),
        "model_type": "CLIP",
        "total_images": stats.get("total_images", 0),
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL,
    }


@app.get("/api/stats")
async def api_stats():
    """Return detailed database statistics."""
    if not searcher:
        return {"error": "Search not initialized"}

    stats = searcher.get_stats()
    return {
        "total_images": stats.get("total_images", 0),
        "model_name": stats.get("model_name", ""),
        "collection_name": stats.get("collection_name", ""),
        "last_indexed": stats.get("last_indexed"),
    }


@app.post("/search")
def search(data: dict):
    """Search images by text query."""
    query = data.get("query", "")
    top_k = data.get("top_k", 200)
    top_k = min(int(top_k), 500)

    start = time.time()

    try:
        results = searcher.search_candidates(query, top_k)
    except Exception as e:
        print(f"Search error: {e}")
        return {"results": [], "error": str(e)}

    return {
        "results": format_search_results(results),
        "reranker": reranker_status(results),
        "time_ms": int((time.time() - start) * 1000),
    }


@app.post("/search-by-image")
async def search_by_image(
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    top_k: int = Form(200),
    exclude_self: str = Form("true"),
):
    """Image-to-image search endpoint."""
    start = time.time()
    top_k = min(top_k, 500)
    exclude_self_bool = exclude_self.lower() == "true"

    try:
        if image_path:
            results = searcher.search_by_image(
                image_path=image_path, top_k=top_k, exclude_self=exclude_self_bool
            )
        elif image:
            image_data = await image.read()
            results = searcher.search_by_image(
                image_data=image_data,
                top_k=top_k,
                exclude_self=False,
            )
        else:
            return {"results": [], "error": "No image provided"}

    except Exception as e:
        print(f"Image search error: {e}")
        return {"results": [], "error": str(e)}

    return {
        "results": format_search_results(results),
        "time_ms": int((time.time() - start) * 1000),
    }


@app.post("/duplicates")
async def find_duplicates(data: dict):
    """Find duplicate images in the indexed collection."""
    start = time.time()

    threshold = float(data.get("threshold", 0.97))
    threshold = max(0.5, min(1.0, threshold))

    try:
        duplicate_groups = searcher.find_duplicates(threshold=threshold)

        formatted_groups = []
        for group in duplicate_groups:
            images = []
            best_path = max(
                group["paths"],
                key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0,
            )
            for path in group["paths"]:
                img_b64 = searcher.get_image_base64(path)
                if img_b64:
                    images.append(
                        {
                            "path": path,
                            "name": os.path.basename(path),
                            "image": img_b64,
                            "is_representative": path == best_path,
                        }
                    )

            if len(images) < 2:
                continue

            if not any(img["is_representative"] for img in images):
                images[0]["is_representative"] = True

            formatted_groups.append(
                {
                    "count": len(images),
                    "avg_similarity": group["avg_similarity"],
                    "images": images,
                }
            )

        return {
            "groups": formatted_groups,
            "total_groups": len(formatted_groups),
            "total_duplicate_images": sum(g["count"] for g in formatted_groups),
            "threshold": threshold,
            "time_ms": int((time.time() - start) * 1000),
        }

    except Exception as e:
        print(f"Duplicate detection error: {e}")
        return {"groups": [], "error": str(e)}


@app.post("/index")
async def index_folder(data: dict):
    """Start indexing a folder."""
    path = data.get("path", "")
    resume = data.get("resume", True)

    if searcher.is_indexing:
        return {"success": False, "error": "Already indexing"}

    def run_indexing():
        searcher.index_images(path, resume=resume)

    threading.Thread(target=run_indexing, daemon=True).start()

    return {"success": True, "message": "Indexing started"}


@app.post("/reindex")
async def reindex_folder(data: dict):
    """Clear database and reindex a folder."""
    path = data.get("path", "")

    if searcher.is_indexing:
        return {"success": False, "error": "Already indexing"}

    def run_reindexing():
        searcher.clear_database()
        searcher.index_images(path, resume=False)

    threading.Thread(target=run_reindexing, daemon=True).start()

    return {"success": True, "message": "Reindexing started"}


@app.post("/clear")
async def clear_database():
    """Clear all images from the database."""
    if searcher.is_indexing:
        return {"success": False, "error": "Cannot clear while indexing"}

    success = searcher.clear_database()
    return {"success": success}


@app.post("/delete")
async def delete_images(data: dict):
    """Delete images from disk and database."""
    paths = data.get("paths", [])
    if not paths:
        return {"deleted": 0, "failed": []}

    failed = []
    deleted = []
    for path in paths:
        try:
            indexed = searcher.collection.get(
                where={"path": path}, limit=1, include=[]
            )
            if not indexed or not indexed.get("ids"):
                raise ValueError("Path is not indexed")
            os.remove(path)
            deleted.append(path)
        except Exception as e:
            failed.append({"path": path, "error": str(e)})

    # Remove from database
    if deleted:
        searcher.remove_images(deleted)

    return {"deleted": len(deleted), "failed": failed}


@app.get("/status")
async def status():
    """Return current indexing status."""
    stats = searcher.get_stats() if searcher else {}

    return {
        "indexing": searcher.is_indexing if searcher else False,
        "progress": searcher.index_progress if searcher else {},
        "loaded": stats.get("total_images", 0),
        "model_name": stats.get("model_name", ""),
        "collection_name": stats.get("collection_name", ""),
    }


@app.get("/api/image")
async def get_image(path: str = Query(...)):
    """Serve full resolution image file."""
    if not path:
        raise HTTPException(status_code=400, detail="Path required")

    indexed = searcher.collection.get(
        where={"path": path}, limit=1, include=[]
    )
    if not indexed or not indexed.get("ids"):
        raise HTTPException(status_code=404, detail="Image is not indexed")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path)


@app.get("/api/thumbnail")
def get_thumbnail(
    path: str = Query(...),
    size: int = Query(640, ge=128, le=1024),
    quality: int = Query(90, ge=70, le=95),
):
    """Serve a cached thumbnail for an indexed image."""
    if not searcher:
        raise HTTPException(status_code=503, detail="Search not initialized")

    indexed = searcher.collection.get(ids=[path], include=[])
    if not indexed or path not in indexed.get("ids", []):
        raise HTTPException(status_code=404, detail="Image is not indexed")

    thumbnail_path = searcher.get_thumbnail_path(
        path,
        max_size=size,
        quality=quality,
    )
    if thumbnail_path is None:
        raise HTTPException(status_code=404, detail="Thumbnail unavailable")

    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/preview/{unit_id}")
def get_media_preview(unit_id: str):
    """Serve one generated preview only after resolving it through the index."""
    records = searcher.collection.get(ids=[unit_id], include=["metadatas"])
    metadatas = records.get("metadatas") or []
    if not metadatas or not metadatas[0].get("preview_path"):
        raise HTTPException(status_code=404, detail="Preview not found")
    preview = Path(str(metadatas[0]["preview_path"])).resolve()
    preview_root = (Path(searcher.db_path).resolve() / "media_previews")
    if preview_root not in preview.parents or not preview.is_file():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/asset/{asset_id}")
def get_media_asset(asset_id: str):
    """Serve a source GIF/video after resolving its opaque indexed asset ID."""
    records = searcher.collection.get(
        where={"asset_id": asset_id}, limit=1, include=["metadatas"]
    )
    metadatas = records.get("metadatas") or []
    if not metadatas:
        raise HTTPException(status_code=404, detail="Asset not found")
    source = Path(str(metadatas[0]["path"])).resolve()
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(source)


@app.get("/api/random")
async def get_random_images(count: int = Query(50)):
    """Return random images from the loaded index."""
    stats = searcher.get_stats() if searcher else {}
    total = stats.get("total_images", 0)

    if total == 0:
        return {"images": [], "error": "No images indexed"}

    count = min(count, 200, total)

    try:
        # Get random sample using ChromaDB's get with random IDs
        import random

        result = searcher.collection.get(
            include=["metadatas"],
        )

        if not result or not result.get("ids"):
            return {"images": [], "error": "No images found"}

        stills = [
            meta
            for meta in (result.get("metadatas") or [])
            if meta
            and meta.get("media_kind", "image") == "image"
            and not meta.get("preview_path")
        ]
        selected = random.sample(stills, min(count, len(stills)))

        images = []
        for meta in selected:
            if not meta or "path" not in meta:
                continue
            path = meta["path"]
            version = searcher.get_image_version(path)
            width = int(meta.get("width", 0))
            height = int(meta.get("height", 0))
            if width <= 0 or height <= 0:
                width, height = searcher.get_image_dimensions(path, version)
            images.append(
                {
                    "path": path,
                    "width": width,
                    "height": height,
                    "media_kind": "image",
                    "thumbnail_url": (
                        f"/api/thumbnail?{urlencode({
                            'path': path,
                            'v': version,
                            'size': 640,
                            'quality': 90,
                        })}"
                    ),
                    "asset_url": f"/api/image?{urlencode({'path': path})}",
                }
            )

        return {"images": images, "total": total}

    except Exception as e:
        print(f"Random images error: {e}")
        return {"images": [], "error": str(e)}


def main():
    """Main entry point for the server."""
    global searcher, model_name

    parser = argparse.ArgumentParser(description="CLIP Image Search Web UI")
    parser.add_argument(
        "--port", "-p", type=int, default=5000, help="Port to run on (default: 5000)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--db-path", default=CHROMA_DB_PATH, help="ChromaDB path"
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--reranker",
        nargs="?",
        const=DEFAULT_RERANKER_MODEL,
        help=(
            "Enable multimodal reranking and require it to load before serving "
            f"(default: {DEFAULT_RERANKER_MODEL})"
        ),
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=30,
        help="Maximum candidates reranked per search (default: 30)",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available CLIP models:")
        print()
        for model in AVAILABLE_MODELS:
            model_type = "OpenAI" if model.startswith("openai/") else "LAION"
            marker = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  {model} [{model_type}]{marker}")
        print()
        return

    model_name = args.model

    print("Starting Fern...")
    print(f"   Retrieval model: {args.model}")
    if args.reranker:
        print(f"   Reranker model: {args.reranker}")
    print(f"   DB Path: {args.db_path}")

    try:
        reranker = None
        if args.reranker:
            reranker = Qwen3VLReranker(
                model_name=args.reranker,
                max_candidates=args.rerank_limit,
                failure_policy="passthrough",
            )
        searcher = LocalImageSearch(
            model_name=args.model,
            db_path=args.db_path,
            reranker=reranker,
        )
        print("[OK] Retrieval model loaded")

        if reranker is not None:
            print("   Loading reranker...")
            reranker.preload()
            print("[OK] Reranker model loaded")

        stats = searcher.get_stats()
        if stats["total_images"] > 0:
            print(f"[OK] Database loaded: {stats['total_images']} images")
            print(f"   Collection: {stats['collection_name']}")
        else:
            print("   No images indexed. Use Index button to create an index.")

    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        sys.exit(1)

    print(f"   URL: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
