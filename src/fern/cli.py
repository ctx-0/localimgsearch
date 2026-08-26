"""CLI entry point for Fern."""

import argparse
import sys

from fern.embed import (
    CHROMA_DB_PATH,
    DEFAULT_MODEL,
    LocalImageSearch,
    list_available_models,
)
from fern.reranker import DEFAULT_MODEL as DEFAULT_RERANKER_MODEL
from fern.reranker import Qwen3VLReranker


def _reranker(args):
    model = getattr(args, "reranker", None)
    if not model:
        return None
    return Qwen3VLReranker(
        model_name=model,
        max_candidates=args.rerank_limit,
        failure_policy="passthrough",
    )


def cmd_embed(args):
    """Embed/index images from a directory."""
    import torch
    
    if not args.folder:
        print("Error: Folder path required")
        print("Usage: fern embed <folder> [options]")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cpu":
        print("[WARNING] Running on CPU - this will be slow. Consider using a CUDA-enabled PyTorch for GPU acceleration.")
    print(f"Loading model: {args.model}")
    
    try:
        searcher = LocalImageSearch(
            model_name=args.model, db_path=args.db_path
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Clear and reindex if requested
    if args.reindex:
        print("Clearing existing collection...")
        searcher.clear_database()
    
    # Index images with specified batch size
    success = searcher.index_images(
        args.folder, 
        resume=not args.reindex,
        batch_size=args.batch,
        include_media=args.media,
    )
    if not success:
        sys.exit(1)
    
    print(f"\n✓ Done! Collection: {searcher.collection_name}")


def cmd_search(args):
    """Search for images using text query."""
    import torch
    
    if not args.query:
        print("Error: Search query required")
        print("Usage: fern search <query> [options]")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cpu":
        print("[WARNING] Running on CPU - this will be slow. Consider using a CUDA-enabled PyTorch for GPU acceleration.")
    print(f"Loading model: {args.model}")
    
    try:
        searcher = LocalImageSearch(
            model_name=args.model,
            db_path=args.db_path,
            reranker=_reranker(args),
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Check if we have any images
    stats = searcher.get_stats()
    if stats["total_images"] == 0:
        print(f"No images indexed for model: {args.model}")
        print(f"Run: fern embed <folder> --model {args.model}")
        sys.exit(1)
    
    # Search
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    print(f"\nSearching for: {query}")
    print(f"Collection: {stats['collection_name']} ({stats['total_images']} images)\n")
    
    try:
        results = searcher.search(query, args.top_k)
        searcher.display_results(results)
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        sys.exit(1)


def cmd_stats(args):
    """Show database statistics."""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cpu":
        print("[WARNING] Running on CPU - this will be slow. Consider using a CUDA-enabled PyTorch for GPU acceleration.")
    print(f"Loading model: {args.model}")
    
    try:
        searcher = LocalImageSearch(
            model_name=args.model, db_path=args.db_path
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    stats = searcher.get_stats()
    print("\nDatabase Stats:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Model: {stats['model_name']}")
    print(f"  Total images: {stats['total_images']}")
    if stats['last_indexed']:
        print(f"  Last indexed: {stats['last_indexed']}")


def cmd_serve(args):
    """Launch the web UI."""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cpu":
        print("[WARNING] Running on CPU - this will be slow. Consider using a CUDA-enabled PyTorch for GPU acceleration.")
    
    from fern.server import main as web_main
    
    # Pass args to server
    import sys
    sys.argv = [sys.argv[0]]  # Reset argv
    if args.model != DEFAULT_MODEL:
        sys.argv.extend(["--model", args.model])
    if args.db_path != CHROMA_DB_PATH:
        sys.argv.extend(["--db-path", args.db_path])
    if args.port != 5000:
        sys.argv.extend(["--port", str(args.port)])
    if args.host != "127.0.0.1":
        sys.argv.extend(["--host", args.host])
    if args.reranker:
        sys.argv.extend(["--reranker", args.reranker])
        sys.argv.extend(["--rerank-limit", str(args.rerank_limit)])
    
    web_main()


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Fern - AI-powered local image search with CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fern embed /path/to/images              # Index images
  fern embed /path/to/images --reindex    # Force reindex
  fern embed /path/to/images --batch 32   # Larger batch size
  fern search "red car"                   # Search images
  fern search sunset --top-k 20           # Search with more results
  fern stats                              # Show database stats
  fern serve                              # Open web UI
  fern serve --port 8080                  # UI on custom port
        """
    )
    
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"CLIP model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--db-path",
        default=CHROMA_DB_PATH,
        help=f"ChromaDB path (default: {CHROMA_DB_PATH})",
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available models and exit",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # embed subcommand
    embed_parser = subparsers.add_parser(
        "embed",
        help="Index/embed images from a folder",
        description="Index images from a directory into the database."
    )
    embed_parser.add_argument("folder", help="Directory containing images to index")
    embed_parser.add_argument(
        "--reindex",
        "-r",
        action="store_true",
        help="Clear existing collection before indexing",
    )
    embed_parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=16,
        help="Batch size for processing images (default: 16)",
    )
    embed_parser.add_argument(
        "--media",
        action="store_true",
        help="Also preprocess and index GIF/video assets",
    )
    embed_parser.set_defaults(func=cmd_embed)
    
    # search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search images by text query",
        description="Search for images using a text description."
    )
    search_parser.add_argument("query", nargs="+", help="Search query text")
    search_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    search_parser.add_argument(
        "--reranker",
        nargs="?",
        const=DEFAULT_RERANKER_MODEL,
        help=f"Enable multimodal reranking (default model: {DEFAULT_RERANKER_MODEL})",
    )
    search_parser.add_argument(
        "--rerank-limit",
        type=int,
        default=30,
        help="Maximum candidates reranked (default: 30)",
    )
    search_parser.set_defaults(func=cmd_search)
    
    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show database statistics",
        description="Display statistics about the indexed collection."
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # serve subcommand
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch web UI",
        description="Start the web interface for visual search."
    )
    serve_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run on (default: 5000)",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--reranker",
        nargs="?",
        const=DEFAULT_RERANKER_MODEL,
        help=f"Enable multimodal reranking (default model: {DEFAULT_RERANKER_MODEL})",
    )
    serve_parser.add_argument(
        "--rerank-limit",
        type=int,
        default=30,
        help="Maximum candidates reranked (default: 30)",
    )
    serve_parser.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
