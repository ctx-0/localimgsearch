# LocalImg Search

AI-powered local image search with CLIP. Search your photos using text or images - all processing happens locally on your machine.

## Features

- **Text Search**: Type what you're looking for (e.g., "sunset beach", "red car")
- **Image Search**: Drop an image to find similar ones
- **Drag & Drop**: Simply drag any image onto the app to search
- **Duplicate Detection**: Find and remove duplicate/near-duplicate images
- **Privacy First**: All processing is local - your photos never leave your computer

## Installation

### Prerequisites

- Python 3.10 - 3.13 (3.14 not supported yet due to PyTorch)
- CUDA-capable NVIDIA GPU (optional, but strongly recommended for performance)
- CUDA Toolkit 12.6 (if using GPU)

### Using uv (Recommended)

The project uses `uv` for fast package management. The `pyproject.toml` is configured to automatically install CUDA-enabled PyTorch from the PyTorch index.

```bash
# Clone the repository
git clone https://github.com/localimg/localimgsearch
cd localimgsearch

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install with CUDA support (recommended)
# This will automatically use the pytorch-cu126 index for torch/torchvision
uv pip install -e .
```

### Verifying CUDA Installation

After installation, verify that PyTorch can see your GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

Expected output with GPU:
```
CUDA available: True
CUDA version: 12.6
```

If you see `CUDA available: False`, the application will still work but will be significantly slower (10-50x slower on CPU).

### Switching from CPU to CUDA

If you previously installed the CPU version and want to switch to CUDA:

```bash
# Uninstall CPU version
uv pip uninstall torch torchvision

# Reinstall with CUDA support
uv pip install -e .
```

### Using pip

If you prefer using pip instead of uv:

```bash
# With CUDA 12.6 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .

# With CPU only (slower)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Usage

### CLI Commands

```bash
# Index images from a folder
localimg embed /path/to/images

# Search for images
localimg search "red car"

# Show database statistics
localimg stats

# Launch the web UI
localimg ui

# Use a different CLIP model
localimg embed /path/to/images --model openai/clip-vit-large-patch14

# Force reindex (clear existing data first)
localimg embed /path/to/images --reindex

# Increase batch size for faster indexing (if you have enough VRAM)
localimg embed /path/to/images --batch 32

# List available models
localimg --list-models
```

### Web UI

```bash
# Start the web server
localimg ui

# Custom port
localimg ui --port 8080

# Custom host
localimg ui --host 0.0.0.0 --port 8080
```

Then open your browser to `http://127.0.0.1:5000` (or your custom port).

### First Run

1. Start the server: `localimg ui`
2. Open browser to `http://127.0.0.1:5000`
3. Enter a folder path with your images and click "Start Indexing"
4. Once indexing completes, start searching!

### Search Methods

| Method | How |
|--------|-----|
| Text Search | Press `/`, type your query, hit `Enter` |
| Image Search | Drag & drop any image onto the app |
| Duplicate Detection | Click "Dedupe" button |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Open search |
| `Enter` | Execute search |
| `Esc` | Close search / clear results |
| `F` | Toggle fullscreen |
| `R` | Refresh the home sample |
| `Ctrl/Cmd + Click` | Open an image in the lightbox |
| `L` | Cycle the current gallery layout |
| `D` | Find duplicates |
| `T` | Toggle theme |
| `?` | Show shortcuts |
| `Left/Right Arrow` | Navigate images in lightbox |

## Available Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `openai/clip-vit-base-patch32` | ~500MB | Fast | Good |
| `openai/clip-vit-base-patch16` | ~500MB | Fast | Good |
| `openai/clip-vit-large-patch14` | ~1.7GB | Slower | Better |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | ~500MB | Fast | Good |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | ~1.7GB | Slower | Better |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | ~2.5GB | Slowest | Best |

Default model: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any modern CPU | Multi-core |
| RAM | 4GB | 8GB+ |
| GPU | Optional | NVIDIA with CUDA 12.6 |
| VRAM | N/A | 4GB+ for large models |
| Storage | 1GB | 5GB+ for model + cache |
| Python | 3.10 | 3.11-3.13 |

## Performance Notes

### GPU vs CPU

- **GPU (CUDA)**: 10-50x faster embedding, highly recommended for large image collections
- **CPU**: Works but is significantly slower, suitable for small collections (< 1000 images)

### Batch Size

The default batch size is 16. You can increase this if you have enough VRAM:

```bash
# For GPUs with 8GB+ VRAM
localimg embed /path/to/images --batch 64

# For GPUs with 16GB+ VRAM
localimg embed /path/to/images --batch 128
```

Higher batch sizes = faster indexing.

### Model Size

Base models (ViT-B-32) are faster and use less memory. Large models (ViT-L-14) provide better accuracy but are slower and require more VRAM.

## Troubleshooting

### "Running on CPU - this will be slow" Warning

This means PyTorch is using CPU instead of GPU. To fix:

1. Verify you have an NVIDIA GPU: `nvidia-smi`
2. Check CUDA is installed: `nvcc --version`
3. Reinstall PyTorch with CUDA:
   ```bash
   uv pip uninstall torch torchvision
   uv pip install -e .
   ```

### Out of Memory Errors

If you get OOM errors during indexing:

1. Reduce batch size: `localimg embed /path --batch 8`
2. Use a smaller model: `--model openai/clip-vit-base-patch32`
3. Close other applications using GPU memory

### ChromaDB Lock Errors

If you get database lock errors, ensure no other process is using the database:

```bash
# Check for running processes
lsof chromadb/chroma.sqlite3  # Linux/Mac
# or just delete the lock file if you're sure nothing is using it
rm chromadb/chroma.sqlite3-journal
```

## pyproject.toml Configuration

The project uses the following configuration to ensure CUDA-enabled PyTorch is installed:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
```

The `explicit = true` setting ensures that only `torch` and `torchvision` are fetched from the PyTorch index, while all other dependencies come from the default PyPI index.

## Cache Files

Index files are saved as `idx-{timestamp}.pkl` and contain:
- Image embeddings
- File paths
- Model metadata

Caches are compatible across sessions and can be shared (as long as image paths remain valid).

The database is stored in `./chromadb/` and persists between runs. Each model gets its own collection, so you can switch models without losing data.

## License

MIT License
