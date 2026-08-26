# fernsearch

Fern is a beautiful media search engine on steroids with multimodal embeddings, smart frame sampling and VLM reranking.

![alt text](screenshot.png)


## Usage

### CLI Commands

```bash
# Index images from a folder
fern embed /path/to/images

# Search for images
fern search "red car"
```

```sh
# Use a different CLIP model
fern embed /path/to/images \
--model openai/clip-vit-large-patch14 \
--batch 32

# Force reindex (clear existing data first)
fern embed /path/to/images --reindex

# List available models
fern --list-models
```

### UI 

Start the web server

```bash
fern serve
```
