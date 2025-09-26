# LLM Document Ingestion

This repository contains a streamlined pipeline for ingesting documents into a Qdrant vector database.  It accepts PDF or image files, extracts their text and figures, and stores embeddings for later retrieval.

Key features:

- **Simple API** – a single `/process` endpoint receives your file and a category.  If a document with the same filename has already been processed it will be skipped automatically.
- **Synchronous pipeline** – all processing runs in normal Python functions; no async/await gymnastics required.  The only parallelism is used internally when describing images.
- **Gemini powered** – uses Google’s *Gemini 2‑5 flash‑lite* model for vision tasks and *gemini‑embedding‑001* for text embeddings.
- **Metadata rich** – each stored embedding carries the original text and category at the top of its payload along with the filename and chunk index, making it easy to filter and inspect.
- **Indexed fields** – the pipeline ensures that payload indexes for `filename` and `category` are created so you can filter your data efficiently.

## Usage

1. Install the dependencies listed in `requirements.txt`.
2. Export your `GOOGLE_API_KEY` and `QDRANT_URL`/`QDRANT_API_KEY` environment variables.
3. Run the API with Uvicorn:

```bash
uvicorn llm_document_ingestion.src.api:app --reload
```

POST a file to `/process` with a `category` form field.  The API will either process it synchronously (default) or skip it if the same filename already exists.

## Development notes

The code lives under `llm_document_ingestion/src`.  The main entry point is `api.py` which performs a pre‑flight check in Qdrant to avoid re‑processing duplicates.  The ingestion pipeline itself is in `pipeline_sync.py`.  See `qdrant_utils.py` for helpers that connect to Qdrant and construct job markers.
