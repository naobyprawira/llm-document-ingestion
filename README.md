# Document Ingestion API (synchronous)

This repository contains a small but capable document ingestion pipeline that takes an input PDF or image, extracts text and figures, describes the figures using a visual language model (Google Gemini), splits the resulting markdown into chunks and stores embeddings in a [Qdrant](https://qdrant.tech/) vector database.  The design here emphasises **simplicity** and **data quality**:

- All of the heavy lifting is performed synchronously.  There are **no `async` functions** in the ingestion code – everything runs in normal Python threads.  FastAPI still supports asynchronous request handling under the hood, but the pipeline itself does not require an event loop.  This makes it easy to reason about, debug and deploy.
- The only concurrency in the pipeline is used when calling the Gemini vision model (VLM).  A thread pool with a configurable maximum number of workers fans out calls to describe multiple figures in parallel.  The default limit is five concurrent VLM calls, which respects Google’s free tier rate limits.  All other stages (parsing, chunking, embedding and Qdrant upserts) run serially.
- The application has been decomposed into small modules (`parser_utils.py`, `qdrant_utils.py`, `pipeline_sync.py`, etc.) to improve readability and testability.  The old monolithic `api.py` has been reduced to a thin FastAPI wrapper that delegates work to these modules.
- Additional data‑quality helpers have been added.  The markdown emitted by the parser is cleaned to collapse excessive whitespace and remove trailing spaces before it is chunked.  The VLM description helper retries very short or empty responses up to three times before giving up.

## Quickstart

Install the dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

Run the API with Uvicorn:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Send a multipart POST request to `/process` with a `file` and a `category` field.  The API returns JSON containing basic information about the processed document.  If you include `async_mode=1` the call returns immediately with a `job_key` and continues processing in the background; otherwise it blocks until the pipeline has finished.

### Environment variables

| Key | Purpose | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google GenAI API key | – |
| `GEMINI_MODEL` | Vision model (e.g. `gemini-2.5-flash`) | `gemini-2.5-flash` |
| `MAX_CONCURRENT_VLM` | Maximum concurrent VLM calls | `5` |
| `EMBED_BATCH_SIZE` | Qdrant upsert batch size | `64` |
| `CHUNK_MAX_CHARS` | Maximum characters per markdown chunk | `1024` |
| `QDRANT_URL` | Base URL for your Qdrant instance | – |
| `QDRANT_API_KEY` | Qdrant Cloud API key | – |
| `QDRANT_COLLECTION` | Name of the Qdrant collection | `documents` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`,`INFO`,…) | `INFO` |

## Project structure

- `src/api.py` – FastAPI app exposing the `/process` endpoint.  It reads files, determines whether to run synchronously or in a background task, and calls into the synchronous pipeline.
- `src/parser_utils.py` – Helper functions to parse PDFs/images into markdown and extract figure metadata.  It also contains concurrency logic for describing figures using Gemini.
- `src/qdrant_utils.py` – Helpers for connecting to Qdrant, ensuring collections exist and constructing job marker points.
- `src/pipeline_sync.py` – The orchestrator.  Given raw bytes, a filename and metadata, it runs parse → clean → chunk → embed and returns a dictionary of results.
- `src/doc_parser.py`, `src/markdown.py`, `src/chunk.py`, `src/embed.py`, `src/vlm.py`, `src/prompts.py`, `src/util_img.py` – Upstream modules, largely unmodified.
- `src/main.py` – Command‑line entry point that runs the synchronous pipeline over a PDF and writes the enriched markdown to disk.

## Changes from the upstream implementation

The original project exposed a single asynchronous FastAPI endpoint with fine‑grained semaphores controlling each stage of the pipeline.  This fork simplifies the concurrency model and improves data quality:

- **Synchronous execution** – all functions are now normal (blocking) Python functions.  FastAPI’s threadpool support handles concurrency behind the scenes.  This eliminates the complexity of managing `async` functions, semaphores and `anyio`.
- **Threaded VLM calls** – only the calls to `GeminiVLM.describe()` are made in parallel.  A `ThreadPoolExecutor` respects the `MAX_CONCURRENT_VLM` limit and fans out descriptions across multiple figures, maximising throughput without exceeding rate limits.
- **Markdown cleaning** – the pipeline collapses multiple blank lines and strips trailing whitespace before chunking, producing more consistent chunks and more robust embeddings.
- **Retry logic for figure descriptions** – very short or empty descriptions are retried up to three times.
- **Modular codebase** – code that previously lived in `api.py` has been broken into modules.  This makes it easier to test individual pieces and to swap out components (e.g. using a different VLM or embedding model).

## License

This project is provided as‑is under the MIT licence.  See `LICENSE` for details.