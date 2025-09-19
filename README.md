# Document Ingestion API

A fast, production-ready pipeline to **parse → chunk → embed** documents into **Qdrant**, with optional **asynchronous fire-and-poll** semantics for workflow tools like **n8n**.

This repository contains:

- `src/api.py` — a FastAPI app exposing a **single** `/process` endpoint with **two modes**  
  - **Sync (default)**: parse → chunk → embed, then respond `200 OK`  
  - **Async (`async=1`)**: respond `202 Accepted` immediately with a `job_key`, then continue work in the background and **write a status “job marker” point** into Qdrant that you can **poll** from n8n  
- `src/vlm.py` — a Gemini (Google GenAI) **vision** wrapper used during parsing to describe images/figures, with **streaming** and **transient-only retries**  
- Supporting modules: `doc_parser`, `markdown`, `chunk`, `embed`, `prompts`, etc.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Run](#run)
- [API](#api)
  - [/process (sync)](#process-sync)
  - [/process (async fire--poll)](#process-async-fire--poll)
- [Concurrency & batching](#concurrency--batching)
- [Qdrant schema & job markers](#qdrant-schema--job-markers)
- [Polling from n8n](#polling-from-n8n)
- [Authentication for Qdrant Cloud](#authentication-for-qdrant-cloud)
- [Logging](#logging)
- [Performance tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [Docker Compose example](#docker-compose-example)
- [Reference links](#reference-links)

---

## Features

- **Single endpoint** UX: `/process` handles end-to-end ingestion
- **Async option** (`async=1`) returns immediately (`202`) and continues work in a **FastAPI BackgroundTask**; progress is exposed as a **Qdrant point** you can poll, so **no extra endpoints** are required.
- **Resource aware**: parsing and VLM calls gated by semaphores; embeddings upserted in **batches**
- **Image-aware parsing**: PDFs → text + figure descriptions (via Gemini)
- **n8n-friendly**: simple loop `[HTTP Request → IF → Wait]` to poll a Qdrant **job marker**
- **Structured logs** by phase: `parse / chunk / embed` with `job=<uuid> file=<name>`

---

## Architecture

```
Client (e.g., n8n) ──POST /process──────────────────────────────► FastAPI
            ▲               │
            │               ├─ parse (Docling)  [gated by semaphore]
            │               ├─ VLM describe (Gemini) [gated]
            │               ├─ chunk (sync in worker thread)
            │               └─ embed (Qdrant upsert, batched)
            │
            └─ (async mode) ◄─ 202 {job_key}
                              └─ BackgroundTask continues and writes
                                 a "job_marker" point into Qdrant:
                                 payload = {type:"job_marker", job_key, status:"running|done|failed", ...}
```

- **BackgroundTasks** are executed **after** FastAPI returns the response. Use them when the work can run out-of-band and you don’t need same-process return values.  
- CPU-bound or blocking code (Docling, embeddings) is run in worker threads via `anyio.to_thread.run_sync`.

---

## Requirements

- Python 3.10+
- FastAPI + Uvicorn
- anyio
- Qdrant Client (`qdrant-client`)
- Google GenAI SDK (`google-genai`) with a valid API key
- (Optional) Tenacity (if you enable additional retry helpers in your own modules)

---

## Installation

```bash
pip install -r requirements.txt
# or explicit:
pip install fastapi uvicorn anyio qdrant-client google-genai
```

---

## Environment variables

| Key | Purpose | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google GenAI API key | — |
| `GEMINI_MODEL` | Vision model (e.g., `gemini-2.5-flash`) | `gemini-2.5-flash` |
| `GENAI_TIMEOUT_MS` | SDK client timeout (ms) | unset |
| `MAX_CONCURRENT_PARSE` | Gate concurrent parsing | `3` |
| `MAX_CONCURRENT_VLM` | Gate concurrent VLM calls | `5` |
| `EMBED_BATCH_SIZE` | Qdrant upsert batch size | `64` |
| `QDRANT_URL` | e.g., `https://<cluster>.cloud.qdrant.io:6333` | — |
| `QDRANT_API_KEY` | Qdrant Cloud DB API key | — |
| `QDRANT_COLLECTION` | Collection name for chunks | `documents` |
| `LOG_LEVEL` | `DEBUG|INFO|WARNING|ERROR` | `INFO` |

> Qdrant Cloud **requires** an API key on every request; use header `api-key: <key>` (or `Authorization: apikey <key>` over the Cloud API).

---

## Run

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080 --workers 1
```

- **Workers**: keep `1` unless you’ve audited thread-safety and external rate limits. The app already uses background threads for blocking work.

---

## API

### `/process` (sync)

**Request**

- `multipart/form-data` with:
  - `file`: PDF or image
  - `category`: free-form string

**Response `200 OK`**

```json
{
  "file_name": "sample.pdf",
  "file_type": "pdf",
  "page_count": 12,
  "images_count": 5,
  "uploaded_chunks": 128,
  "collection": "documents",
  "total_processing_seconds": 9.42,
  "category": "hr-policy"
}
```

---

### `/process` (async fire-&-poll)

Send the same request but include `async_mode=1` form field.

**Immediate response `202 Accepted`**

```json
{
  "status": "accepted",
  "job_key": "b0b7b6e3e6b448c0b2e0a20d85ef1c2a",
  "filename": "sample.pdf",
  "category": "hr-policy",
  "collection": "documents"
}
```

The pipeline then continues in a **BackgroundTask** and writes a **job marker** into Qdrant that you can poll. (Background tasks execute **after** the response is sent.)

---

## Concurrency & batching

- **Parsing gate**: `MAX_CONCURRENT_PARSE` (async semaphore)
- **VLM gate**: `MAX_CONCURRENT_VLM` to avoid API rate spikes (another semaphore)
- **Embeddings**: `EMBED_BATCH_SIZE` controls upsert batch size to Qdrant
- Blocking work uses `anyio.to_thread.run_sync`, which is the recommended pattern for calling sync code inside async apps.

---

## Qdrant schema & job markers

- Each **chunk point** has payload like:
  ```json
  {"filename":"sample.pdf","category":"hr-policy","chunk_index":42,"dim":768}
  ```
- Each **job marker** point (single, upserted repeatedly) has payload:
  ```json
  {
    "type":"job_marker",
    "job_key":"<uuid>",
    "filename":"sample.pdf",
    "category":"hr-policy",
    "collection":"documents",
    "status":"running|done|failed",
    "expected_chunks": 128,
    "uploaded_chunks": 64,
    "finished_at": "2025-09-19T15:12:03.123Z"   // only when done/failed
  }
  ```

**Why Scroll?** Use **Scroll** (not search) to fetch points **by payload filter** without vectors. The Scroll API pages results, supports filters, and returns `next_page_offset` for pagination.

---

## Polling from n8n

You can implement a compact loop: **HTTP Request → IF → Wait → (loop back to HTTP Request)**.

1) **HTTP Request (POST)** to Qdrant **Scroll**:

- **URL**
  ```
  https://<your-cluster>.cloud.qdrant.io:6333/collections/{{$json["collection"]}}/points/scroll
  ```

  If the collection value lives on a previous node named `HTTP Request6`, use:
  ```
  https://<your-cluster>.cloud.qdrant.io:6333/collections/{{$node["HTTP Request6"].json["collection"]}}/points/scroll
  ```

- **Headers**
  ```
  api-key: {{ $env.QDRANT_API_KEY || 'paste-your-key' }}
  content-type: application/json
  ```

- **Body (raw JSON)**
  ```json
  {
    "limit": 1,
    "with_payload": true,
    "filter": {
      "must": [
        {"key":"type","match":{"value":"job_marker"}},
        {"key":"job_key","match":{"value":"{{$json["job_key"]}}"}}
      ]
    }
  }
  ```

2) **IF** — check `{{$json["result"]["points"][0]["payload"]["status"]}}`  
   - If `== "done"` → proceed  
   - If `== "failed"` → handle error  
   - Else → go to **Wait**

3) **Wait** — 5–10 seconds, then connect back to the **HTTP Request** node to poll again (simple loop).

> Tip: If you ingest multiple files, either run them **sequentially** in n8n or set **small concurrency** in your workflow so you don’t trip upstream rate limits.

---

## Authentication for Qdrant Cloud

For **every** REST request to Qdrant Cloud, include one of:

- `api-key: <YOUR_DB_API_KEY>`  
- or `Authorization: apikey <YOUR_MANAGEMENT_KEY>` (Cloud Management API)

Both are supported; see Qdrant Cloud documentation for details.

---

## Logging

- The API emits structured logs:  
  `timestamp level ingestion job=<uuid> file=<name> phase=<parse|chunk|embed> msg=<...>`
- To silence Google GenAI SDK logs globally in your app (optional):
  ```python
  import logging; [logging.getLogger(n).setLevel(logging.CRITICAL) for n in ("google_genai","google_genai.models")]
  ```

---

## Performance tuning

- **Semaphores**: reduce `MAX_CONCURRENT_VLM` if you see 429s from the model API; increase cautiously
- **Batching**: adjust `EMBED_BATCH_SIZE` to trade memory for throughput
- **AnyIO threads**: keep long CPU/blocking calls in worker threads (`to_thread.run_sync`) for smooth async I/O

---

## Troubleshooting

- **n8n HTTP 404/503 via ngrok**: ngrok sometimes returns gateway pages when upstream is unreachable or responds with incomplete HTTP (ERR_NGROK_3004). Verify your FastAPI server is reachable and your tunnel is healthy.
- **OpenSSL “decryption failed or bad record mac”**: typically a TLS connection problem (proxy or interrupted TLS). Re-try with HTTPS directly to Qdrant Cloud and ensure the correct port (`6333`/HTTPS).
- **Torch `pin_memory` warning on CPU**: safe to ignore on CPU-only runs; consider filtering it out if noisy.
- **FastAPI BackgroundTasks don’t “block”**: by design, tasks run **after** the response is sent; if you need to wait, use the **sync** mode (no `async=1`).

---

## Security notes

- Never log secrets (`GOOGLE_API_KEY`, `QDRANT_API_KEY`).
- Use **TLS** (your Qdrant Cloud endpoint is HTTPS) and keep API keys in environment variables.

---

## Docker Compose example

```yaml
services:
  ingestion-api:
    build: .
    image: your-org/ingestion-api:latest
    ports:
      - "8080:8080"
    environment:
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
      GEMINI_MODEL: gemini-2.5-flash
      GENAI_TIMEOUT_MS: "60000"
      MAX_CONCURRENT_PARSE: "3"
      MAX_CONCURRENT_VLM: "5"
      EMBED_BATCH_SIZE: "64"
      QDRANT_URL: "https://<cluster>.cloud.qdrant.io:6333"
      QDRANT_API_KEY: ${QDRANT_API_KEY}
      QDRANT_COLLECTION: "documents"
      LOG_LEVEL: "INFO"
    command: ["uvicorn","src.api:app","--host","0.0.0.0","--port","8080","--workers","1"]
```

---

## Reference links

- FastAPI BackgroundTasks: https://fastapi.tiangolo.com/tutorial/background-tasks/
- AnyIO `to_thread.run_sync`: https://anyio.readthedocs.io/en/stable/api.html#anyio.to_thread.run_sync
- Qdrant Scroll Points API: https://api.qdrant.tech/api-reference/points/scroll-points
- Qdrant filtering & payload conditions: https://qdrant.tech/documentation/concepts/filtering/
- Qdrant Cloud authentication (DB keys): https://qdrant.tech/documentation/cloud/authentication/
- n8n Expressions: https://docs.n8n.io/code/expressions/
