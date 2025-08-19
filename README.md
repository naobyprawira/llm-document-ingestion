# Document Ingestion Pipeline

This project ingests HR/compliance documents and turns them into searchable vector
representations stored in Qdrant.  The pipeline detects flow charts on PDF
pages, extracts and chunks text intelligently, embeds the resulting chunks
using a single generative embedding model, and finally writes the vectors
with rich metadata to a Qdrant collection.

## Overview

1. **Flowchart detection** – Each page of a PDF is analysed to determine
   whether it contains a flow chart.  Detected flow charts are passed to
   a VLM (Gemini) which returns a strict JSON structure describing the
   steps; these steps are treated as text chunks.
2. **Text extraction and chunking** – Pages that are not flow charts are
   parsed for text.  The text is split either on numbered headings (e.g.
   `6`, `6.1`, `6.1.1`) or, if no headings exist, via a paragraph‑aware
   “smart chunker” that groups paragraphs up to configurable word limits.
3. **Embedding** – All text chunks, whether from flow charts or body text,
   are embedded with Google’s `models/gemini-embedding-001` model.  A
   deterministic point identifier derived from the document ID, page and
   chunk index is used for each vector to ensure idempotent upserts.
4. **Storage** – Vectors and associated metadata (document ID, page,
   heading, chunk index, scope, company, country, language, and effective
   dates) are stored in a Qdrant collection.  The storage layer exposes
   helpers to create collections, upsert points and delete (hard/soft)
   documents.

## Packages

The code is organised into several subpackages under `app`:

- `parsers/` – PDF parsing and optional OCR.
- `chunking/` – Structured and smart chunking functions.
- `embeddings/` – Embedding providers.
- `storage/` – Qdrant client utilities.
- `utils/` – Shared helpers such as deterministic ID generation.
- `pipelines/` – Orchestration logic for ingesting a PDF.
- `vision/` – Flowchart detection and VLM integration.

CLI wrappers live under `cli/` (to be implemented) and provide commands
for ingesting documents and deleting them from Qdrant with optional
dry‑run and soft‑delete behaviour.

## Installation

Install the dependencies and this package in editable mode:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Environment variables defined in `.env` or your shell control the behaviour
of the ingestion pipeline (e.g. embedding model names, Qdrant host/port,
maximum chunk size).  Once configured, run the CLI script to ingest
documents:

```bash
# Example usage; CLI script lives in cli/ingest_pdf.py
python -m cli.ingest_pdf --path /path/to/file.pdf --collection hr_docs --scope compliance
```

Deleting a document is similarly accomplished via a CLI command (to be
implemented).

## Contributing

Pull requests and issues are welcome.  Contributions should adhere to the
existing coding style and include tests where possible.
