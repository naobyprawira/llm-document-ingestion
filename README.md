# Document Ingestion Pipelines

**Document Ingestion Pipelines** is a reference implementation of an
end‑to‑end pipeline for transforming PDF documents into a searchable
vector index. The repository is designed to ingest HR and compliance
documents for Indonesian companies and supports heavy flowchart
detection, vision‑language model (VLM) descriptions, structured and
smart text chunking, vector embeddings via Gemini, and storage in a
Qdrant database. The code is written in Python 3.12 and is published
under the MIT license.

## Features

* **Heavy flowchart detection:** Pages are analysed at the vector level
  using PyMuPDF to count rectangles, ellipses, diamonds and
  orthogonal connectors. If a page likely contains a flowchart it
  will be rendered to a PNG and passed to Google's Gemini VLM to
  produce a numbered description of the process.
* **Vision–language integration:** The VLM is prompted to return
  strictly formatted JSON containing `has_flowchart`, `title`,
  `steps_text` and `confidence`. Only pages that both pass the
  detection heuristic and contain at least two numbered steps are
  considered valid flowcharts.
* **Structured chunking:** Text pages are chunked according to
  hierarchical headings of the form `6`, `6.1`, `6.1.1`. Only the
  first three levels are retained; deeper levels such as `6.1.1.1`
  are grouped into their immediate parent (`6.1.1`). Each chunk
  preserves its heading for downstream retrieval. If a page contains
  no headings it falls back to **smart chunking**.
* **Smart chunking:** When headings are absent or unreliable, text is
  grouped by paragraphs until an approximate token limit is reached
  (default 900 words). This prevents overly long chunks while
  respecting paragraph boundaries.
* **Gemini embeddings:** Text and flowchart descriptions are
  embedded using Google’s [text‑embedding‑004](https://ai.google.dev/docs/text_embedding_overview)
  model via the Generative AI API. If no API key is provided the
  pipeline transparently falls back to a local
  [sentence-transformers](https://www.sbert.net/) model.
* **Metadata‑rich payloads:** Each vector stores metadata fields
  including `doc_id`, `scope`, `company`, `country`, `language`,
  `version`, `page`, `chunk_id`, `heading` and validity dates. This
  enables fine‑grained filtering in downstream applications.
* **Qdrant integration:** Vectors and payloads are stored in a
  single Qdrant collection. The code automatically ensures the
  collection exists and creates payload indexes for fast filtering.
* **Hard deletion:** Documents or individual pages can be removed by
  their identifiers. There is no soft deletion mechanism; points are
  permanently deleted.

## Getting Started

### Prerequisites

* Python 3.12
* Access to the [Google Generative AI API](https://ai.google.dev/) for
  flowchart descriptions and embeddings (optional but recommended).
* A running Qdrant instance. You can either
  [provision a managed cluster](https://cloud.qdrant.io/) (recommended
  for production) or run the included Docker Compose setup for local
  development.

### Installation

1. **Clone this repository** and enter the project directory:

   ```bash
   git clone <repo-url> document-ingestion-pipelines
   cd document-ingestion-pipelines
   ```

2. **Install dependencies** using pip. It is recommended to do this
   within a virtual environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root (or export these
   variables in your environment). At a minimum set your Google API
   key and Qdrant URL:

   ```env
   # .env
   QDRANT_URL=https://5f76a8f4-bbac-4a92-92e2-9174b63fc76a.us-west-2-0.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=
   QDRANT_COLLECTION=prb-documents-gemini
   GOOGLE_API_KEY=your-google-api-key
   DEFAULT_LANGUAGE=id
   IMAGE_DPI=350
   MAX_TOKENS_PER_CHUNK=900
   ```

   If you don’t provide `GOOGLE_API_KEY` the pipeline will still run
   but will use a local embedding model and skip flowchart extraction.

### Ingesting a Document

Run the `ingest_pdf.py` CLI from the project root. The most
important parameters are the PDF file path, document identifier,
scope, company and country:

```bash
python cli/ingest_pdf.py \
  --file path/to/document.pdf \
  --doc-id PRB-REG-001 \
  --scope Internal \
  --company PRB \
  --country ID \
  --version v3.2 \
  --effective-from 2025-01-01
```

**Parameter definitions:**

| Parameter | Description |
|----------|-------------|
| `--file` | Path to the PDF file to ingest. |
| `--doc-id` | Unique identifier for the document. |
| `--scope` | One of `GovReg`, `AdidasCompliance` or `Internal`. |
| `--company` | Company code (`PRB`, `PBB`, `GROUP`). Always required but ignored for non‑Internal scopes. |
| `--country` | Country code (e.g. `ID` or `GLOBAL`). |
| `--language` | Language code (defaults to `id` if omitted). |
| `--version` | Arbitrary version string. |
| `--effective-from` | Start date (ISO 8601) from which the document is valid. |
| `--effective-to` | End date (ISO 8601) until which the document is valid. Leave blank for no expiry. |
| `--security-class` | Label such as `internal`, `confidential`. |

The command prints a JSON summary of ingested pages, listing
flowcharts detected and text chunks created.

### Deleting Data

Use `delete_docs.py` to remove ingested data. You can delete an
entire document or a single page:

```bash
python cli/delete_docs.py --doc-id PRB-REG-001        # delete entire document
python cli/delete_docs.py --doc-id PRB-REG-001 --page 3  # delete page 3 only
```

Deletion is permanent; there is no soft delete. Use with caution.

### Running with Docker Compose

This project includes a `docker-compose.yml` file that spins up a
local Qdrant server and builds the application image. To run the
pipeline against the local Qdrant instance:

```bash
export GOOGLE_API_KEY=your-google-api-key
docker compose up --build
```

You can then run ingestion inside the `app` service by overriding
the default command. See the comments in `docker-compose.yml` for
examples.

## How It Works

### Flowchart Detection and Extraction

For each page, the pipeline counts rectangles, ellipses, diamonds and
lines extracted from the PDF's vector layer. Diamonds are detected
from four‑point paths rotated approximately 45 °. A simple score is
computed and pages with at least two shapes and connectors are
flagged as flowchart candidates. Candidate pages are rendered to
PNG (default 350 DPI) and sent to the Gemini VLM with a tightly
constrained prompt asking for numbered steps. The model returns
JSON; only results with `has_flowchart = true` and at least two
numbered lines are kept.

### Chunking Strategy

If a page does not contain a flowchart, its text is extracted via
PyMuPDF. The structured chunker splits the text into sections based
on numbered headings matching the regex `^\d+(\.\d+){0,2} `. Only
headings up to three levels deep are retained; deeper headings such
as `6.1.1.1` are grouped into their parent `6.1.1`. Each (heading,
body) pair becomes a chunk. If no headings are found on a page the
fallback smart chunker groups paragraphs until a token limit is
reached.

### Embeddings and Storage

Both flowchart step descriptions and text chunks are embedded via the
Google text embedding API. If the API key is unavailable or a
request fails the pipeline falls back to a local sentence‑transformer
model. The resulting vectors and payloads are upserted into the
configured Qdrant collection with cosine distance and a memmap
optimizer. Payload indexes are created automatically for all
metadata fields used by the pipeline.

## Extending

The repository is designed to be easy to adapt to new use cases:

* **Alternative embedding models:** Update `app/embeddings.py` to call
  a different API or model. Only the vector dimension needs to be
  propagated to Qdrant.
* **Custom chunking:** Implement your own chunkers in `app/chunking_*` and
  call them from `app/ingestion.py`.
* **Soft deletion:** You can modify `app/deletion.py` to set
  `effective_to` instead of hard deleting points if you need
  non‑destructive updates.

## License

This project is licensed under the MIT License. See the
[`LICENSE`](LICENSE) file for details.