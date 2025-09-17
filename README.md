Simple Document Ingestion with Docling and VLM

Convert PDFs to Markdown with image descriptions using Vision Language Models.

## How it works

- Extract clean Markdown from PDFs using Docling
- Describe images separately with Gemini vision model
- Append image descriptions to the final output
- No complex integrations or brittle dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your `GOOGLE_API_KEY`.

## Usage (CLI)

```bash
python -m llm_document_ingestion.src.main path/to/file.pdf --out output.md
```
