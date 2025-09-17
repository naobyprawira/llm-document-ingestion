"""
Simple FastAPI application exposing a single ingestion endpoint.

This module mirrors the upstream implementation and provides a
``/ingest`` endpoint that accepts a file upload, runs the full
pipeline and returns the enriched Markdown. It's kept here for
compatibility but is superseded by the more feature‑rich ``api.py``.
"""

from __future__ import annotations

import os
import platform
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

# Load .env before importing modules that read env at import time
load_dotenv()

from .pipeline import run

app = FastAPI(title="Simple VLM+Docling")


@app.get("/healthz")
def healthz():
    """Return simple health information, including library versions."""
    info = {
        "python": platform.python_version(),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
    }
    try:
        import docling  # type: ignore
        import docling_core  # type: ignore
        import google.genai as gg  # type: ignore
        info["docling"] = getattr(docling, "__version__", "present")
        info["docling_core"] = getattr(docling_core, "__version__", "present")
        info["google_genai"] = getattr(gg, "__version__", "present")
    except Exception as e:
        info["error"] = str(e)
    return JSONResponse(info)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Accept a document, run the pipeline and return enriched Markdown."""
    try:
        suffix = os.path.splitext(file.filename or "doc.pdf")[1] or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = run(tmp_path)
        return PlainTextResponse(result.markdown, media_type="text/markdown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm_document_ingestion.src.web:app", host="0.0.0.0", port=8000, reload=True)