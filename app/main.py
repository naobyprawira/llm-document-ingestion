"""
FastAPI application exposing the ingestion and deletion pipelines.

This module defines three endpoints:

* ``POST /ingest/pdf`` – Ingest a PDF document into Qdrant. Accepts
  multipart form data with the PDF file and metadata fields.
* ``DELETE /docs/{doc_id}`` – Hard delete all data belonging to a
  document.
* ``DELETE /docs/{doc_id}/pages/{page}`` – Hard delete a specific page
  of a document.

The API relies on the underlying functions in ``app.ingestion`` and
``app.deletion``. It is intended for integration with other
services or automation tools such as Postman.
"""
from __future__ import annotations

import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import load_config
from .ingestion import ingest_document
from .qdrant_client import get_client
from .deletion import delete_document, delete_document_page

logger = logging.getLogger(__name__)

app = FastAPI(title="Document Ingestion API", version="1.0.0")


@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    scope: str = Form(...),
    company: str = Form(...),
    country: str = Form(...),
    language: str | None = Form(None),
    version: str = Form("v1"),
    effective_from: str = Form("1970-01-01"),
    effective_to: str | None = Form(None),
    security_class: str = Form("internal"),
) -> JSONResponse:
    """Ingest a PDF document via HTTP."""
    content = await file.read()
    # Save uploaded file to a temporary location on disk. The ingestion
    # function requires a file path. Use NamedTemporaryFile to avoid
    # collisions.
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    config = load_config()
    try:
        result = ingest_document(
            filepath=tmp_path,
            doc_id=doc_id,
            scope=scope,
            company=company,
            country=country,
            language=language,
            version=version,
            effective_from=effective_from,
            effective_to=effective_to,
            security_class=security_class,
            config=config,
        )
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always remove the temporary file
        import os
        os.unlink(tmp_path)
    return JSONResponse(content=result)


@app.delete("/docs/{doc_id}")
async def delete_doc_endpoint(doc_id: str) -> JSONResponse:
    """Delete an entire document."""
    config = load_config()
    client = get_client(config)
    try:
        delete_document(client, config, doc_id)
    except Exception as e:
        logger.exception("Deletion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"deleted": True, "doc_id": doc_id})


@app.delete("/docs/{doc_id}/pages/{page}")
async def delete_doc_page_endpoint(doc_id: str, page: int) -> JSONResponse:
    """Delete a specific page of a document."""
    config = load_config()
    client = get_client(config)
    try:
        delete_document_page(client, config, doc_id, page)
    except Exception as e:
        logger.exception("Deletion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"deleted": True, "doc_id": doc_id, "page": page})
