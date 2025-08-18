"""
Command-line interface for ingesting PDF documents.

This script wraps the :func:`app.ingestion.ingest_document` function
so that it can be invoked from the command line. It reads command
line arguments using the ``click`` library, loads application
configuration, and passes the parameters into the ingestion pipeline.

Example usage::

    python ingest_pdf.py --file path/to/doc.pdf --doc-id DOC123 \
        --scope Internal --company PRB --country ID --version v3.2 \
        --effective-from 2025-01-01

The scope and company fields are mandatory. The company should be
provided even for non-internal documents but will be ignored by the
pipeline for other scopes.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import click

from app.config import load_config
from app.ingestion import ingest_document


@click.command()
@click.option("--file", "filepath", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the PDF file to ingest.")
@click.option("--doc-id", type=str, required=True, help="Unique document identifier.")
@click.option("--scope", type=str, required=True, help="Top-level scope (GovReg, AdidasCompliance, Internal).")
@click.option("--company", type=str, required=True, help="Company code (PRB, PBB, GROUP).")
@click.option("--country", type=str, required=True, help="Country code (e.g. ID or GLOBAL).")
@click.option("--language", type=str, default=None, help="Language code of the document content (default from config).")
@click.option("--version", type=str, default="v1", help="Version of the document.")
@click.option("--effective-from", type=str, default="1970-01-01", help="Effective from date (ISO 8601).")
@click.option("--effective-to", type=str, default=None, help="Effective to date (ISO 8601), leave blank for indefinite.")
@click.option("--security-class", type=str, default="internal", help="Security classification (internal, confidential, etc.).")
def main(
    filepath: str,
    doc_id: str,
    scope: str,
    company: str,
    country: str,
    language: Optional[str],
    version: str,
    effective_from: str,
    effective_to: Optional[str],
    security_class: str,
) -> None:
    """Ingest a PDF into Qdrant via the configured pipeline."""
    config = load_config()
    result = ingest_document(
        filepath=filepath,
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
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()