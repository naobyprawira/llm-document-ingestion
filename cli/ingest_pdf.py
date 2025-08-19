"""Command line interface for ingesting PDF documents.

This script provides a convenient wrapper around :func:`app.pipelines.ingest_pdf.ingest_document`
allowing users to ingest a PDF into Qdrant from the command line.  It supports
resuming from a given page, limiting the number of pages to ingest, and
performing a dry run where no data is written to the database.
"""

import uuid
import click

from app.config import load_config
from app.pipelines.ingest_pdf import ingest_document


@click.command(help="Ingest a PDF document into Qdrant")
@click.option("--path", "path_", type=click.Path(exists=True), required=True, help="Path to the PDF file to ingest.")
@click.option("--doc-id", type=str, default=None, help="Identifier for the document; if omitted a UUID will be generated.")
@click.option("--scope", type=str, required=True, help="Top‑level scope of the document, e.g. GovReg, AdidasCompliance, Internal.")
@click.option("--company", type=str, required=True, help="Company identifier (e.g. PRB, PBB, GROUP). Ignored for non‑internal scopes.")
@click.option("--country", type=str, required=True, help="Country code (e.g. ID or GLOBAL).")
@click.option("--language", type=str, default=None, help="Language code of the document. Defaults to configuration default.")
@click.option("--version", type=str, default="v1", help="Version identifier of the document.")
@click.option("--effective-from", type=str, default="1970-01-01", help="ISO date from which the document is effective.")
@click.option("--effective-to", type=str, default=None, help="ISO date until which the document is effective. None means indefinite.")
@click.option("--security-class", type=str, default="internal", help="Security classification, e.g. internal, confidential.")
@click.option("--start-page", type=int, default=1, help="1‑indexed page number from which to start ingestion.")
@click.option("--max-pages", type=int, default=None, help="Maximum number of pages to ingest.")
@click.option("--dry-run", is_flag=True, help="When set, perform a dry run without writing to Qdrant.")
def main(
    path_: str,
    doc_id: str,
    scope: str,
    company: str,
    country: str,
    language: str,
    version: str,
    effective_from: str,
    effective_to: str,
    security_class: str,
    start_page: int,
    max_pages: int,
    dry_run: bool,
) -> None:
    """
    Ingest a PDF file into a Qdrant collection using the configured pipeline.

    Parameters are provided via command line options.  See ``--help`` for
    details on each option.
    """
    config = load_config()
    # Generate a document ID if not provided
    if not doc_id:
        doc_id = str(uuid.uuid4())
    # Call the ingestion function
    result = ingest_document(
        filepath=path_,
        doc_id=doc_id,
        scope=scope,
        company=company,
        country=country,
        language=language,
        version=version,
        effective_from=effective_from,
        effective_to=effective_to,
        security_class=security_class,
        start_page=start_page,
        max_pages=max_pages,
        dry_run=dry_run,
        config=config,
    )
    click.echo(f"Ingested document {doc_id} with {len(result['flowcharts'])} flowchart(s) and {len(result['text_chunks'])} text chunk(s)")


if __name__ == "__main__":
    main()