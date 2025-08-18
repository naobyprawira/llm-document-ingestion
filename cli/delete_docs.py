"""
Command-line interface for deleting ingested documents or specific pages.

This script wraps the deletion utilities in ``app.deletion``. It
provides two mutually exclusive modes:

1. Delete an entire document by its document ID.
2. Delete a specific page of a document by supplying both the document
   ID and the page number.

Deletion is permanent and irreversible. Use with caution.
"""
from __future__ import annotations

import click

from app.config import load_config
from app.qdrant_client import get_client
from app.deletion import delete_document, delete_document_page


@click.command()
@click.option("--doc-id", type=str, required=True, help="Document identifier used during ingestion.")
@click.option("--page", type=int, default=None, help="Page number to delete (1-based). If omitted, deletes the entire document.")
def main(doc_id: str, page: int | None) -> None:
    """Delete a document or a specific page from Qdrant."""
    config = load_config()
    client = get_client(config)
    if page is not None:
        delete_document_page(client, config, doc_id, page)
        click.echo(f"Deleted page {page} of document {doc_id}")
    else:
        delete_document(client, config, doc_id)
        click.echo(f"Deleted document {doc_id}")


if __name__ == "__main__":
    main()