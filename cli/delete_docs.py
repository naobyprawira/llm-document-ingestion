"""Command line interface for deleting documents or pages from Qdrant.

This script wraps the deletion helpers in :mod:`app.storage.qdrant_store` and
provides both hard and soft delete options.  You can delete all points
belonging to a document or restrict deletion to a particular page.
"""

import click

from app.config import load_config
from app.storage.qdrant_store import get_client, delete_document, delete_document_page


@click.command(help="Delete a document or a page from Qdrant")
@click.option("--doc-id", type=str, required=True, help="Identifier of the document to delete.")
@click.option("--page", type=int, default=None, help="Page number to delete. If omitted, the entire document is deleted.")
@click.option("--soft", is_flag=True, help="Perform a soft deletion by setting the effective_to field instead of hard deleting.")
def main(doc_id: str, page: int, soft: bool) -> None:
    """
    Delete or soft delete a document or a single page.
    """
    config = load_config()
    client = get_client(config)
    if page is not None:
        delete_document_page(client, config, doc_id, page, soft=soft)
        click.echo(f"Deleted page {page} of document {doc_id} (soft={soft})")
    else:
        delete_document(client, config, doc_id, soft=soft)
        click.echo(f"Deleted document {doc_id} (soft={soft})")


if __name__ == "__main__":
    main()