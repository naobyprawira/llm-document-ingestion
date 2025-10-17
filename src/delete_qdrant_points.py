"""CLI utility to delete Qdrant points by filename."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from .qdrant_utils import delete_points_by_filename


@click.command()
@click.argument("filename")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--category", default=None, help="Optional category filter")
@click.option("--batch", default=256, show_default=True, help="Scroll batch size")
def cli(filename: str, collection: str | None, category: str | None, batch: int) -> None:
    """Delete all Qdrant points for FILENAME."""
    load_dotenv()
    deleted = delete_points_by_filename(
        filename,
        collection=collection,
        category=category,
        batch=batch,
    )
    click.echo(f"Deleted {deleted} points for {filename}")


if __name__ == "__main__":  # pragma: no cover
    cli()
