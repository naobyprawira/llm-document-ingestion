"""
CLI entrypoint for the document ingestion pipeline.

Running ``python -m llm_document_ingestion.src.main`` will invoke the
ingestion pipeline on the given PDF and write an enriched Markdown
document alongside the input file. See the ``pipeline`` module for
details on the processing steps.
"""

from __future__ import annotations

import os
import click
from dotenv import load_dotenv

# Load .env before importing modules that read env at import time
load_dotenv()

from .pipeline import run


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output .md path",
)
def cli(pdf_path: str, out: str | None) -> None:
    """Process a PDF and write the enriched Markdown to disk."""
    res = run(pdf_path)
    # Determine output path if not specified
    if not out:
        base, _ = os.path.splitext(pdf_path)
        out = base + "_enriched.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(res.markdown)
    click.echo(f"Saved: {out}")


if __name__ == "__main__":  # pragma: no cover
    cli()