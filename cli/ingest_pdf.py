"""Command line interface for ingesting PDF documents.

Wraps :func:`app.pipelines.ingest_pdf.ingest_document`.

Examples (Windows CMD):
  python -m cli.ingest_pdf ^
    --path "D:\docs\policy.pdf" ^
    --scope GovReg --company PRB --country ID ^
    --doc-id TEST-001 ^
    --dry-run --start-page 1 --max-pages 2 --force-vlm
"""
from __future__ import annotations

import click

from app.config import load_config
from app.pipelines.ingest_pdf import ingest_document


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--path", required=True, type=click.Path(exists=True, dir_okay=False), help="Input PDF file.")
@click.option("--doc-id", required=True, help="Stable document identifier.")
@click.option("--scope", required=True, help="Business scope or category.")
@click.option("--company", required=True, help="Company code.")
@click.option("--country", required=True, help="Country code, e.g., ID.")
@click.option("--language", default=None, help="BCP-47 language tag, e.g., id.")
@click.option("--version", default="v1", show_default=True, help="Document version string.")
@click.option("--effective-from", default="1970-01-01", show_default=True, help="ISO date when the doc becomes effective.")
@click.option("--effective-to", default=None, help="Optional ISO date when the doc expires.")
@click.option("--security-class", default="internal", show_default=True, help="Security classification label.")
@click.option("--start-page", "--resume-from", type=int, default=1, show_default=True, help="First page to process (1-based).")
@click.option("--max-pages", type=int, default=0, show_default=True, help="Max pages to process starting from --start-page. 0 = all.")
@click.option("--dry-run", is_flag=True, help="Parse and chunk only. Do not write to Qdrant.")
@click.option("--force-vlm", is_flag=True, help="Force VLM flowchart extraction on processed pages.")
def main(
    path: str,
    doc_id: str,
    scope: str,
    company: str,
    country: str,
    language: str | None,
    version: str,
    effective_from: str,
    effective_to: str | None,
    security_class: str,
    start_page: int,
    max_pages: int,
    dry_run: bool,
    force_vlm: bool,
) -> None:
    cfg = load_config()

    result = ingest_document(
        filepath=path,
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
        force_vlm=force_vlm,
        config=cfg,
    )

    print(
        f"Ingested document {doc_id} with "
        f"{result.get('n_flowcharts', 0)} flowchart(s) and "
        f"{result.get('n_text_chunks', 0)} text chunk(s)"
    )


if __name__ == "__main__":
    main()
