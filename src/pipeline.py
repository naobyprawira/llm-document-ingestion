"""
High-level ingestion pipeline.

The ``run`` function orchestrates the full flow from raw PDF input
through parsing, image description, markdown assembly, chunking and
embedding into Qdrant. It mirrors the upstream implementation and
provides a simple entry point for CLI and web usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import List

from .doc_parser import DocParser, Figure
from .vlm import GeminiVLM, VLMConfig
from .prompts import default_prompt
from .markdown import FigureDesc, compose_markdown
from .chunk import chunk_markdown
from .embed import embed_and_upload_json


@dataclass
class Result:
    markdown: str


def run(pdf_path: str) -> Result:
    """End-to-end ingestion of a PDF file.

    The steps are:

    1. Parse the PDF with Docling, obtaining the base markdown and
       extracted figures.
    2. Describe each figure using ``GeminiVLM`` and the default prompt.
    3. Compose the enriched markdown by inserting descriptions.
    4. Chunk the markdown and upload embeddings to Qdrant.
    """
    parser = DocParser(images_scale=1.4, keep_page_images=False)
    base_md, figures = parser.parse(pdf_path)

    # 2) describe images with VLM (respecting max concurrent in config)
    vlm = GeminiVLM(VLMConfig())
    descs: List[FigureDesc] = []
    for f in figures:
        try:
            text = vlm.describe(
                image_bytes=f.jpeg_bytes,
                prompt=default_prompt()
            )
        except Exception:
            # simple retry loop
            max_retries = 2
            for _ in range(max_retries):
                try:
                    text = vlm.describe(
                        image_bytes=f.jpeg_bytes,
                        prompt=default_prompt()
                    )
                    break
                except Exception:
                    text = ""
            if not text:
                print(
                    f"Warning: Failed to generate description for figure {f.index} "
                    f"on page {f.page} after {max_retries} retries"
                )
        if text.startswith("Small image (likely logo or decorative element)"):
            print(f"Info: Skipped small image for figure {f.index} on page {f.page}")
            continue
        descs.append(FigureDesc(index=f.index, page=f.page, caption=f.caption, description=text))
    # 3) compose
    md = compose_markdown(base_md, descs)
    # 4) chunk and embed
    chunks = chunk_markdown(md)
    embed_and_upload_json(
        chunks,
        filename=Path(pdf_path).stem,
        category=os.getenv("DOC_CATEGORY", "Internal"),
    )
    print(f"Process finished: {len(chunks)} chunks embedded")
    return Result(markdown=md)
