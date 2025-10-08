from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument, PictureItem
from docling_core.types.doc.base import ImageRefMode

from PIL import Image
from .util_img import to_jpeg_bytes


@dataclass
class Figure:
    index: int
    page: Optional[int]
    caption: str
    jpeg_bytes: bytes


class DocParser:
    def __init__(self, images_scale: float = 1.4, keep_page_images: bool = False):
        opts = PdfPipelineOptions()
        opts.generate_picture_images = True
        opts.images_scale = images_scale
        opts.generate_page_images = bool(keep_page_images)
        if hasattr(opts, "do_picture_description"):
            opts.do_picture_description = False
        if hasattr(opts, "enable_remote_services"):
            opts.enable_remote_services = False
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        })

    def parse(self, pdf_path: str) -> Tuple[str, List[Figure]]:
        res = self.converter.convert(pdf_path)
        doc: DoclingDocument = res.document
        # Base markdown: placeholder mode so we don't embed images
        base_md = doc.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER, include_annotations=True
        )

        figures: List[Figure] = []
        idx = 1
        for item, _lvl in doc.iterate_items():
            if isinstance(item, PictureItem):
                try:
                    img = item.get_image(doc)
                except Exception:
                    img = None
                if img is None:
                    continue
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                page = getattr(item, "page_no", None)
                # Fallback to provenance
                if not page:
                    try:
                        if getattr(item, "prov", None):
                            page = getattr(item.prov[0], "page_no", None)
                    except Exception:
                        page = None
                jpeg = to_jpeg_bytes(img)
                figures.append(Figure(index=idx, page=page, caption=caption, jpeg_bytes=jpeg))
                idx += 1
        return base_md, figures
