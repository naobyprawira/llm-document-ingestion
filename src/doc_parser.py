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
        self.keep_page_images = bool(keep_page_images)
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

    def parse(
        self,
        pdf_path: str,
        *,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[DoclingDocument, str, List[Figure], List[Tuple[int, bytes]]]:
        if page_range:
            res = self.converter.convert(pdf_path, page_range=page_range)
        else:
            res = self.converter.convert(pdf_path)
        doc: DoclingDocument = res.document
        # Base markdown: placeholder mode so we don't embed images
        base_md = doc.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER, include_annotations=True
        )

        figures: List[Figure] = []
        page_images: List[Tuple[int, bytes]] = []
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
        if self.keep_page_images:
            images_attr = getattr(res, "page_images", None)
            if images_attr:
                for page_idx, entry in enumerate(images_attr, start=1):
                    try:
                        if hasattr(entry, "image"):
                            pil_img = entry.image
                            page_no = getattr(entry, "page_no", page_idx)
                        elif isinstance(entry, tuple) and len(entry) == 2:
                            page_no, pil_img = entry
                        else:
                            pil_img = entry
                            page_no = page_idx
                        if not isinstance(pil_img, Image.Image):
                            continue
                        jpeg_bytes = to_jpeg_bytes(pil_img)
                        page_images.append((int(page_no or page_idx), jpeg_bytes))
                    except Exception:
                        continue
        return doc, base_md, figures, page_images
