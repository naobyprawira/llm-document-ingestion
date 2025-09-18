from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

"""
DocParser module.

This module wraps Docling's PDF parsing functionality behind a simple
interface that extracts a `DoclingDocument`, Markdown, and any
embedded figures. In environments where the Docling library is not
installed, the module imports fail. To allow the rest of the API to
load (for example when testing image parsing only), the imports are
performed conditionally. A global flag `_HAS_DOCLING` indicates
whether the necessary classes are available. When Docling is
unavailable, `DocParser.parse` will raise an `ImportError` when
invoked.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
    from docling_core.types.doc import DoclingDocument, PictureItem  # type: ignore
    from docling_core.types.doc.base import ImageRefMode  # type: ignore
    _HAS_DOCLING = True
except Exception:
    # If docling is not installed, set flag and import stubs for type hints
    DocumentConverter = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    DoclingDocument = None  # type: ignore
    PictureItem = None  # type: ignore
    ImageRefMode = None  # type: ignore
    _HAS_DOCLING = False

from PIL import Image  # type: ignore
from .util_img import to_jpeg_bytes

from PIL import Image  # type: ignore
from .util_img import to_jpeg_bytes


@dataclass
class Figure:
    index: int
    page: Optional[int]
    caption: str
    jpeg_bytes: bytes


class DocParser:
    """Wrapper around Docling's DocumentConverter with sane defaults.

    If Docling is not installed, instantiating this class still
    succeeds, but calling :meth:`parse` will raise an
    ``ImportError``. This allows the API module to import without
    triggering a ``ModuleNotFoundError``, so that other endpoints
    (like image parsing) remain functional when Docling is absent.
    """

    def __init__(self, images_scale: float = 1.4, keep_page_images: bool = False) -> None:
        if not _HAS_DOCLING:
            # Store options for potential future use; no converter is created
            self.converter = None  # type: ignore
            self.images_scale = images_scale
            self.keep_page_images = keep_page_images
            return
        # Configure Docling pipeline options
        opts = PdfPipelineOptions()
        opts.generate_picture_images = True
        opts.images_scale = images_scale
        opts.generate_page_images = bool(keep_page_images)
        # Disable optional behaviour if available on this version
        if hasattr(opts, "do_picture_description"):
            opts.do_picture_description = False
        if hasattr(opts, "enable_remote_services"):
            opts.enable_remote_services = False
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        })

    def parse(self, pdf_path: str) -> Tuple["DoclingDocument", str, List[Figure]]:
        """Convert a PDF to a DoclingDocument and extract figures as JPEG bytes.

        :param pdf_path: Path to the PDF file on disk.
        :return: A tuple of (DoclingDocument, base markdown, list of figures).
        :raises ImportError: If Docling is not installed.
        """
        if not _HAS_DOCLING or self.converter is None:
            raise ImportError(
                "DocParser.parse requires the docling library to be installed."
            )
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
        return doc, base_md, figures