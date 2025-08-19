"""
Optional OCR support.

This module provides a fallback for scanned PDFs.  When no text can be
extracted from a page via the vector layer, you may call ``ocr_page``
to perform optical character recognition and return the recognised
text.  The default implementation simply returns an empty string; to
enable OCR, install a library such as pytesseract or Docling and
update this function accordingly.
"""
from __future__ import annotations

from typing import Optional


def ocr_page(image_bytes: bytes) -> Optional[str]:
    """
    Perform OCR on an image and return recognised text.

    Parameters
    ----------
    image_bytes: bytes
        PNG image data for a PDF page.

    Returns
    -------
    Optional[str]
        The recognised text, or ``None`` if OCR is not available.
    """
    # OCR is not implemented by default.  Return None to signal that
    # OCR could not be performed.  To enable OCR, install pytesseract
    # and call pytesseract.image_to_string on the image.
    return None
