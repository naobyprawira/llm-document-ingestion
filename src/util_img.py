"""
Utilities for image handling in the document ingestion pipeline.

Docling returns images in various internal formats. This helper
normalises those images into JPEG bytes suitable for sending to the
Google VLM. The images are resized to a maximum side length to
prevent unnecessarily large payloads and converted to RGB if needed.
The resulting JPEG is also written to a temporary directory on disk
for debugging purposes.
"""

from __future__ import annotations

from io import BytesIO
from PIL import Image
import os

# Maximum dimension (in pixels) for the longest side of the image. Set via
# environment variable ``IMG_MAX_SIDE`` or defaults to 1600.
IMG_MAX_SIDE = int(os.getenv("IMG_MAX_SIDE", "1600"))

# JPEG quality for saved images. Lower values produce smaller files at the
# expense of quality. Configurable via ``IMG_JPEG_QUALITY`` env var.
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "85"))


def to_jpeg_bytes(img) -> bytes:
    """Convert a Docling image wrapper or PIL image into JPEG bytes.

    The function handles a variety of object types returned by Docling,
    including objects with ``to_pil``, ``pil`` or ``pil_image``
    attributes. It normalises transparency by compositing onto a white
    background and resizes the image if its longest side exceeds
    ``IMG_MAX_SIDE``. The resulting JPEG is saved in a ``temp`` folder
    adjacent to the package for debugging, and the raw bytes are
    returned.

    :param img: A PIL.Image or Docling image wrapper.
    :return: JPEG bytes representing the normalised image.
    :raises TypeError: If ``img`` cannot be converted to a PIL.Image.
    """
    # Normalise to a PIL.Image instance
    if hasattr(img, "to_pil"):
        img = img.to_pil()
    elif hasattr(img, "pil") and img.pil is not None:
        img = img.pil
    elif hasattr(img, "pil_image") and img.pil_image is not None:
        img = img.pil_image
    if not isinstance(img, Image.Image):
        raise TypeError("Expected PIL.Image or Docling image wrapper")

    # Convert transparent images to RGB by compositing over white
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    # Resize if the longest side exceeds IMG_MAX_SIDE
    w, h = img.size
    m = max(w, h)
    if m > IMG_MAX_SIDE:
        scale = IMG_MAX_SIDE / float(m)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, "JPEG", quality=IMG_JPEG_QUALITY, optimize=True)

    # Save a copy to the package's temp folder for debugging/inspection
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"image_{hash(buf.getvalue())}.jpg")
    with open(temp_path, "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()