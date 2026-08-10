"""Image conversion helpers."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def to_jpeg_bytes(image: Image.Image, quality: int = 85) -> BytesIO:
    """Encode a PIL image as an in-memory JPEG, rewound and ready to read.

    JPEG has no grayscale-with-alpha mode, so anything unusual is normalised to
    RGB first.
    """
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer
