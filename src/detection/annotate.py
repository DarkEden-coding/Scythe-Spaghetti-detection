"""Draw detection boxes onto a frame.

Pure PIL — no OpenCV, and no colour-space juggling. The old implementation ran
``cv2.cvtColor(..., COLOR_RGB2BGR)`` over an array that had already been
converted to single-channel grayscale, which is not a conversion that means
anything.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
BOX_WIDTH = 3


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1 has no size parameter
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes, *, label: bool = True) -> Image.Image:
    """Return a copy of ``image`` with ``boxes`` outlined.

    The input is never mutated, so the caller can still send the clean frame.
    """
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for box in boxes:
        draw.rectangle(box.as_tuple(), outline=BOX_COLOR, width=BOX_WIDTH)
        if not label:
            continue

        text = f"{box.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top

        # Keep the label inside the frame when the box hugs the top edge.
        label_y = box.y1 - text_h - 4
        if label_y < 0:
            label_y = min(box.y1 + 2, canvas.height - text_h - 4)

        draw.rectangle(
            (box.x1, label_y, box.x1 + text_w + 6, label_y + text_h + 4),
            fill=BOX_COLOR,
        )
        draw.text((box.x1 + 3, label_y + 2), text, fill=TEXT_COLOR, font=font)

    return canvas
