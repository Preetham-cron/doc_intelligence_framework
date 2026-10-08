"""
engines/zxing_engine.py
-----------------------
QR detection using zxing-cpp.
Returns a list of QR strings found, or None.
"""

from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

try:
    import numpy as np
    import zxingcpp

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    log.warning("zxing-cpp not installed — zxing engine disabled. Run: pip install zxing-cpp")


def detect(pil_img: Image.Image) -> list[str] | None:
    """
    Run ZXing-cpp QR detection on the given PIL image.

    Args:
        pil_img: A PIL Image (any mode).

    Returns:
        List of decoded QR strings if found, else None.
    """
    if not _AVAILABLE:
        return None

    try:
        img_rgb = np.array(pil_img.convert("RGB"))
        results = zxingcpp.read_barcodes(img_rgb)
        texts = [
            result.text
            for result in results
            if result.format == zxingcpp.BarcodeFormat.QRCode and result.text
        ]
        if texts:
            log.debug("zxing: %d QR(s) detected", len(texts))
            return texts
    except Exception as exc:
        log.warning("zxing: scan error — %s", exc)

    return None
