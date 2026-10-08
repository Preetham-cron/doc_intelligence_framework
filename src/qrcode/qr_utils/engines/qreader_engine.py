"""
engines/qreader_engine.py
--------------------------
QR detection using QReader (deep-learning, slowest but most accurate).
Returns the first QR string found, or None.
"""

from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

_INSTANCE = None
_AVAILABLE = False

try:
    import numpy as np
    from qreader import QReader as _QReaderCls

    _INSTANCE = _QReaderCls(model_size="l", min_confidence=0.3)
    _AVAILABLE = True
except Exception:
    log.warning(
        "QReader not installed or failed to initialise — qreader engine disabled. " "Run: pip install qreader"
    )


def detect(pil_img: Image.Image) -> str | None:
    """
    Run QReader QR detection on the given PIL image.

    Args:
        pil_img: A PIL Image (any mode).

    Returns:
        Decoded QR string if found, else None.
    """
    if not _AVAILABLE or _INSTANCE is None:
        return None

    try:
        img_arr = np.array(pil_img.convert("RGB"))
        results = _INSTANCE.detect_and_decode(image=img_arr)
        for result in results:
            if result:
                log.debug("qreader: QR detected (%d chars)", len(result))
                return result
    except Exception as exc:
        log.warning("qreader: scan error — %s", exc)

    return None
