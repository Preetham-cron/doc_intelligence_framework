"""
engines/pyzbar_engine.py
------------------------
QR detection using pyzbar.
Returns the first QR string found, or None.
"""

from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False
    log.warning(
        "pyzbar not installed — pyzbar engine disabled. " "Run: apt install libzbar0 && pip install pyzbar"
    )


def detect(pil_img: Image.Image) -> str | None:
    """
    Run pyzbar QR detection on the given PIL image.

    Args:
        pil_img: A PIL Image (any mode).

    Returns:
        Decoded QR string if found, else None.
    """
    if not _AVAILABLE:
        return None

    try:
        for obj in _pyzbar_decode(pil_img):
            if obj.type == "QRCODE":
                try:
                    text = obj.data.decode("utf-8")
                except UnicodeDecodeError:
                    text = obj.data.decode("latin-1", errors="replace")
                if text:
                    log.debug("pyzbar: QR detected (%d chars)", len(text))
                    return text
    except Exception as exc:
        log.warning("pyzbar: scan error — %s", exc)

    return None
