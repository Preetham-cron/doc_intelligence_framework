"""
engines/opencv_engine.py
-------------------------
QR detection using OpenCV (QRCodeDetectorAruco, QRCodeDetector, WeChatQRCode).
Returns the first QR string found, or None.
"""

from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    log.warning("OpenCV not installed — opencv engine disabled. " "Run: pip install opencv-contrib-python")

_DETECTORS: list = []
_WECHAT = None

if _AVAILABLE:
    try:
        _DETECTORS.append(cv2.QRCodeDetectorAruco())
    except AttributeError:
        pass
    _DETECTORS.append(cv2.QRCodeDetector())

    try:
        _WECHAT = cv2.wechat_qrcode_WeChatQRCode()
    except (cv2.error, AttributeError):
        pass


def detect(pil_img: Image.Image) -> str | None:
    """
    Run OpenCV QR detection on the given PIL image.

    Args:
        pil_img: A PIL Image (any mode).

    Returns:
        Decoded QR string if found, else None.
    """
    if not _AVAILABLE:
        return None

    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    for detector in _DETECTORS:
        try:
            ok, texts, _, _ = detector.detectAndDecodeMulti(bgr)
            if ok and texts:
                for text in texts:
                    if text:
                        log.debug("opencv: QR detected (%d chars)", len(text))
                        return text
        except cv2.error:
            pass

    if _WECHAT:
        try:
            texts, _ = _WECHAT.detectAndDecode(bgr)
            for text in texts:
                if text:
                    log.debug("opencv (wechat): QR detected (%d chars)", len(text))
                    return text
        except cv2.error:
            pass

    return None
