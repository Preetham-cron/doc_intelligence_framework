"""
utilities/pdf_utils.py
----------------------
PDF-to-image conversion utilities for the OCR pipeline.

Ported from QrDecoder/utils/pdf_utils.py with an additional
`blob_to_images()` function that works on raw PDF bytes (from DB blobs)
instead of file paths.

Both functions return a list of PIL Images (one per requested page),
or None / empty list on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_DPI = 400

try:
    from pdf2image import convert_from_bytes, convert_from_path, pdfinfo_from_bytes
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    )

    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False
    log.error(
        "pdf2image not found — QR detection will be disabled. "
        "Run: pip install pdf2image  (also requires Poppler on PATH)"
    )


# ---------------------------------------------------------------------------
# Blob (bytes) → PIL Image  — used by QRCodeExtractor in the OCR pipeline
# ---------------------------------------------------------------------------


def blob_to_images(
    blob_data: bytes,
    page_idx: int,
    dpi: int = DEFAULT_DPI,
) -> list[Image.Image]:
    """
    Convert a single page of a PDF blob (bytes) to a PIL Image.

    Used by QRCodeExtractor when the PDF comes from the database as bytes.

    Args:
        blob_data : Raw PDF bytes from the database.
        page_idx  : 1-based page number to rasterise.
        dpi       : Rasterisation resolution (400 recommended for QR detection).

    Returns:
        List with one PIL Image on success, empty list on any failure.
    """
    if not _PDF2IMAGE_AVAILABLE:
        log.error("pdf2image not installed — cannot rasterise PDF blob.")
        return []

    try:
        pages: list[Image.Image] = convert_from_bytes(
            blob_data,
            dpi=dpi,
            fmt="jpeg",
            first_page=page_idx,
            last_page=page_idx,
            thread_count=1,
        )
        log.debug("blob_to_images: rasterised page %d @ %d DPI (%d image(s))", page_idx, dpi, len(pages))
        return pages

    except PDFInfoNotInstalledError:
        log.error("blob_to_images: Poppler not installed or not on PATH.")
    except (PDFPageCountError, PDFSyntaxError) as exc:
        log.error("blob_to_images: corrupted or unreadable PDF — %s", exc)
    except Exception as exc:
        log.error("blob_to_images: unexpected error — %s: %s", type(exc).__name__, exc)

    return []


def get_blob_page_count(blob_data: bytes) -> int | None:
    """
    Cheaply probe the total page count of a PDF blob.

    Args:
        blob_data: Raw PDF bytes.

    Returns:
        Page count (int) on success, None if detection fails.
    """
    if not _PDF2IMAGE_AVAILABLE:
        return None

    try:
        info = pdfinfo_from_bytes(blob_data)
        return info.get("Pages")
    except Exception as exc:
        log.debug("get_blob_page_count: failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Path → PIL Image  — original function from QrDecoder, kept for completeness
# ---------------------------------------------------------------------------


def pdf_to_images(
    pdf_path: Path,
    page_idx: int,
    dpi: int = DEFAULT_DPI,
) -> list[Image.Image] | None:
    """
    Convert a single page of a PDF file to a PIL Image.

    Args:
        pdf_path  : Path to the PDF file on disk.
        page_idx  : 1-based page number to rasterise.
        dpi       : Rasterisation resolution (400 recommended for QR detection).

    Returns:
        List of PIL Images (one per page), or None if conversion fails.
    """
    if not _PDF2IMAGE_AVAILABLE:
        log.error("pdf2image not installed — cannot rasterise PDF file.")
        return None

    try:
        pages: list[Image.Image] = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="jpeg",
            first_page=page_idx,
            last_page=page_idx,
            thread_count=1,
        )
        log.info("pdf_to_images: rasterised %d page(s) from '%s' @ %d DPI", len(pages), pdf_path.name, dpi)
        return pages

    except PDFInfoNotInstalledError:
        log.error("'%s': Poppler not installed or not on PATH.", pdf_path.name)
    except (PDFPageCountError, PDFSyntaxError) as exc:
        log.error("'%s': corrupted or unreadable PDF — %s", pdf_path.name, exc)
    except Exception as exc:
        log.error("'%s': unexpected error — %s: %s", pdf_path.name, type(exc).__name__, exc)

    return None
