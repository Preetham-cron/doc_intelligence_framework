"""
src/qrcode/qr_extractor.py
---------------------------
QRCodeExtractor — unified QR detection for the V2 OCR pipeline.

Accepts a raw PDF blob (bytes) from the database, rasterises each page,
and runs a two-engine fallback chain (ZXing → Pyzbar) to find and decode
the e-Invoice QR code entirely in memory (no temp files).

Processing order (per page):
  1. ZXing  (zxing-cpp)  — fast C++ library, tried first
  2. Pyzbar (libzbar)    — reliable fallback, tried ONLY if ZXing finds nothing

Canonical fields returned on success:
    {
        "seller_gstin"  : str   ← used for vendor prompt routing
        "buyer_gstin"   : str   ← Bill-To GSTN
        "irn"           : str   ← Invoice Reference Number
        "doc_no"        : str   ← Supplier Invoice Number
        "doc_dt"        : str   ← Invoice Date
        "tot_inv_val"   : float ← Total Invoice Value
        "doc_typ"       : str
        "_raw_qr_data"  : dict  ← full QR payload for debugging
    }
"""

from __future__ import annotations

import logging

from PIL import Image
from src.qrcode.qr_utils.pdf_utils import blob_to_images, get_blob_page_count
from src.qrcode.qr_utils.qr_utils import _EXPECTED_KEYS, extract_invoice_fields, parse_qr_text

log = logging.getLogger(__name__)

DEFAULT_DPI = 400
MAX_PAGES = 5  # scan at most this many pages when page count is unknown


class QRCodeExtractor:
    """
    Unified QR code extractor for the INVOICE_CHECK_OCR_OPS_V2 pipeline.

    Accepts a PDF blob (bytes), iterates its pages, and applies a
    two-engine fallback chain (ZXing → Pyzbar) on each page image.

    Usage::

        extractor = QRCodeExtractor()
        qr_data = extractor.extract_from_blob(blob_data, invoice_id)
    """

    # ------------------------------------------------------------------ #
    # Engine 1 — ZXing
    # ------------------------------------------------------------------ #

    def extract_zxing(self, pil_img: Image.Image) -> list[str]:
        """
        Attempt QR extraction using ZXing-cpp on a PIL image.

        Returns:
            List of decoded QR strings (empty list if none found).
        """
        from src.qrcode.qr_utils.engines.zxing_engine import detect

        return detect(pil_img)

    # ------------------------------------------------------------------ #
    # Engine 2 — Pyzbar  (fallback — called ONLY when ZXing finds nothing)
    # ------------------------------------------------------------------ #

    def extract_pyzbar(self, pil_img: Image.Image) -> list[str]:
        """
        Attempt QR extraction using Pyzbar on a PIL image.

        Returns:
            List of decoded QR strings (empty list if none found).
        """
        from src.qrcode.qr_utils.engines.pyzbar_engine import detect

        return detect(pil_img)

    # ------------------------------------------------------------------ #
    # Combined engine chain on a single PIL image
    # ------------------------------------------------------------------ #

    def extract(self, pil_img: Image.Image) -> list[str]:
        """
        Run the ZXing → Pyzbar fallback chain on a PIL image.

        Args:
            pil_img: A PIL Image (any colour mode).

        Returns:
            List of raw decoded QR strings (empty list if both engines fail).
        """
        texts = self.extract_zxing(pil_img)
        if texts:
            log.info("QR: %d QR(s) extracted via ZXing.", len(texts))
            return texts

        log.debug("QR: ZXing found nothing — trying Pyzbar.")
        texts = self.extract_pyzbar(pil_img)
        if texts:
            log.info("QR: %d QR(s) extracted via Pyzbar (fallback).", len(texts))
            return texts

        return []

    # ------------------------------------------------------------------ #
    # Main entry point — blob → structured invoice dict
    # ------------------------------------------------------------------ #

    def extract_from_blob(
        self,
        blob_data: bytes,
        invoice_id: int | str = "?",
        dpi: int = DEFAULT_DPI,
    ) -> dict | None:
        """
        Extract QR invoice data from a PDF blob.

        Iterates pages one at a time and stops as soon as a valid QR is
        found and successfully parsed.  Returns None if every page fails.

        Args:
            blob_data  : Raw PDF bytes from the database.
            invoice_id : Invoice ID (used only for log messages).
            dpi        : Rasterisation DPI (default 400).

        Returns:
            Parsed invoice dict with canonical keys, or None.
        """
        page_count = get_blob_page_count(blob_data) or MAX_PAGES
        log.debug("[QR] Invoice ID %s: scanning %d page(s).", invoice_id, page_count)

        for page_idx in range(1, page_count + 1):

            # ── Rasterise page ──────────────────────────────────────────────
            pages = blob_to_images(blob_data, page_idx, dpi)
            if not pages:
                break
            pil_img = pages[0]

            # ── Step 1: ZXing ───────────────────────────────────────────────
            candidates = self.extract_zxing(pil_img)

            # ── Step 2: Pyzbar fallback ─────────────────────────────────────
            if not candidates:
                log.debug(
                    "[QR] ID %s page %d: ZXing found nothing — trying Pyzbar.",
                    invoice_id,
                    page_idx,
                )
                candidates = self.extract_pyzbar(pil_img)

            if not candidates:
                log.debug("[QR] ID %s page %d: both engines found nothing.", invoice_id, page_idx)
                continue

            log.debug(
                "[QR] ID %s page %d: %d QR candidate(s) to evaluate.",
                invoice_id,
                page_idx,
                len(candidates),
            )

            # ── Step 3: Validate candidates against expected keys ───────────
            for raw_text in candidates:
                log.debug(
                    "[QR] ID %s page %d: raw text acquired (%d chars).",
                    invoice_id,
                    page_idx,
                    len(raw_text),
                )

                qr_data = parse_qr_text(raw_text)
                if qr_data is None:
                    log.debug(
                        "[QR] ID %s page %d: could not parse QR text — skipping.",
                        invoice_id,
                        page_idx,
                    )
                    continue

                missing_keys = _EXPECTED_KEYS - set(qr_data.keys())
                if missing_keys:
                    log.debug(
                        "[QR] ID %s page %d: QR missing expected keys %s — skipping.",
                        invoice_id,
                        page_idx,
                        sorted(missing_keys),
                    )
                    continue

                # ── Step 4: Normalise to canonical fields ───────────────────
                invoice = extract_invoice_fields(qr_data)
                invoice["_raw_qr_data"] = qr_data

                log.info(
                    "[QR]   Invoice ID %s: QR extracted on page %d — " "seller=%s, doc=%s, irn=%s",
                    invoice_id,
                    page_idx,
                    invoice.get("seller_gstin", "?"),
                    invoice.get("doc_no", "?"),
                    (invoice.get("irn") or "")[:16] + "…" if invoice.get("irn") else "?",
                )
                return invoice

        log.info(
            "[QR]   Invoice ID %s: no QR found on any page → fallback to OCR flow.",
            invoice_id,
        )
        return None
