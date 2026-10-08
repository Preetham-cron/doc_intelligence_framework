"""
utilities/qr_utils.py
---------------------
Helper utilities for parsing raw QR strings into structured invoice dicts.

Two formats are handled:
  1. Plain JSON  — the QR string is a JSON object directly.
  2. JWT         — 3-segment dot-delimited token (Indian e-Invoice format).

Public API
  parse_qr_text(raw: str)          -> dict | None
  extract_invoice_fields(qr_data)  -> dict
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

_EXPECTED_KEYS = {"SellerGstin", "BuyerGstin", "DocNo", "DocTyp", "DocDt", "TotInvVal", "Irn"}


def _try_plain_json(raw: str) -> dict | None:
    """Attempt to parse raw as a plain JSON object."""
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            log.debug("qr_utils: parsed as plain JSON (%d keys)", len(data))
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_jwt(raw: str) -> dict | None:
    """Attempt to decode raw as an Indian e-Invoice JWT."""
    from src.qrcode.qr_utils.jwt_decoder import decode as jwt_decode

    result = jwt_decode(raw)
    if result:
        log.debug("qr_utils: parsed as JWT (%d fields)", len(result))
    return result


def parse_qr_text(raw: str | None) -> dict | None:
    """
    Parse a raw QR string into a structured invoice dictionary.

    Tries plain-JSON first (fast path), then falls back to JWT decoding
    (standard Indian e-Invoice format).

    Args:
        raw: The decoded QR string from QRCodeExtractor.

    Returns:
        A dict containing invoice fields, or None if unparseable.
    """
    if not raw:
        return None

    data = _try_plain_json(raw)
    if data is None:
        data = _try_jwt(raw)

    if data is None:
        log.debug("qr_utils: could not parse QR string.")
        return None

    found = _EXPECTED_KEYS & set(data.keys())
    if found:
        log.debug("qr_utils: recognised invoice keys — %s", sorted(found))

    return data


def extract_invoice_fields(qr_data: dict) -> dict:
    """
    Extract canonical invoice fields from a parsed QR data dict.

    Maps QR keys → pipeline field names expected by invoice_extraction_agent.

    Args:
        qr_data: Parsed dict from parse_qr_text().

    Returns:
        Dict with keys: seller_gstin, buyer_gstin, doc_no, doc_typ,
        doc_dt, tot_inv_val, irn.  Missing keys default to None.
    """
    return {
        "seller_gstin": qr_data.get("SellerGstin"),
        "buyer_gstin": qr_data.get("BuyerGstin"),
        "doc_no": qr_data.get("DocNo"),
        "doc_typ": qr_data.get("DocTyp"),
        "doc_dt": qr_data.get("DocDt"),
        "tot_inv_val": qr_data.get("TotInvVal"),
        "irn": qr_data.get("Irn"),
    }
