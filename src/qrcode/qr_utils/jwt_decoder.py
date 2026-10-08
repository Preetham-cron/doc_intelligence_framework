"""
utilities/jwt_decoder.py
------------------------
Decode Indian e-Invoice QR codes (compact JWT format).

The QR payload is a JWT whose second segment (the payload) contains a
`data` field — a JSON-encoded e-Invoice object.

This module decodes that payload and returns all fields as a flat dict,
or None if the input is not a valid e-Invoice JWT.
"""

from __future__ import annotations

import base64
import json
import logging

log = logging.getLogger(__name__)


def _b64_decode(segment: str) -> bytes:
    """Pad and decode a base64url segment."""
    segment += "=" * (4 - len(segment) % 4)
    return base64.urlsafe_b64decode(segment)


def _flatten_json(data: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Recursively flatten a nested dictionary.

    Nested keys are joined with `sep`. List items are indexed numerically.
    """
    flat: dict = {}
    for key, value in data.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(_flatten_json(value, parent_key=full_key, sep=sep))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                indexed_key = f"{full_key}{sep}{idx}"
                if isinstance(item, dict):
                    flat.update(_flatten_json(item, parent_key=indexed_key, sep=sep))
                else:
                    flat[indexed_key] = item
        else:
            flat[full_key] = value
    return flat


def decode(raw: str) -> dict | None:
    """
    Decode an Indian e-Invoice JWT QR payload.

    JWT structure:  <header>.<payload>.<signature>
    The payload contains a `data` field with the serialised invoice JSON.

    Args:
        raw: Raw QR string (expected to be a 3-part JWT).

    Returns:
        Flattened dict of all invoice fields if decoding succeeds, else None.
    """
    parts = raw.strip().split(".")
    if len(parts) != 3:
        log.debug("jwt_decoder: not a JWT (got %d segments)", len(parts))
        return None

    try:
        outer = json.loads(_b64_decode(parts[1]))
    except Exception as exc:
        log.debug("jwt_decoder: failed to decode JWT payload — %s", exc)
        return None

    raw_data = outer.get("data", "")
    if not raw_data:
        log.debug("jwt_decoder: JWT payload has no 'data' field")
        return None

    try:
        invoice = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except Exception as exc:
        log.debug("jwt_decoder: failed to parse inner invoice JSON — %s", exc)
        return None

    if not isinstance(invoice, dict):
        log.debug("jwt_decoder: invoice data is not a dict")
        return None

    flattened = _flatten_json(invoice)
    log.debug("jwt_decoder: decoded OK — %d fields", len(flattened))
    return flattened
