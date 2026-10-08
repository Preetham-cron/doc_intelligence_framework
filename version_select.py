"""
version_select.py  —  pick the right config VERSION for one invoice, then extract.

Why: a vendor file can hold several blocks, e.g.
    29AAECT1871F1Z8:TAXINVOICE:V1   (layout_hash=d3f985d303)
    29AAECT1871F1Z8:CREDITNOTE:V1   (layout_hash=a53d2abedd)
    29AAECT1871F1Z8:TAXINVOICE:V2   (layout_hash=dcbdce553b)
header_extraction.load_config() merges ALL blocks' "header" dicts into one, so the
last block (V2) overwrites V1 and the credit-note regexes. Invoices of the V1 layout
are then extracted with V2's regexes -> wrong values even though each block is right
for its own layout.

This module selects ONE block per invoice and runs your unchanged extract_headers
on just that block:
  1. doc type from the text (CREDIT NOTE / TAX INVOICE / GST INVOICE), same rule as the agent
  2. layout_hash of the text, same as the agent; the block whose prompt_note says
     layout_hash=<that hash> wins
  3. no hash match (e.g. a new layout) -> try every block (same doc type first) and
     keep the one that fills the most fields

Usage (run_test.py or production):
    from version_select import extract_headers_versioned
    out = extract_headers_versioned(H, text, config_path)["header"]
"""
import hashlib
import json
import os
import re
import tempfile

# --- copies of src/fingerprint.layout_hash and graph._doc_type (keep in sync) ---
_VOCAB = [
    "tax invoice", "gst invoice", "credit note", "under section 31", "taxable value",
    "assessable", "cgst", "sgst", "utgst", "igst", "cess", "tcs", "hsn", "sac",
    "description of goods", "part no", "customer part", "part code", "place of supply",
    "reverse charge", "e-way bill", "irn", "total invoice value", "in figures", "in words",
    "detail of receiver", "detail of consignee", "bill to", "ship to", "nagare", "asn",
    "pds no", "po.no", "ack no", "ack date", "vehicle number", "transportation mode",
    "gross total", "total sales amount", "amortization", "rate/unit", "order qty",
    "invoice no", "invoice date", "gstin", "uom", "trolleys", "boxes", "mode of transport",
]
_TAX_WORDS = {"cgst", "sgst", "utgst", "igst", "cess", "tcs"}


def layout_hash(raw_text):
    low = (raw_text or "").lower()
    present = tuple(sorted(v for v in _VOCAB if v not in _TAX_WORDS and v in low))
    return hashlib.sha1("|".join(present).encode("utf-8")).hexdigest()[:10]


def doc_type(raw):
    u = (raw or "").upper()
    lines = [l.strip() for l in u.splitlines()]
    if "CREDIT NOTE NO" in u or any(l == "CREDIT NOTE" for l in lines):
        return "CREDITNOTE"
    if "TAX INVOICE" in u:
        return "TAXINVOICE"
    if "GST INVOICE" in u:
        return "GSTINVOICE"
    return "INVOICE"
# ---------------------------------------------------------------------------------

_HASH_RE = re.compile(r"layout_hash=([0-9a-f]{6,})")
_TMP = {}            # (config_path, mtime, key) -> temp single-block config path


def _blocks(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "fields" in cfg:                       # legacy flat config: nothing to select
        return None
    return {k: v for k, v in cfg.items() if isinstance(v, dict) and "header" in v}


def _block_hash(block):
    for note in block.get("prompt_note") or []:
        m = _HASH_RE.search(str(note))
        if m:
            return m.group(1)
    return None


def _single_path(config_path, key, block):
    tag = (config_path, os.path.getmtime(config_path), key)
    p = _TMP.get(tag)
    if p and os.path.exists(p):
        return p
    fd, p = tempfile.mkstemp(prefix="cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({key: block}, f, ensure_ascii=False)
    _TMP[tag] = p
    return p


def _filled(header):
    return sum(1 for v in header.values() if v not in ("", 0.0, None))


def select_version(text, config_path):
    """Return (key, how) for the block to use, or (None, 'flat') for a legacy config."""
    blocks = _blocks(config_path)
    if not blocks:
        return None, "flat"
    if len(blocks) == 1:
        return next(iter(blocks)), "only block"
    dt, lh = doc_type(text), layout_hash(text)
    for k, b in blocks.items():
        if _block_hash(b) == lh and k.split(":")[1:2] == [dt]:
            return k, f"layout_hash {lh}"
    for k, b in blocks.items():
        if _block_hash(b) == lh:
            return k, f"layout_hash {lh} (doc type differs)"
    return None, f"no block for layout_hash {lh} ({dt})"


def extract_headers_versioned(H, text, config_path, output_keys=None, verbose=False):
    """Drop-in for H.extract_headers(text, config_path=...). Same return shape,
    plus "_version" = the block key used (pop it if you don't want it)."""
    blocks = _blocks(config_path)
    if not blocks:
        return H.extract_headers(text, config_path=config_path, output_keys=output_keys)
    key, how = select_version(text, config_path)
    if key is not None:
        out = H.extract_headers(text, config_path=_single_path(config_path, key, blocks[key]),
                                output_keys=output_keys)
    else:
        # unknown layout: try same-doc-type blocks first, then the rest; best fill wins
        dt = doc_type(text)
        order = sorted(blocks, key=lambda k: (k.split(":")[1:2] != [dt], k))
        best = None
        for k in order:
            o = H.extract_headers(text, config_path=_single_path(config_path, k, blocks[k]),
                                  output_keys=output_keys)
            if best is None or _filled(o["header"]) > _filled(best[1]["header"]):
                best = (k, o)
        key, out = best
        how += " -> best fill"
    if verbose:
        print(f"   [version] {key}  ({how})")
    out = dict(out)
    out["_version"] = key
    return out
