"""
fingerprint.py  —  layout hash (for V1/V2 versioning) + QR cross-check.

layout_hash(raw_text): a stable signature of the TEMPLATE, not the values.
We strip out numbers/amounts/dates and keep the label-like text, normalized,
so two invoices from the same template hash the same, and a different template
hashes differently. Different hash for a GSTIN => a new version (V1, V2, ...).

qr_crosscheck(present, qr): compares discovered header values with the signed QR
values where they overlap. It NEVER changes what was discovered (raw text is the
source of truth) — it only reports agreement/mismatch for confidence.
"""
import re, hashlib

# Fixed TEMPLATE vocabulary: structural phrases that define a layout. We hash
# only WHICH of these are present, so variable line-item content (part names,
# amounts, dates) is ignored -> same template = same hash, new template = new hash.
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


def layout_hash(raw_text: str) -> str:
    low = (raw_text or "").lower()
    # exclude TAX-TYPE words: the same vendor/template can be IGST on one invoice
    # and CGST/SGST on another (inter- vs intra-state) — that must NOT split the
    # template. The union step handles whichever tax is present per invoice.
    tax_words = {"cgst", "sgst", "utgst", "igst", "cess", "tcs"}
    present = tuple(sorted(v for v in _VOCAB if v not in tax_words and v in low))
    return hashlib.sha1("|".join(present).encode("utf-8")).hexdigest()[:10]


# map discovered header fields <-> QR field names (for cross-check only)
_QR_MAP = {
    "gstin_number":            "seller_gstin",
    "supplier_document_number":"doc_no",
    "supplier_document_date":  "doc_dt",
    "total_invoice_value":     "tot_inv_val",
    "irn_number":              "irn",
}


def _norm(v):
    s = str(v).strip().lower().replace(",", "").replace(" ", "")
    # dates: unify separators so 13.11.2025 == 13/11/2025 == 13-11-2025
    if re.fullmatch(r'\d{2}[./-]\d{2}[./-]\d{4}', s):
        return re.sub(r'[./-]', '', s)
    s = re.sub(r'[-/]', "", s)
    return s


def qr_crosscheck(present: dict, qr: dict) -> dict:
    """Return {field: 'match'|'mismatch'|'no_qr'} — informational only."""
    report = {}
    for field, qr_key in _QR_MAP.items():
        if field not in present:
            continue
        qv = qr.get(qr_key) if qr else None
        if qv in (None, "", []):
            report[field] = "no_qr"
        else:
            same = _norm(present[field]) == _norm(qv)
            if not same and field == "total_invoice_value":
                # amounts: compare as numbers (5,255.50 == 5255.5) and allow the
                # rupee round-off between the printed total and the signed total
                try:
                    a = float(str(present[field]).replace(",", "").strip())
                    b = float(str(qv).replace(",", "").strip())
                    same = abs(a - b) <= 1.0
                except ValueError:
                    pass
            report[field] = "match" if same else "mismatch"
    return report
