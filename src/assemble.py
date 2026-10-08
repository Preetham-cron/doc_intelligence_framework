"""
assemble.py  —  collect Gemini's FULL field blocks into your config shape:
    { "<GSTIN>:<DOCTYPE>:<VERSION>": { "prompt_note":[...], "header": { field: {...} } } }

Gemini now generates the WHOLE block for each field (alias, direction, datatype,
scope, priority, required, validation, regex, prompt, fallback_alias, negative_alias).
Code does NOT build these — it only VALIDATES them against the schema below: each
value Gemini emitted is kept if valid, and corrected to the canonical value if wrong
or missing (so a bad datatype/validation can never break the pipeline). The
invoice-specific fields (regex/alias/direction/negative_alias) are Gemini's and pass
through unchanged (the regex has already been extraction-tested upstream).
"""
import json

# Canonical schema (validation reference). Gemini emits these; code checks/corrects.
FIELD_TEMPLATE = {
    "gstin_number": {
        "datatype": "gstin", "scope": "vendor", "priority": 10, "required": True,
        "validation": ["gstin", "gstin_checksum"],
        "prompt": ["Find the GSTIN associated with the supplier's address at the top."],
        "default_negative_alias": ["Ship-From", "Bill To", "Ship To"],
    },
    "supplier_document_number": {
        "datatype": "string", "scope": "invoice", "priority": 10, "required": True,
        "validation": ["non_empty"],
        "prompt": ["Find the supplier invoice number near the top supplier details."],
    },
    "supplier_document_date": {
        "datatype": "date", "scope": "invoice", "priority": 10, "required": True,
        "validation": ["date"],
        "prompt": ["Look for the invoice date near the top / IRN number."],
    },
    "total_invoice_value": {
        "datatype": "amount", "scope": "invoice", "priority": 10, "required": True,
        "validation": ["numeric"],
        "prompt": ["Find 'Total Invoice Value (in figures)' and capture the number."],
    },
    "total_invoice_taxable_value": {
        "datatype": "amount", "scope": "invoice", "priority": 8, "required": False,
        "validation": ["numeric"], "prompt": ["Taxable value total in the summary row."]},
    "total_invoice_cgst_tax_value": {
        "datatype": "amount", "scope": "invoice", "priority": 9, "required": False,
        "validation": ["numeric"], "prompt": ["CGST amount total in the summary row."]},
    "total_invoice_sgst_tax_value": {
        "datatype": "amount", "scope": "invoice", "priority": 9, "required": False,
        "validation": ["numeric"], "prompt": ["SGST amount total in the summary row."]},
    "total_invoice_igst_tax_value": {
        "datatype": "amount", "scope": "invoice", "priority": 9, "required": False,
        "validation": ["numeric"], "prompt": ["IGST amount total in the summary row."]},
    "total_invoice_cess_tax_value": {
        "datatype": "amount", "scope": "invoice", "priority": 8, "required": False,
        "validation": ["numeric"], "prompt": ["CESS/TCS amount total in the summary row."]},
    "invoice_total_basic_value": {
        "datatype": "amount", "scope": "invoice", "priority": 8, "required": False,
        "validation": ["numeric"], "prompt": ["Total sales amount in the summary row."]},
    "document_type": {
        "datatype": "string", "scope": "invoice", "priority": 10, "required": True,
        "validation": ["non_empty"], "prompt": ["Look for the document type at the top."],
    },
    "irn_number": {
        "datatype": "string", "scope": "invoice", "priority": 10, "required": False,
        "validation": ["non_empty"], "prompt": ["Find 'IRN NO.:' near the top."],
    },
    "po_pds_number": {
        "datatype": "string", "scope": "invoice", "priority": 5, "required": False,
        "validation": [], "prompt": []},
    "order_date": {
        "datatype": "date", "scope": "invoice", "priority": 5, "required": False,
        "validation": [], "prompt": []},
    "vendor_code": {
        "datatype": "string", "scope": "invoice", "priority": 5, "required": False,
        "validation": [], "prompt": []},
}

CANONICAL_ORDER = [
    "document_type", "gstin_number", "po_pds_number", "order_date", "vendor_code",
    "supplier_document_number", "supplier_document_date", "invoice_total_basic_value",
    "total_invoice_taxable_value", "total_invoice_igst_tax_value",
    "total_invoice_cgst_tax_value", "total_invoice_sgst_tax_value",
    "total_invoice_cess_tax_value", "total_invoice_value", "irn_number",
]

_DATATYPES = {"amount", "gstin", "date", "percentage", "string"}
_DIRECTIONS = {"right", "below", "bottom", "above", "left"}


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _field_block(name, spec):
    """Validate Gemini's block against the schema: keep valid values it emitted,
    correct invalid/missing ones to canonical. Regex/alias/direction pass through."""
    tpl = FIELD_TEMPLATE.get(name, {})
    spec = spec or {}

    # --- fixed schema fields: keep Gemini's value only if it matches canonical ---
    datatype = spec.get("datatype")
    if datatype not in _DATATYPES:
        datatype = tpl.get("datatype", "string")
    scope = spec.get("scope") if spec.get("scope") in ("vendor", "invoice") else tpl.get("scope", "invoice")
    priority = spec.get("priority")
    if not (isinstance(priority, int) and 1 <= priority <= 10):
        priority = tpl.get("priority", 5)
    required = spec.get("required")
    if not isinstance(required, bool):
        required = tpl.get("required", False)
    validation = spec.get("validation")
    if not isinstance(validation, list):
        validation = tpl.get("validation", [])
    prompt = spec.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        prompt = tpl.get("prompt", [])

    # --- invoice-specific fields: Gemini's, pass through (regex already tested) ---
    direction = spec.get("direction")
    if direction not in _DIRECTIONS:
        direction = ("bottom" if name.startswith("total_invoice_") and name != "total_invoice_value" else "right")
    negative_alias = _as_list(spec.get("negative_alias")) or tpl.get("default_negative_alias", [])

    block = {
        "alias": _as_list(spec.get("alias")),
        "direction": direction,
        "datatype": datatype,
        "scope": scope,
        "priority": priority,
        "required": required,
        "validation": validation,
        "regex": _as_list(spec.get("regex")),
        "prompt": prompt,
        "fallback_alias": _as_list(spec.get("fallback_alias")),
        "negative_alias": negative_alias,
    }
    if not block["regex"] and not block["alias"]:
        block["fallback_value"] = "" if block["datatype"] == "string" else 0.0
    return block


def assemble_config(passing_specs, *, gstin, doc_type="GSTINVOICE", version="V1",
                    prompt_note=None, include_unfilled=True):
    header = {}
    order = CANONICAL_ORDER if include_unfilled else list(passing_specs.keys())
    for name in order:
        if name in passing_specs:
            header[name] = _field_block(name, passing_specs[name])
        elif include_unfilled:
            header[name] = _field_block(name, None)      # empty stub, pipeline-safe
    key = f"{gstin}:{doc_type}:{version}"
    return {key: {"prompt_note": prompt_note or [], "header": header}}
