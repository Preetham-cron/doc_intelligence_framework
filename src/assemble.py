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
from src.field_schema import CANONICAL_ORDER, validate_and_correct  # noqa: F401 (CANONICAL_ORDER
# is re-exported here unchanged, since graph.py does `from src.assemble import CANONICAL_ORDER`)

_DIRECTIONS = {"right", "below", "bottom", "above", "left"}


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _field_block(name, spec):
    """Validate Gemini's block against the schema: keep valid values it emitted,
    correct invalid/missing ones to canonical. Regex/alias/direction pass through."""
    spec = spec or {}

    # --- fixed schema fields: keep Gemini's value only if it matches canonical ---
    # (the actual validation/correction logic — and the canonical values themselves —
    # now live in src/field_schema.py, the single source of truth shared with
    # discover.py's prompt text and generate.py's prompt table)
    fixed = validate_and_correct(name, spec)

    # --- invoice-specific fields: Gemini's, pass through (regex already tested) ---
    direction = spec.get("direction")
    if direction not in _DIRECTIONS:
        direction = ("bottom" if name.startswith("total_invoice_") and name != "total_invoice_value" else "right")
    negative_alias = _as_list(spec.get("negative_alias")) or fixed["default_negative_alias"]

    block = {
        "alias": _as_list(spec.get("alias")),
        "direction": direction,
        "datatype": fixed["datatype"],
        "scope": fixed["scope"],
        "priority": fixed["priority"],
        "required": fixed["required"],
        "validation": fixed["validation"],
        "regex": _as_list(spec.get("regex")),
        "prompt": fixed["prompt"],
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