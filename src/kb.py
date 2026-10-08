"""
kb.py  —  file-based Knowledge Base for solved templates.

Stores, per template fingerprint, the WORKING config so it is reused as-is next
time (no LLM). This is what makes the agent scale to 1000s of vendors: each
distinct template is solved ONCE, then remembered forever.

KB file: knowledge_base.json  (in the project root)
Key    : "<GSTIN>:<DOCTYPE>:<layout_hash>"   (unique per template)
Value  : {
    "config_key" : "<GSTIN>:<DOCTYPE>:V<n>",   # the versioned key used in output
    "gstin", "doc_type", "layout_hash",
    "version"    : "V1",
    "config"     : { ...header... },           # the working config (inner block)
    "fields_filled", "fields_total",
    "created", "updated", "seen_count"
}
"""
import os, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(os.path.dirname(HERE), "knowledge_base.json")   # project root


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_kb() -> dict:
    if os.path.exists(KB_PATH):
        try:
            return json.load(open(KB_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_kb(kb: dict):
    json.dump(kb, open(KB_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _key(gstin, doc_type, layout_hash):
    return f"{gstin}:{doc_type}:{layout_hash}"


def kb_get(kb, gstin, doc_type, layout_hash):
    """Return the stored entry for this template, or None."""
    return kb.get(_key(gstin, doc_type, layout_hash))


def kb_versions_for(kb, gstin, doc_type):
    """Existing {layout_hash: version} for this (gstin, doctype) — for version assignment."""
    out = {}
    for k, v in kb.items():
        if v.get("gstin") == gstin and v.get("doc_type") == doc_type:
            out[v["layout_hash"]] = v["version"]
    return out


def kb_put(kb, gstin, doc_type, layout_hash, version, config_key, inner,
           fields_filled, fields_total):
    """Insert or update a solved template."""
    k = _key(gstin, doc_type, layout_hash)
    existing = kb.get(k)
    kb[k] = {
        "config_key": config_key,
        "gstin": gstin, "doc_type": doc_type, "layout_hash": layout_hash,
        "version": version,
        "config": inner,
        "fields_filled": fields_filled, "fields_total": fields_total,
        "created": existing["created"] if existing else _now(),
        "updated": _now(),
        "seen_count": (existing["seen_count"] + 1) if existing else 1,
    }
    return kb[k]


def kb_bump_seen(kb, gstin, doc_type, layout_hash):
    e = kb_get(kb, gstin, doc_type, layout_hash)
    if e:
        e["seen_count"] = e.get("seen_count", 1) + 1
        e["updated"] = _now()


def kb_configs_for_gstin(kb, gstin):
    """All {config_key: inner} for a vendor — used to write the per-GSTIN output file."""
    out = {}
    for v in kb.values():
        if v.get("gstin") == gstin:
            out[v["config_key"]] = v["config"]
    return order_blocks(out)


def order_blocks(cfg: dict) -> dict:
    """Blocks in a stable, readable order: a plain 'GSTIN' block first, then by
    document type, then V1, V2, ... V10 (numeric, so V10 comes after V9, not after V1)."""
    def rank(key):
        parts = str(key).split(":")
        if len(parts) < 3:
            return (0, str(key), 0)
        try:
            n = int(parts[2].lstrip("Vv"))
        except ValueError:
            n = 10 ** 6
        return (1, f"{parts[0]}:{parts[1]}", n)
    return {k: cfg[k] for k in sorted(cfg, key=rank)}
