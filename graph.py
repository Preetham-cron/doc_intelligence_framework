"""
graph.py  —  the extraction agent as a LangGraph state machine (one invoice).

  upload -> ocr -> qr -> discover -> generate -> test
                                        ^__________|   (retry while unresolved)
                                   -> fallback -> crosscheck -> assemble -> END

Spec compliance:
  - discover/verify: only raw-text evidence; a value must literally be in the text.
  - generate/test  : the LLM proposes an ORDERED LIST of candidate regex per field
                     (primary + fallbacks); test keeps only the candidates that
                     extract the value AND generalize to siblings AND aren't overfit.
  - fallback       : for fields the LLM could not resolve, a GENERAL (vendor-agnostic)
                     regex is auto-built from where the value sits in the raw text.
                     No hardcoded, per-vendor pattern library is involved.
  - crosscheck     : QR compared for confidence only; never changes decisions.
  - assemble       : old config format; ONLY present fields get rules.
"""
import os, sys, time
from typing import TypedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END

from src.markdownextractor.markdown_extractor import extract_markdown_from_blob
from src.qrcode.qr_extractor import QRCodeExtractor
from src.discover import discover, verify_in_text
from src.generate import call_gemini
from src.field_tester import (test_field, match_span, value_context, matches,
                              spec_is_overfit, regex_is_dangerous)
from src.fingerprint import layout_hash, qr_crosscheck
from src.assemble import assemble_config

MAX_RETRIES = 2   # Gemini is the sole regex generator now, so give it up to 3 tries
                  # (attempt 0 + 2 retries) with failure feedback before a field goes
                  # to review. No Python-built regex fallback.
DISTINCT = [{"total_invoice_cgst_tax_value", "total_invoice_sgst_tax_value"}]
_QR = QRCodeExtractor()


class S(TypedDict, total=False):
    pdf_path: str
    blob: bytes
    raw_text: str
    is_scanned: bool
    qr: dict
    gstin: str
    doc_type: str
    layout_hash: str
    sibling_texts: list
    deadline: float
    reverify: list
    member_paths: list
    primary_name: str
    settings: dict
    pre_passing: dict
    sibling_qrs: list
    sibling_names: list
    sibling_truth: list
    nil: list
    forced_present: dict
    present: dict
    evidence: dict
    trusted: dict
    pending: list
    passing: dict
    attempt: int
    report: list
    feedback: str
    _specs: dict
    crosscheck: dict
    config: dict


def _doc_type(raw):
    """A real CREDIT NOTE carries a 'CREDIT NOTE NO' label or a standalone
    'CREDIT NOTE' title. Boilerplate mentioning 'credit note rule' does NOT make
    a tax invoice a credit note."""
    u = (raw or "").upper()
    lines = [l.strip() for l in u.splitlines()]
    if "CREDIT NOTE NO" in u or any(l == "CREDIT NOTE" for l in lines):
        return "CREDITNOTE"
    if "TAX INVOICE" in u:
        return "TAXINVOICE"
    if "GST INVOICE" in u:
        return "GSTINVOICE"
    return "INVOICE"


def doctype_from_discovery(discovered_dt, raw):
    s = (discovered_dt or "").upper()
    if "CREDIT NOTE" in s: return "CREDITNOTE"
    if "GST INVOICE" in s: return "GSTINVOICE"
    if "TAX INVOICE" in s: return "TAXINVOICE"
    return _doc_type(raw)


def _step(n, title, detail=""):
    print(f"\n── STEP {n}: {title} " + "─" * max(0, 46 - len(title)))
    if detail:
        print(f"   {detail}")


def n_upload(s: S) -> S:
    with open(s["pdf_path"], "rb") as f:
        blob = f.read()
    _step(1, "UPLOAD", f"read PDF as bytes  ({len(blob)} bytes)")
    return {"blob": blob}


def n_ocr(s: S) -> S:
    from src.pipeline_text import pipeline_text
    if s.get("raw_text"):                        # already built during grouping -> reuse
        raw, scanned = s["raw_text"], bool(s.get("is_scanned"))
    else:
        raw, scanned, _ = pipeline_text(s["blob"])   # SAME builder the siblings use
    _step(2, "OCR / TEXT EXTRACTION", "PDF -> raw text (matches header_extraction_v2.py)")
    print(f"   raw_text = {len(raw)} chars | scanned = {scanned} | doc_type = {_doc_type(raw)}")
    return {"raw_text": raw, "is_scanned": scanned, "doc_type": _doc_type(raw)}


def n_qr(s: S) -> S:
    if s.get("qr") is not None:                  # already decoded during grouping -> reuse
        qr = s["qr"] or {}
    else:
        qr = _QR.extract_from_blob(s["blob"], invoice_id=os.path.basename(s["pdf_path"])) or {}
    lh = layout_hash(s["raw_text"])
    _step(3, "QR EXTRACTION (cross-check only)", "decode signed e-invoice QR")
    print(f"   seller_gstin = {qr.get('seller_gstin')} | qr_fields = {sorted(qr.keys()) if qr else 'none'}")
    print(f"   layout_hash = {lh}  (identifies the template)")
    return {"qr": qr, "gstin": qr.get("seller_gstin") or "UNKNOWN", "layout_hash": lh}


_TAX_AMT = {"total_invoice_cgst_tax_value", "total_invoice_sgst_tax_value",
            "total_invoice_igst_tax_value", "total_invoice_cess_tax_value"}


def _is_zero(v):
    try:
        return float(str(v).replace(",", "").strip() or "0") == 0.0
    except ValueError:
        return str(v).strip().lower() in ("nil", "-", "na", "n/a")


def _trusted_values(raw, qr, present, forced=None, who="this invoice"):
    """Turn one invoice's discovered values into TRUSTED values — the same rules for
    the primary and every sibling:
      - signed QR overrides: invoice no/date, SELLER gstin, IRN;
      - every value must literally appear in this invoice's text;
      - the tax regime (inter/intra-state from the QR state codes) files the tax
        under the right field;
      - a tax whose value is ZERO is 'nil' (defaults to 0.0, no regex needed).
    Returns (values, nil_fields, info_for_log)."""
    qr = qr or {}
    present = dict(present or {})
    for qk, f in (("doc_no", "supplier_document_number"), ("doc_dt", "supplier_document_date"),
                  ("seller_gstin", "gstin_number"), ("irn", "irn_number")):
        if qr.get(qk):
            present[f] = qr[qk]
    for f, v in (forced or {}).items():
        if f not in present:
            from src.autofallback import tax_present_in
            if tax_present_in(f, raw):
                present[f] = v
    verified = {f: v for f, v in present.items() if verify_in_text(v, raw)}
    # placeholder dates (30-12-1899 / 01-01-1900: an EMPTY date in the vendor's ERP)
    # are not real values — never train a regex to extract them
    from src.field_tester import parse_date
    for f in [f for f in verified if "date" in f]:
        p = parse_date(verified[f])
        if p and p[2] < 1950:
            del verified[f]
    before = set(verified)
    regime_note = _fix_tax_regime(verified, qr, raw)
    regime_moved = before - set(verified)
    nil = [f for f in list(verified) if f in _TAX_AMT and _is_zero(verified[f])]
    for f in nil:
        del verified[f]
    arith = _arith_taxable(verified, raw, qr, who)
    dropped = [f for f in present if f not in verified and f not in nil and f not in regime_moved]
    return verified, set(nil), {"regime": regime_note, "dropped": dropped, "nil": nil,
                                "arith": arith}


SIBLING_CHECK = int(os.environ.get("SIBLING_CHECK", "3"))   # siblings value-checked per template


def n_discover(s: S) -> S:
    _step(4, "DISCOVER (evidence-grounded)", "LLM finds which schema fields are ACTUALLY present + their values")
    present, evidence = discover(s["raw_text"])
    verified, nil, info = _trusted_values(s["raw_text"], s.get("qr"), present, s.get("forced_present"),
                                          who=s.get("primary_name") or "this invoice")
    absent = [f for f in evidence if f not in verified]
    print(f"   present (evidence-backed): {sorted(verified)}")
    if info["regime"]:
        print(f"   [tax regime] {info['regime']}")
    if info.get("arith"):
        print(f"   [rate check] {info['arith']}")
    if info["dropped"]:
        print(f"   [hallucination guard] dropped (value not literally in text): {info['dropped']}")
    if info["nil"]:
        print(f"   [nil-tax] treated as 0.00 default (not blocking): {sorted(info['nil'])}")
    print(f"   skipped (no evidence in this invoice): {sorted(absent)}")

    # ---- SIBLINGS: discover their values too, so every regex is validated against
    # the CORRECT value on EVERY invoice of the template — not just "extracts
    # something". (A regex requiring 2 decimals passes on '5532.38' but fails on a
    # sibling printing '48203'; only a value check catches that.)
    sibs = s.get("sibling_texts", []) or []
    qrs = s.get("sibling_qrs", []) or [{}] * len(sibs)
    names = s.get("sibling_names", []) or [f"sibling{i+1}" for i in range(len(sibs))]
    truth = [None] * len(sibs)
    k = min(len(sibs), SIBLING_CHECK)
    if k:
        from concurrent.futures import ThreadPoolExecutor

        from src import usage as _usage
        _tag = _usage.get_tag()                      # carry the template tag into the pool

        def _one(i):
            _usage.set_tag(_tag)
            try:
                p, _ = discover(sibs[i])
            except Exception as e:
                return i, None, f"{type(e).__name__}: {e}"
            nm = names[i] if i < len(names) else f"sibling{i+1}"
            v, n, _info = _trusted_values(sibs[i], qrs[i] if i < len(qrs) else {}, p, who=nm)
            return i, {"name": nm, "values": v, "nil": n, "arith": _info.get("arith")}, None
        with ThreadPoolExecutor(max_workers=k) as ex:
            for i, t, err in ex.map(_one, range(k)):
                truth[i] = t
                if t and t.get("arith"):
                    print(f"   [rate check] {t['arith']}")
                if t:
                    shown = {f: t["values"][f] for f in sorted(t["values"])
                             if f in verified and f not in ("irn_number",)}
                    print(f"   sibling {t['name']}: {len(t['values'])} values discovered "
                          f"(checked against): {shown}")
                else:
                    print(f"   sibling {names[i] if i < len(names) else i}: discovery failed ({err}); "
                          f"only a weak 'extracts something' check will be used for it")
    # ---- REVIEW: re-verify the unresolved fields from the actual PDFs ----
    rv = list(s.get("reverify") or [])
    if rv:
        _reverify(s, rv, verified, nil, truth, sibs, qrs, names, k)

    passing = {}
    pre = s.get("pre_passing") or {}
    if pre:
        kept = []
        for f, spec in pre.items():
            if f not in verified:
                continue
            spec = {**spec, "name": f}
            good, alias_ok, _ = _validate_candidates(spec, f, s["raw_text"], verified[f], sibs, truth)
            if good or alias_ok:
                passing[f] = {**spec, "regex": good} if good else {**spec, "regex": []}
                kept.append(f)
        print(f"   [repair] kept {len(kept)} previously-solved field(s) that still validate on "
              f"all invoices: {sorted(kept)}")
    return {"present": verified, "evidence": evidence, "nil": sorted(nil),
            "trusted": dict(verified), "pending": [f for f in verified if f not in passing],
            "sibling_truth": truth, "passing": passing, "attempt": 0, "report": [],
            "feedback": ""}


_OPTIONAL_FIELDS = {"po_pds_number", "vendor_code", "order_date", "irn_number"}
import re as _re
_AMT_TOKEN_RE = _re.compile(r'(?<![\d.,])\d[\d,]*(?:\.\d{1,2})?(?![\d])')


_CG_RATES = (0.025, 0.06, 0.09, 0.14, 0.0125, 0.015, 0.0005)   # CGST / SGST rates
_IG_RATES = (0.05, 0.12, 0.18, 0.28, 0.025, 0.03, 0.001)        # IGST rates


def _rate_ok(base, tax, rates):
    """Is `tax` = `base` x one of the GST rates (allowing line-level rounding)?"""
    return any(abs(base - tax / r) <= max(1.5, 0.0001 * base) for r in rates)


def _arith_taxable(vals, raw, qr, who=""):
    """Taxable-value check. The taxable / assessable value is the base the tax is
    computed on, so one of these must hold:
      (1) RATE:  CGST (= SGST) = taxable x rate   or   IGST = taxable x rate
                 for a standard GST rate;
      (2) SUM:   taxable + all taxes = invoice total (+/- 1)   — this is what holds
                 when the lines carry MIXED rates, so no single rate fits.
    (The SUM alone is not enough to reject a value: on many supplier invoices the
    assessable value includes amortisation that is not billed in the total, and there
    the RATE identity is the one that holds.)
    Current value fits (1) or (2) -> kept. Otherwise another amount PRINTED on this
    invoice that fits (1), else one that fits (2), is the taxable value (copied from
    the text, never computed). If nothing fits and the current value is not even
    larger than the tax (e.g. the '18.00' of a rate column), it is REMOVED so a wrong
    number is never used as the expected value. Edits `vals`; returns a log note."""
    from src.field_tester import clean_amount
    cur = vals.get("total_invoice_taxable_value")
    if cur is None:
        return None
    cur_a = clean_amount(str(cur))
    tax, rates = None, None
    for f, rs in ((_CG, _CG_RATES), (_SG, _CG_RATES), (_IG, _IG_RATES)):
        if f in vals:
            t = clean_amount(str(vals[f]))
            if t and t > 0:
                tax, rates = t, rs
                break
    if tax is None or cur_a is None:
        return None
    tot = clean_amount(str(vals.get("total_invoice_value") or "")) if vals.get("total_invoice_value") else None
    taxes = sum((clean_amount(str(vals[f])) or 0.0) for f in (_CG, _SG, _IG, _CE) if f in vals)

    def sum_ok(a):
        return tot is not None and taxes > 0 and abs(a + taxes - tot) <= 1.0

    if _rate_ok(cur_a, tax, rates) or sum_ok(cur_a):
        return None                                       # consistent: keep
    by_rate, rate_d, by_sum = None, None, None
    for m in _AMT_TOKEN_RE.finditer(raw):
        a = clean_amount(m.group(0))
        if a is None or a <= tax or (tot is not None and abs(a - tot) < 0.01):
            continue
        if _rate_ok(a, tax, rates):
            d = min(abs(a - tax / r) for r in rates)
            if rate_d is None or d < rate_d:
                by_rate, rate_d = m.group(0), d
        elif by_sum is None and sum_ok(a):
            by_sum = m.group(0)
    if by_rate is not None:
        vals["total_invoice_taxable_value"] = by_rate
        return (f"taxable on {who}: {cur!r} -> {by_rate!r} (the tax {tax:g} = {by_rate} x GST rate; "
                f"{cur!r} is not the base the tax was computed on)")
    if by_sum is not None:
        vals["total_invoice_taxable_value"] = by_sum
        return (f"taxable on {who}: {cur!r} -> {by_sum!r} ({by_sum} + taxes {taxes:g} = invoice "
                f"total {tot:g}; {cur!r} fits neither a GST rate nor the total)")
    if cur_a <= tax:
        del vals["total_invoice_taxable_value"]
        return (f"taxable on {who}: {cur!r} removed — it is not larger than the tax {tax:g} and "
                f"no printed amount fits a GST rate or the invoice total")
    return None


_QR_FIELDS = {"gstin_number": "seller_gstin", "supplier_document_number": "doc_no",
              "supplier_document_date": "doc_dt", "irn_number": "irn"}


def _reverify(s, fields, verified, nil, truth, sibs, qrs, names, k):
    """Send the PDFs of the primary + value-checked siblings to Gemini in ONE request
    and re-read `fields` consistently across them. Corrected values replace the
    discovered ones only if they are literally printed in that invoice's text; values
    the signed QR provides are never overridden; taxes go through the same nil /
    inter-intra-state rules. Edits `verified`, `nil` and `truth` in place."""
    from src.discover import verify_fields
    cfg = s.get("settings") or {}
    paths = s.get("member_paths") or []
    pname = s.get("primary_name") or "primary"
    mem = [{"name": pname, "pdf_path": s.get("pdf_path"), "raw": s["raw_text"]}]
    for j in range(k):
        mem.append({"name": names[j] if j < len(names) else f"sibling{j+1}",
                    "pdf_path": paths[j + 1] if j + 1 < len(paths) else None, "raw": sibs[j]})
    stores = [(verified, nil, s["raw_text"], s.get("qr") or {})]
    for j in range(k):
        t = truth[j]
        if t is None:
            t = truth[j] = {"name": mem[j + 1]["name"], "values": {}, "nil": set()}
        stores.append((t["values"], t["nil"], sibs[j], qrs[j] if j < len(qrs) else {}))
    previous = {m["name"]: {f: st[0].get(f) for f in fields} for m, st in zip(mem, stores)}
    print(f"   [verify] re-reading {fields} from the PDFs of {len(mem)} invoice(s) "
          f"({cfg.get('verify_model', cfg.get('model', 'gemini-2.5-pro'))})")
    try:
        res = verify_fields(mem, fields, previous,
                            model_name=cfg.get("verify_model", cfg.get("model", "gemini-2.5-pro")),
                            think=max(1024, int(cfg.get("think_first") or 4096)))
    except Exception as e:
        print(f"   [verify] failed ({type(e).__name__}: {e}); keeping the discovered values")
        return
    for m, (vals, nilset, raw, qr) in zip(mem, stores):
        got = res.get(m["name"]) or {}
        for f in fields:
            if f in _QR_FIELDS and qr.get(_QR_FIELDS[f]):
                continue                                 # the signed QR is authoritative
            if f not in got:
                continue
            new, old = got[f], vals.get(f)
            optional = f in _OPTIONAL_FIELDS
            if new in (None, "", "null"):
                if old is not None and optional:
                    vals.pop(f, None)
                    print(f"   [verify] {f} on {m['name']}: {old!r} -> not on this invoice")
                elif old is not None:
                    # a REQUIRED value that discovery found must not silently vanish —
                    # that would store the template with this field empty (reads 0.00
                    # in production). Keep it; if no regex can be validated the
                    # template stays in review.
                    print(f"   [verify] {f} on {m['name']}: Gemini says not on this invoice, but "
                          f"discovery found {old!r}; keeping it (stays in review if unsolved)")
                continue
            if not verify_in_text(new, raw):
                print(f"   [verify] {f} on {m['name']}: {new!r} is not printed in the text; ignored")
                continue
            if f in _TAX_AMT and _is_zero(new):
                if old is not None and not _is_zero(old):
                    print(f"   [verify] {f} on {m['name']}: Gemini says 0 but discovery found "
                          f"{old!r}; keeping it (stays in review if unsolved)")
                else:
                    vals.pop(f, None); nilset.add(f)
                continue
            nilset.discard(f)
            vals[f] = new
            if str(new) != str(old):
                print(f"   [verify] {f} on {m['name']}: {old!r} -> {new!r}")
        _fix_tax_regime(vals, qr, raw)
        note = _arith_taxable(vals, raw, qr, m["name"])
        if note:
            print(f"   [rate check] {note}")


_CG, _SG, _IG, _CE = ("total_invoice_cgst_tax_value", "total_invoice_sgst_tax_value",
                      "total_invoice_igst_tax_value", "total_invoice_cess_tax_value")


def _fix_tax_regime(present, qr, raw_text):
    """Put the tax amount under the RIGHT tax, using the SIGNED QR's seller/buyer
    GSTIN state codes (first 2 digits):
      seller state != buyer state -> INTER-state -> only IGST applies
      seller state == buyer state -> INTRA-state -> only CGST + SGST apply
    Invoices often print "CGST % | SGST/UTGST % | IGST %" as one header with a single
    amount under it, and discovery then files e.g. an 18% IGST of 941.09 as CGST/SGST.
    That value would be wrong in production and can never validate (CGST and SGST
    cannot both own one position). The moved amount must still literally appear in
    the text, and when there is a choice, the reading that makes
    taxable + tax (+ cess) == invoice total wins. Edits `present` in place; returns
    a note for the log, or None."""
    from src.field_tester import clean_amount
    seller = str((qr or {}).get("seller_gstin") or "")[:2]
    buyer = str((qr or {}).get("buyer_gstin") or "")[:2]
    if not (seller.isdigit() and buyer.isdigit()):
        return None

    def amt(f):
        v = present.get(f)
        a = clean_amount(str(v)) if v not in (None, "") else None
        return a or 0.0

    c, sg, i = amt(_CG), amt(_SG), amt(_IG)
    tot, tv, ce = amt("total_invoice_value"), amt("total_invoice_taxable_value"), amt(_CE)

    if seller != buyer and not i and (c or sg):                       # inter-state
        opts = {}
        for x, src in ((c, present.get(_CG)), (sg, present.get(_SG)), (c + sg, None)):
            if x:
                opts.setdefault(round(x, 2), src)
        if tot and tv:
            pick = min(opts, key=lambda x: abs(tv + x + ce - tot))
        else:
            pick = c if (c and abs(c - sg) < 0.01) else round(c + sg, 2)
        val = opts.get(pick) or f"{pick:.2f}"
        if verify_in_text(val, raw_text):
            present[_IG] = val
            present.pop(_CG, None); present.pop(_SG, None)
            return f"inter-state ({seller}->{buyer}): tax {val} moved to IGST (was CGST/SGST)"
    elif seller == buyer and i and not (c or sg):                     # intra-state
        half = f"{round(i / 2, 2):.2f}"
        if verify_in_text(half, raw_text):
            present[_CG] = half; present[_SG] = half
            present.pop(_IG, None)
            return f"intra-state ({seller}): IGST {i} split into CGST/SGST {half} each"
    elif seller != buyer and i and (c or sg):                         # both reported
        present.pop(_CG, None); present.pop(_SG, None)
        return f"inter-state ({seller}->{buyer}): kept IGST, dropped CGST/SGST"
    elif seller == buyer and i and c and sg:
        present.pop(_IG, None)
        return f"intra-state ({seller}): kept CGST/SGST, dropped IGST"
    return None


def _siblings_info(fields, texts, truths):
    """For the prompt: each field's CORRECT value on the OTHER invoices of this
    template and the lines around it, so the LLM writes ONE regex that fits all of
    them (e.g. it sees '48203' printed without decimals on another invoice)."""
    out = {}
    for f in fields:
        rows = []
        for i, t in enumerate(truths or []):
            if not t or i >= len(texts):
                continue
            if f in t["values"]:
                v = t["values"][f]
                rows.append({"invoice": t.get("name"), "value": v,
                             "where": value_context(texts[i], v, ctx=1, max_snips=2)})
            elif f in t["nil"]:
                rows.append({"invoice": t.get("name"), "value": "NIL on this invoice — the "
                             "regex must extract nothing (or 0) here"})
        if rows:
            out[f] = rows
    return out


def n_generate(s: S) -> S:
    text, trusted, pending = s["raw_text"], s["trusted"], s["pending"]
    if not pending:
        return {"_specs": {}}
    cfg = s.get("settings") or {}
    dl = s.get("deadline")
    if dl and time.time() > dl:
        print(f"   [time budget] this template's time budget is used up — no more LLM "
              f"attempts; unresolved {pending} stay for review")
        return {"_specs": {}}
    # REPAIR: values were just re-read from the PDFs, so try the instant code-built
    # candidates FIRST and only call the (slow) model for what they cannot solve
    if cfg.get("positional_first") and s.get("attempt", 0) == 0:
        passing = dict(s["passing"]); still = []
        for name in pending:
            pos = _try_positional(name, text, trusted[name], s.get("sibling_texts", []),
                                  s.get("sibling_truth"), None,
                                  _distinct_avoid(name, passing, text))
            if pos:
                passing[name] = pos
                print(f"     {name:<32}PASS   kept {len(pos['regex'])} regex (position-built, "
                      f"before any LLM call)")
            else:
                still.append(name)
        if not still:
            return {"_specs": {}, "passing": passing, "pending": []}
        pending = still
        s = {**s, "passing": passing, "pending": still}
    attempt = s["attempt"]
    _step(5, f"GENERATE REGEX (attempt {attempt})",
          f"LLM writes a regex for {len(pending)} pending field(s)"
          + (" — with failure feedback" if s.get("feedback") else ""))
    want = {k: trusted[k] for k in pending}
    contexts = {k: value_context(text, trusted[k]) for k in pending}
    # attempt 0: thinking off (fast, all fields). Retries carry only the hard fields,
    # so let the model reason about line positions there.
    cfg = s.get("settings") or {}
    think = cfg.get("think_first", 0) if attempt == 0 else cfg.get("think_retry", 2048)
    try:
        specs = call_gemini(text, want, pending, feedback=s.get("feedback", ""),
                            contexts=contexts, think=think,
                            model_name=cfg.get("model", "gemini-2.5-flash"),
                            siblings_info=_siblings_info(pending, s.get("sibling_texts", []),
                                                         s.get("sibling_truth")))
    except TypeError:
        specs = call_gemini(text, want, pending)
    out = {"_specs": {sp["name"]: sp for sp in specs}}
    if cfg.get("positional_first") and s.get("attempt", 0) == 0:
        out.update({"passing": s["passing"], "pending": pending})
    return out


def _tax_name_for(field):
    """The tax's own name for a tax-amount field (used to require the regex be
    anchored on it), else None."""
    for t in ("cgst", "sgst", "igst", "cess"):
        if t in field:
            return t.upper()
    return None


# Words that legitimately LABEL each tax on real invoices. A tax regex/alias must
# contain one of these, so it is anchored on that tax's own label (not purely
# positional). CESS is printed as "TCS" on many invoices ("TOTAL TCS VALUE"), and
# CGST/SGST/IGST appear as "Central/State/Integrated ... Tax" or under a generic
# "Total GST value" header (Asahi/AIS layout) — all of these are valid anchors.
_TAX_ANCHORS = {
    "CGST": ("CGST", "CENTRAL", "GST"),
    "SGST": ("SGST", "UTGST", "STATE", "GST"),
    "IGST": ("IGST", "INTEGRATED", "GST"),
    "CESS": ("CESS", "TCS"),
}


def _has_tax_anchor(tax_kw, text):
    up = (text or "").upper()
    return any(w in up for w in _TAX_ANCHORS.get(tax_kw, (tax_kw,)))


def _generalizes(spec, name, siblings, truth=None):
    """Does the spec work on the OTHER invoices of this template?
    For a sibling whose values were discovered (`truth[i]`), the spec must extract
    that sibling's CORRECT value — or, where this tax is nil on that sibling, nothing
    / 0. Only for a sibling without discovered values does it fall back to the weak
    'extracts something' check. Returns (ok, note)."""
    from src.autofallback import tax_present_in
    truth = truth or []
    for i, sib in enumerate(siblings):
        t = truth[i] if i < len(truth) else None
        if t is not None:
            nm = t.get("name", f"sibling{i+1}")
            if name in t["values"]:
                exp = t["values"][name]
                r = test_field(spec, sib, exp)
                if not r["passed"]:
                    return False, f"on {nm} it extracted {r['got']!r}, expected {exp!r}"
                continue
            if name in t["nil"]:
                try:
                    from src.field_tester import extract_field, RegexTimeout
                    got = extract_field(spec, sib)
                except Exception:
                    got = None
                if got not in (None, 0, 0.0, ""):
                    return False, f"on {nm} (where this tax is nil) it extracted {got!r}"
                continue
            # value not discovered on this sibling -> weak check below
        if name in TOTALS_FIELDS and not tax_present_in(name, sib):
            continue
        if not matches(spec, sib):
            return False, f"on sibling {i+1} it extracted nothing"
    return True, None


def _validate_candidates(spec, name, text, expected, siblings, truth=None):
    """Validate EACH regex in the LLM spec's ordered list independently and return
    the sublist that (a) is safe to run (no catastrophic backtracking), (b) is not
    copied page text, (c) for a TAX amount is anchored on that tax's label, (d)
    extracts `expected`, (e) is not value-overfit, and (f) generalizes to siblings —
    preserving the LLM's order (primary first).

    If NO regex works, try the LLM's alias+direction (production's extract_field runs
    the same label scan when the regex misses); accept an alias-only spec when it
    extracts `expected` and generalizes. A tax alias must contain the tax's label.

    Returns (good_regexes, alias_ok, reason). `reason` reports EVERY candidate's own
    failure, so the retry knows exactly what went wrong with each pattern."""
    pats = spec.get("regex", []) or []
    tax_kw = _tax_name_for(name)
    good, notes = [], []
    for pat in pats:
        short = (pat[:70] + "...") if len(pat) > 70 else pat
        if regex_is_dangerous(pat):
            notes.append(f"REJECTED: nested repeats like (...+)+ backtrack catastrophically; use a "
                         f"flat pattern [{short!r}]"); continue
        if _too_literal(pat):
            notes.append(f"copies a long stretch of page text; use a SHORT label anchor [{short!r}]"); continue
        if tax_kw and not _has_tax_anchor(tax_kw, pat):
            words = "/".join(_TAX_ANCHORS.get(tax_kw, (tax_kw,)))
            notes.append(f"has no {tax_kw} label word ({words}) in it [{short!r}]"); continue
        # isolate the REGEX (drop the alias) so we judge the pattern itself
        single = {**spec, "regex": [pat], "alias": []}
        r = test_field(single, text, expected)
        if r.get("timeout"):
            notes.append(f"too slow (catastrophic backtracking); simplify it [{short!r}]"); continue
        if not r["passed"]:
            notes.append(f"extracted {r['got']!r} on this invoice, expected {expected!r} [{short!r}]"); continue
        if name != "document_type" and spec_is_overfit(single, expected):
            notes.append(f"embeds characters copied from the value; capture a GENERIC token after "
                         f"the printed label [{short!r}]"); continue
        ok, why = _generalizes(single, name, siblings, truth)
        if not ok:
            notes.append(f"correct on this invoice but WRONG {why}; it must work on every invoice "
                         f"of the template [{short!r}]"); continue
        good.append(pat)
    if good:
        return good, False, None

    # ---- ALIAS FALLBACK (label + direction), production-faithful ----
    aliases = spec.get("alias", []) or []
    if aliases and (not tax_kw or any(_has_tax_anchor(tax_kw, a) for a in aliases)):
        alias_spec = {**spec, "regex": []}           # force the label scan only
        r = test_field(alias_spec, text, expected)
        ok, why = _generalizes(alias_spec, name, siblings, truth) if r["passed"] else (False, None)
        if ok:
            return [], True, None
        if r["passed"]:
            r = {**r, "got": f"{r.get('got')!r} here, but {why}"}
        notes.append(f"alias {aliases} + direction {spec.get('direction')!r}: extracted "
                     f"{r.get('got')!r}, expected {expected!r}")
    if not pats and not aliases:
        notes.append("no regex and no alias were given")
    detail = " | ".join(notes)
    return [], False, (f"- {name}: nothing extracted {expected!r}. Per candidate: {detail}. "
                       f"Look at WHERE {expected!r} sits in the raw text (see the context "
                       f"lines), anchor on the nearest STABLE label or neighbouring line, "
                       f"and also give an `alias` (printed label) + `direction`.")


def _too_literal(pat: str) -> bool:
    """True if a regex is mostly COPIED PAGE TEXT rather than a pattern: after
    removing escapes, classes and quantifiers, more than ~120 literal characters
    remain (a normal label anchor is well under 40)."""
    import re as _re
    s = _re.sub(r'\\.', '', pat or '')
    s = _re.sub(r'\[[^\]]*\]|\{[^}]*\}|\(\?[:=!<]+|[()*+?|^$]', '', s)
    return len(s) > 120


def _canon_dt(name):
    return "gstin" if name == "gstin_number" else _datatype_of(name)


def _distinct_avoid(name, passing, text):
    """Spans already taken by the other member of a CGST/SGST pair (they are often
    equal in value, so each must capture a DIFFERENT position)."""
    spans = set()
    for group in DISTINCT:
        if name in group:
            for other in group:
                if other != name and other in passing:
                    sp = match_span(passing[other], text)
                    if sp is not None:
                        spans.add(sp)
    return spans


def _try_positional(name, text, expected, siblings, truth, spec=None, avoid_spans=()):
    """CODE-BUILT candidates (src/position_regex.py) for a field the LLM could not
    solve: anchored on text that is identical on every invoice of the template at a
    fixed line distance from the value. Validated EXACTLY like the LLM's candidates
    (correct value on every invoice, not overfit, safe, tax label present).
    Returns a passing spec or None."""
    if os.environ.get("POSITION_REGEX", "1") == "0":      # switch off for A/B comparison
        return None
    from src.position_regex import build_candidates, MAX_KEEP
    dt = _canon_dt(name)
    members = [(text, expected)]
    for i, sib in enumerate(siblings or []):
        t = truth[i] if truth and i < len(truth) else None
        if t and name in t["values"]:
            members.append((sib, t["values"][name]))
    try:
        cands = build_candidates(name, dt, members)
    except Exception:
        return None
    if avoid_spans:
        cands = [c for c in cands if match_span({"regex": [c]}, text) not in avoid_spans]
    if not cands:
        return None
    base = {k: v for k, v in (spec or {}).items() if k not in ("regex", "alias", "direction")}
    sp = {**base, "name": name, "datatype": dt, "regex": cands, "alias": [], "direction": "right",
          "source": "position"}
    good, _, _ = _validate_candidates(sp, name, text, expected, siblings, truth)
    return {**sp, "regex": good[:MAX_KEEP]} if good else None


def n_test(s: S) -> S:
    text, trusted = s["raw_text"], s["trusted"]
    siblings = s.get("sibling_texts", [])
    truth = s.get("sibling_truth")
    by = s.get("_specs", {})
    passing = dict(s["passing"]); report = list(s["report"]); attempt = s["attempt"]
    _step(6, f"TEST + VALIDATE (attempt {attempt})",
          f"validate each candidate regex; keep the ones that extract the value AND "
          f"generalize to {len(siblings)} sibling(s)")
    still, fails = [], []

    def positional(name, spec, why):
        pos = _try_positional(name, text, trusted[name], siblings, truth, spec,
                              _distinct_avoid(name, passing, text))
        if pos:
            passing[name] = pos
            print(f"     {name:<32}PASS   kept {len(pos['regex'])} regex (position-built; "
                  f"LLM {why})")
            return True
        return False

    for name in s["pending"]:
        spec = by.get(name)
        if not spec:
            if positional(name, None, "returned no spec"):
                continue
            still.append(name)
            fails.append(f"- {name}: you returned NO spec for this field. Return a spec "
                         f"for it (name={name!r}) with regex candidates AND an alias+direction.")
            print(f"     {name:<32}FAIL   no spec returned by the LLM")
            continue
        good, alias_ok, reason = _validate_candidates(spec, name, text, trusted[name], siblings,
                                                      truth)
        report.append({"attempt": attempt, "field": name,
                       "candidates": len(spec.get("regex", []) or []), "kept": len(good)})
        if good:
            passing[name] = {**spec, "regex": good}
            extra = f"  (+{len(good)-1} fallback)" if len(good) > 1 else ""
            print(f"     {name:<32}PASS   kept {len(good)}/{len(spec.get('regex',[]) or [])} regex{extra}")
        elif alias_ok:
            # no working regex, but the printed-label scan extracts it -> keep an
            # alias-only spec (production's extract_field runs the same label scan).
            passing[name] = {**spec, "regex": []}
            print(f"     {name:<32}PASS   via alias {spec.get('alias')} (label scan, no regex)")
        elif positional(name, spec, "regex failed"):
            pass
        else:
            still.append(name)
            fails.append(reason)
            print(f"     {name:<32}FAIL   {reason.split('Per candidate: ',1)[-1][:110]}")
    # CGST/SGST must capture DIFFERENT positions; on a clash re-solve the second one
    # with position-built candidates that avoid the first one's span
    for group in DISTINCT:
        seen = {}
        for name in group:
            if name in passing:
                sp = match_span(passing[name], text)
                if sp is not None and sp in seen.values():
                    spec = passing.pop(name)
                    if not positional(name, spec, "clashed with its pair"):
                        if name not in still:
                            still.append(name)
                else:
                    seen[name] = sp
    if still:
        print(f"   -> {len(still)} field(s) still unresolved -> retry: {still}")
    else:
        print(f"   -> all fields resolved")
    return {"passing": passing, "pending": still,
            "report": report, "feedback": "\n".join(f for f in fails if f), "attempt": attempt + 1}


def n_crosscheck(s: S) -> S:
    _step(7, "QR CROSS-CHECK (confidence only)", "compare extracted values to the signed QR (never changes them)")
    cc = qr_crosscheck({k: s["trusted"][k] for k in s["passing"] if k in s["trusted"]},
                       s.get("qr", {}))
    matched = [f for f, v in cc.items() if v == "match"]
    mism = [f for f, v in cc.items() if v == "mismatch"]
    print(f"   agree with QR: {matched}")
    if mism:
        from src.fingerprint import _QR_MAP
        detail = {f: f"extracted {s['trusted'].get(f)!r} vs QR {(s.get('qr') or {}).get(_QR_MAP[f])!r}"
                  for f in mism}
        print(f"   MISMATCH vs QR (info only, not changed): {detail}")
    return {"crosscheck": cc}


def n_assemble(s: S) -> S:
    _step(8, "ASSEMBLE CONFIG", "write old-format config; only present fields get rules")
    cfg = assemble_config(s["passing"], gstin=s["gstin"], doc_type=s["doc_type"],
                          version="V1", include_unfilled=True,
                          prompt_note=[f"Evidence-grounded. layout_hash={s['layout_hash']}."])
    print(f"   filled {len(s['passing'])} field(s): {sorted(s['passing'])}")
    if s["pending"]:
        print(f"   left empty (unresolved): {s['pending']}")
    _add_irn_fallback(s, cfg)
    _add_field_prompts(s, cfg)
    return {"config": cfg}


# An IRN has the same format on every e-invoice in India (64 hex characters), so one
# generic pattern can find it even when none of the sample invoices printed an IRN.
IRN_FALLBACK = r"\b([0-9a-f]{64})\b"


def _add_irn_fallback(s, cfg):
    """Add the generic IRN pattern as the LAST regex of irn_number (vendor-specific
    rules are tried first). Left out when, on any sample invoice, it would pick up a
    64-character value that is not the IRN in that invoice's QR code."""
    import re
    if os.environ.get("IRN_FALLBACK", "1") == "0":
        return
    try:
        fdef = list(cfg.values())[0].get("header", {}).get("irn_number")
        if not isinstance(fdef, dict):
            return
        texts = [s.get("raw_text") or ""] + list(s.get("sibling_texts") or [])
        qrs = [s.get("qr") or {}] + list(s.get("sibling_qrs") or [])
        for i, text in enumerate(texts):
            m = re.search(IRN_FALLBACK, text or "", re.I)
            want = str(((qrs[i] if i < len(qrs) else None) or {}).get("irn") or "").strip().lower()
            if m and want and m.group(1).lower() != want:
                print("   generic IRN pattern NOT added: on a sample invoice it finds a "
                      "64-character value that is not the QR's IRN")
                return
        rx = fdef.setdefault("regex", [])
        if IRN_FALLBACK not in rx:
            rx.append(IRN_FALLBACK)
            print("   added the generic IRN pattern (64 hex characters) as last fallback")
    except Exception as e:
        print(f"   generic IRN pattern skipped ({type(e).__name__}: {e})")


_PROMPT_OPTIONAL = {"po_pds_number", "vendor_code", "order_date", "irn_number"}


def _add_field_prompts(s, cfg):
    """Replace the generic per-field "prompt" lines with instructions written for THIS
    vendor layout (label, position, format, what not to pick). Production sends that
    prompt to Gemini when a regex fails on an invoice. Only done for a config that is
    going to be stored (nothing required is pending); one Gemini call per layout."""
    if any(f not in _PROMPT_OPTIONAL for f in (s.get("pending") or [])):
        return
    try:
        from src.field_prompts import write_field_prompts
        from src.assemble import CANONICAL_ORDER
        inner = list(cfg.values())[0]
        header = inner.get("header", {})
        trusted = s.get("trusted") or {}
        values = {f: trusted[f] for f in s["passing"] if f in trusted}
        patterns = {f: {"regex": (header.get(f) or {}).get("regex", [])[:2],
                        "alias": (header.get(f) or {}).get("alias", [])}
                    for f in values}
        absent = [f for f in CANONICAL_ORDER if f in header and f not in s["passing"]]
        sibs = [{k: v for k, v in (t.get("values") or {}).items() if k in values}
                for t in (s.get("sibling_truth") or []) if t][:2]
        written = write_field_prompts(s["raw_text"], values, patterns, absent, sibs)
        for f, lines in written.items():
            if f in header:
                header[f]["prompt"] = lines
        if written:
            print(f"   wrote layout-specific prompts for {len(written)} field(s)")
    except Exception as e:                       # a prompt must never break a build
        print(f"   [prompts] skipped ({type(e).__name__}: {e})")


TOTALS_FIELDS = {"invoice_total_basic_value", "total_invoice_taxable_value",
                 "total_invoice_igst_tax_value", "total_invoice_cgst_tax_value",
                 "total_invoice_sgst_tax_value", "total_invoice_cess_tax_value",
                 "total_invoice_value"}


def _datatype_of(field):
    if "date" in field:
        return "date"
    if field in TOTALS_FIELDS:
        return "amount"
    if "gstin" in field:
        return "gstin"
    return "string"


def n_fallback(s: S) -> S:
    """UNION generation via GEMINI (no Python-built regexes). The main generate/test
    loop already gave Gemini several tries on the PRIMARY invoice; whatever it could
    not resolve there is left for review. The one thing that loop cannot cover is the
    TAX UNION: a template can be IGST on some invoices and CGST/SGST on others (same
    layout_hash), and a tax that is ABSENT on the primary but present on a SIBLING
    still needs a regex in the one shared config. Here we ask GEMINI to write that
    regex from the sibling that actually carries the tax, then validate it (extract
    the value, generalize across members that have the tax, distinct cgst/sgst span).
    Every regex still comes from Gemini; Python only validates and assembles."""
    from src.autofallback import tax_present_in, find_tax_value
    primary = s["raw_text"]
    siblings = s.get("sibling_texts", [])
    members = [primary] + list(siblings)
    trusted = dict(s["trusted"])
    forced = s.get("forced_present", {}) or {}
    passing = dict(s["passing"]); pending = list(s["pending"])
    # per-member truth: primary's trusted values + each value-checked sibling's
    truth = ([{"name": "primary", "values": trusted, "nil": set(s.get("nil") or [])}]
             + list(s.get("sibling_truth") or [None] * len(siblings)))
    truth += [None] * (len(members) - len(truth))

    tax_amount_fields = {"total_invoice_cgst_tax_value", "total_invoice_sgst_tax_value",
                         "total_invoice_igst_tax_value", "total_invoice_cess_tax_value"}

    def carries(f, k):
        """Does member k carry tax f (non-nil)? Discovered values first; the adjacency
        detector only for members whose values were not discovered."""
        if truth[k] is not None:
            return f in truth[k]["values"]
        return tax_present_in(f, members[k])

    def value_of(f, k):
        if truth[k] is not None:
            return truth[k]["values"].get(f)
        return find_tax_value(f, members[k])

    # union tax fields: absent/nil on the PRIMARY but carried by SOME sibling
    candidates = set(forced) | {f for t in truth[1:] if t for f in t["values"]}
    union = [f for f in sorted(candidates) if f in tax_amount_fields and f not in passing
             and f not in trusted and any(carries(f, k) for k in range(1, len(members)))]
    if not union:
        still = [f for f in pending if f not in passing]
        return {"passing": passing, "pending": still, "trusted": trusted}

    _step(6, "UNION GENERATE (Gemini on the sibling that carries the tax)",
          f"asking Gemini for {sorted(union)}")

    for k, m in enumerate(members):
        want = {f: value_of(f, k) for f in union if f not in passing and carries(f, k)}
        want = {f: v for f, v in want.items() if v}
        if not want:
            continue
        contexts = {f: value_context(m, want[f]) for f in want}
        others = [x for j, x in enumerate(members) if j != k]
        others_truth = [t for j, t in enumerate(truth) if j != k]
        try:
            specs = call_gemini(m, want, list(want), feedback=s.get("feedback", ""),
                                contexts=contexts,
                                siblings_info=_siblings_info(list(want), others, others_truth))
        except TypeError:
            specs = call_gemini(m, want, list(want))
        by = {sp["name"]: sp for sp in specs}
        for name in list(want):
            spec = by.get(name)
            if not spec:
                pos = _try_positional(name, m, want[name], others, others_truth, None,
                                      _distinct_avoid(name, passing, m))
                if pos:
                    passing[name] = pos
                    print(f"     {name:<32}PASS   got={want[name]}  (position-built/union)")
                continue
            good, alias_ok, reason = _validate_candidates(spec, name, m, want[name], others,
                                                          others_truth)
            if good:
                passing[name] = {**spec, "regex": good}
                print(f"     {name:<32}PASS   got={want[name]}  (Gemini/union)")
            elif alias_ok:
                passing[name] = {**spec, "regex": []}
                print(f"     {name:<32}PASS   got={want[name]}  (alias/union)")
            else:
                pos = _try_positional(name, m, want[name], others, others_truth, spec,
                                      _distinct_avoid(name, passing, m))
                if pos:
                    passing[name] = pos
                    print(f"     {name:<32}PASS   got={want[name]}  (position-built/union)")
                else:
                    print(f"     {name:<32}unresolved  ({reason.split('Per candidate: ',1)[-1][:90]})")
    # a union tax left unresolved would read 0.00 on the invoices that carry it ->
    # that is a WRONG value in production, so it blocks storage like any other field
    for f in union:
        if f not in passing and f not in pending:
            pending.append(f)

    # cgst vs sgst must capture DIFFERENT offsets (often equal in value)
    for group in DISTINCT:
        seen = {}
        for name in group:
            if name in passing:
                sp = match_span(passing[name], primary) or match_span(passing[name], members[-1])
                if sp is not None and sp in seen.values():
                    del passing[name]
                else:
                    seen[name] = sp

    still = [f for f in pending if f not in passing]
    if still:
        print(f"   -> still unresolved (-> review): {still}")
    return {"passing": passing, "pending": still, "trusted": trusted}


def _retry(s: S) -> str:
    dl = s.get("deadline")
    if dl and time.time() > dl:
        return "fallback"                       # time budget used up -> finish now
    max_r = (s.get("settings") or {}).get("max_retries", MAX_RETRIES)
    return "generate" if (s["pending"] and s["attempt"] <= max_r) else "fallback"


def build():
    g = StateGraph(S)
    for name, fn in [("upload", n_upload), ("ocr", n_ocr), ("qr", n_qr),
                     ("discover", n_discover), ("generate", n_generate),
                     ("test", n_test), ("fallback", n_fallback),
                     ("crosscheck", n_crosscheck), ("assemble", n_assemble)]:
        g.add_node(name, fn)
    g.set_entry_point("upload")
    g.add_edge("upload", "ocr"); g.add_edge("ocr", "qr"); g.add_edge("qr", "discover")
    g.add_edge("discover", "generate"); g.add_edge("generate", "test")
    g.add_conditional_edges("test", _retry, {"generate": "generate", "fallback": "fallback"})
    g.add_edge("fallback", "crosscheck")
    g.add_edge("crosscheck", "assemble"); g.add_edge("assemble", END)
    return g.compile()


AGENT = build()