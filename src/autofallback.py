r"""
autofallback.py — a GENERAL, vendor-agnostic fallback regex generator.

Runs ONLY when the LLM fails to produce a working regex for a field. It contains
NO vendor-specific labels, prefixes or shapes. It looks at WHERE the field's
(trusted) value actually appears in THIS invoice's raw text and builds a candidate
regex from the value's own neighbourhood, using general strategies:

  A) LABEL ON THE SAME LINE   "PO No.  6000012317,"        -> <label>\s*[:.\-]?\s*(token)
  B) LABEL ON THE LINE ABOVE  "SGST Value 9.00 %\n1,274.41" -> <label>[^\n]*\n\s*(token)
  C) POSITIONAL TO A KEYWORD  a bare amount in a totals row ending in "Total"
                              -> (amt)(\s*\n\s*amt){k}\s*\n\s*<keyword>
  T) TAX-NAME ANCHORED        for a tax amount, anchor on the tax's own word
                              (CGST/SGST/IGST/CESS) and walk to the amount.

Every candidate is handed back to the SAME tester the LLM specs go through (must
extract the trusted value AND generalize to siblings AND not be value-overfit).
If none survive, the field is left unresolved (-> review). No shared, hand-written
pattern library is involved.
"""
import re
from src.field_tester import date_variants

AMT = r'[\d,]+\.\d{2}'


def _token_for(value, datatype):
    """A GENERIC capture token appropriate to the value's type — never the value."""
    v = str(value).strip()
    if datatype == "amount":
        return AMT
    if datatype == "date":
        return r'\d{1,2}[./\-][A-Za-z0-9]{2,}[./\-]\d{2,4}'
    if datatype == "gstin":
        return r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]'
    if re.fullmatch(r'\d+', v):
        n = len(v)
        lo = max(4, n - 2)
        return r'\d{' + f'{lo},{n+2}' + r'}'
    return r'[A-Z0-9][A-Z0-9/\-]{3,}'


def _printed_variants(value, datatype):
    """The printed forms the value might take in the raw text."""
    v = str(value).strip()
    out = {v}
    if datatype == "amount":
        try:
            f = float(v.replace(',', ''))
            out.add(f"{f:,.2f}"); out.add(f"{f:.2f}")
        except ValueError:
            pass
    if datatype == "date":
        out.update(date_variants(value))
    return [x for x in out if x]


def _label_prefix(text_before):
    """Trailing LABEL (alphabetic, allowing . / # spaces) preceding the value on the
    same line. Returns the label or None."""
    m = re.search(r'([A-Za-z][A-Za-z0-9 .\-/#&]*?)\s*[:.\-]?\s*$', text_before)
    if not m:
        return None
    label = m.group(1).strip()
    return label if re.search(r'[A-Za-z]{2,}', label) else None


def _label_line(line):
    """The alphabetic prefix of a 'label line'."""
    m = re.match(r'\s*([A-Za-z][A-Za-z0-9 .\-/#&]*[A-Za-z])', line)
    return m.group(1).strip() if m else None


def _field_hint(field):
    """A disambiguating keyword derived from the FIELD'S OWN NAME (tax type)."""
    for t in ("cgst", "sgst", "igst", "cess"):
        if t in field:
            return t.upper()
    return None


def _candidates_at(field, idx, lines, variants, token, datatype):
    """Yield (priority, regex) candidates for the value occurrence on line `idx`.
    Lower priority = more reliable. Label-BEFORE-value anchors (0,1) beat positional
    anchors (2,3)."""
    line = lines[idx]
    pos = min((line.find(v) for v in variants if v in line), default=-1)

    # Strategy A (prio 0): label on the SAME line, before the value
    if pos > 0:
        lab = _label_prefix(line[:pos])
        if lab:
            yield 0, re.escape(lab) + r'\s*[:.\-]?\s*(' + token + r')'

    # Strategy B (prio 1): label on the nearest non-empty line ABOVE
    for j in range(idx - 1, max(-1, idx - 4), -1):
        prev = lines[j].strip()
        if not prev:
            continue
        lab = _label_line(prev)
        if lab and not any(v in prev for v in variants):
            yield 1, re.escape(lab) + r'[^\n]*\n\s*(' + token + r')'
        break

    # Strategy C (amounts only): positional to a stable KEYWORD
    if datatype == "amount":
        # C-back (prio 2): keyword ABOVE, then k amounts, then the value
        k, keyword = 0, None
        for j in range(idx - 1, max(-1, idx - 12), -1):
            s = lines[j].strip()
            if not s:
                continue
            if re.fullmatch(AMT, s) or re.fullmatch(r'[\d,]+\.?\d*', s) or re.fullmatch(r'\d+', s):
                k += 1; continue
            kw = re.search(r'([A-Za-z][A-Za-z .]*[A-Za-z])\s*$', s)
            if kw:
                keyword = kw.group(1).strip()
            break
        if keyword:
            mid = (AMT + r'\s*\n\s*') * k
            yield 2, re.escape(keyword) + r'\b[^\n]*\n\s*' + mid + r'(' + AMT + r')'
        # C-fwd (prio 3): the value, then k amounts, then a keyword BELOW
        k, keyword = 0, None
        for j in range(idx + 1, min(len(lines), idx + 12)):
            s = lines[j].strip()
            if not s:
                continue
            if re.fullmatch(AMT, s) or re.fullmatch(r'[\d,]+\.?\d*', s):
                k += 1; continue
            kw = re.match(r'([A-Za-z][A-Za-z .]*[A-Za-z])', s)
            if kw:
                keyword = kw.group(1).strip()
            break
        if keyword:
            mid = (r'\s*\n\s*' + AMT) * k
            yield 3, r'(' + AMT + r')' + mid + r'\s*\n\s*' + re.escape(keyword) + r'\b'


def build_fallback_specs(field, value, datatype, raw_text):
    """Yield candidate specs built from the value's context, best first.

    For a TAX-AMOUNT field (CGST/SGST/IGST/CESS) the regex MUST contain that tax's
    own name — a purely positional anchor is rejected, because on a sibling that
    carries a DIFFERENT tax it would match the wrong amount (the 'IGST grabbed the
    total on a CGST/SGST invoice' failure). The tax name is the only anchor that
    makes a tax regex fire on the right invoices and no others."""
    variants = _printed_variants(value, datatype)
    token = _token_for(value, datatype)
    lines = raw_text.split("\n")
    hint = _field_hint(field)

    idxs = [i for i, ln in enumerate(lines)
            if ln.strip() in variants or any(v in ln for v in variants)]
    if not idxs:
        return

    seen, ranked = set(), []
    for idx in idxs:
        for prio, rx in _candidates_at(field, idx, lines, variants, token, datatype):
            if rx in seen:
                continue
            if hint and hint not in rx.upper():   # tax field: require its own tax word
                continue
            seen.add(rx)
            hit = 0 if (hint and hint in rx.upper()) else 1
            ranked.append(((hit, prio), {"name": field, "datatype": datatype, "regex": [rx]}))

    for _, spec in sorted(ranked, key=lambda t: t[0]):
        yield spec


# ---------- GST tax-presence (general; uses GST tax NAMES, not vendor labels) ----------
# CGST/SGST/IGST/CESS and taxable/assessable value are the tax categories defined by
# GST law itself — domain constants, not per-vendor patterns. Used to drive the tax
# UNION (a vendor may be IGST on some invoices, CGST/SGST on others) and to let a tax
# regex skip siblings that don't carry that tax.
_TAX_KW = {
    "total_invoice_taxable_value":  r'(?:Taxable|Assessable|Asse\.?)\s*Value',
    "invoice_total_basic_value":    r'(?:Basic|Total\s+Sales)\s*(?:Value|Amount)',
    "total_invoice_cgst_tax_value": r'(?<![A-Za-z])CGST',
    "total_invoice_sgst_tax_value": r'(?<![A-Za-z])(?:SGST|UTGST)',
    "total_invoice_igst_tax_value": r'(?<![A-Za-z])IGST',
    "total_invoice_cess_tax_value": r'(?<![A-Za-z])(?:CESS|TCS)',
}


def build_tax_specs(field, value, text):
    """Yield TAX-NAME-ANCHORED regex candidates for a tax amount, best first. Each
    CONTAINS the tax word as its anchor (so it can NEVER fire on an invoice lacking
    that tax -> no cross-tax misfire), then reaches the amount with increasing
    distance tolerance: same line, next line, past a rate/label, or a few lines /
    intervening amounts down. Safe generalization of the summary-box handling."""
    kw = _TAX_KW.get(field)
    if not kw:
        return
    for gap in (r'\s*',                                        # CGST 27960.64 / CGST\n27960.64
                r'[^\n]*\n\s*',                                # CGST Value 9.00 %\n1,274.41
                r'[^\n]*\n(?:\s*' + AMT + r'\s*\n)*\s*',       # header, rows of amounts, target
                r'(?:[^\n]*\n){0,3}\s*'):                      # tax word within next few lines
        yield {"name": field, "datatype": "amount", "regex": [kw + gap + r'(' + AMT + r')']}


def find_tax_value(field, text):
    """Return a tax field's value if the tax is actually present (non-nil) on `text`,
    else None. From the tax NAME, the amount must be reachable through only LABEL-ISH
    gap tokens (whitespace, ':', '-', a 'rate %', or words like Value/Amount/Payable).
    A stray word breaks the gap — so 'TCS ON SCRAP' followed by an unrelated amount
    does NOT match CESS (the 'ON SCRAP' words break it), and a bare 'IGST' column
    header cannot reach across other text to steal a neighbouring value. Guards:
      - a rate ('9.00 %') is never taken as the value;
      - if a DIFFERENT tax word sits between this tax and the amount, skip it;
      - a ZERO/NIL amount counts as absent (e.g. IGST 0.00 intra-state)."""
    kw = _TAX_KW.get(field)
    if not kw:
        return None
    others = [p for f, p in _TAX_KW.items()
              if f != field and f in ("total_invoice_cgst_tax_value",
                                      "total_invoice_sgst_tax_value",
                                      "total_invoice_igst_tax_value",
                                      "total_invoice_cess_tax_value")]
    other_pat = re.compile("|".join(others), re.I) if others else None
    amt = re.compile(r'([\d,]+\.\d{2})(?!\s*%)')
    for m in re.finditer(kw, text, re.I):
        g = _TAX_GAP.match(text, m.end())
        pos = g.end() if g else m.end()
        am = amt.match(text, pos)
        if not am:
            continue
        if other_pat and other_pat.search(text[m.end():pos]):
            continue                      # the amount belongs to a different tax
        val = am.group(1)
        if field not in ("total_invoice_taxable_value", "invoice_total_basic_value"):
            try:
                if float(val.replace(",", "")) == 0:
                    continue
            except ValueError:
                pass
        return val
    return None


# label-ish gap allowed between a tax word and its amount (a stray word breaks it,
# so 'TCS ON SCRAP' cannot reach an unrelated amount)
_TAX_GAP = re.compile(r'(?:[\s:.\-#]|[\d.,]*\d\s*%|Value|Amount|Payable|Payble|Total|Rs\.?|Tax)*', re.I)


def tax_present_in(field, text):
    """True if this tax field is present (non-nil) on `text`."""
    return find_tax_value(field, text) is not None


# ---------- labels-block-then-values-block summary (Toyota-Boshoku style) ----------
# Some invoices print the totals as a BLOCK of labels followed by a BLOCK of amounts:
#   TOTAL BASIC VALUE / TOTAL CGST PAYABLE / TOTAL SGST PAYABLE / TOTAL IGST PAYABLE / TCS ON SCRAP
#   1,569.20 / 192.70 / 192.70 / 0.00 / 0.00
# Here a tax amount has NO adjacent tax word, so build_tax_specs cannot anchor it.
# This parser aligns the label block with the value block by POSITION, then emits a
# regex anchored on the LAST label (which sits directly before the values) that skips
# to the target field's amount. Fully general — the anchor text is taken from THIS
# invoice, not hardcoded — and every candidate is still validated by the tester.
_TAX_LABEL_KW = {
    "invoice_total_basic_value":    r'BASIC',
    "total_invoice_taxable_value":  r'TAXABLE|ASSESS|ASSE',
    "total_invoice_cgst_tax_value": r'CGST',
    "total_invoice_sgst_tax_value": r'SGST|UTGST',
    "total_invoice_igst_tax_value": r'IGST',
    "total_invoice_cess_tax_value": r'CESS|TCS',
}

# a line that looks like a totals label (used to bound the label block)
_LABELISH = re.compile(r'TOTAL|VALUE|PAYABLE|CGST|SGST|UTGST|IGST|CESS|TCS|BASIC|'
                       r'TAXABLE|ASSESS|ASSE|AMOUNT|SCRAP|GST', re.I)


def build_taxbox_specs(field, value, text):
    """Yield a positional regex for a tax amount in a labels-block/values-block
    summary, anchored on the last label before the value block. Best-effort: only
    yields when the label/value alignment reproduces `value`."""
    kw = _TAX_LABEL_KW.get(field)
    if not kw:
        return
    want = str(value).replace(",", "")
    lines = [ln.strip() for ln in text.split("\n")]
    n = len(lines)
    is_amt = lambda s: re.fullmatch(r'[\d,]+\.\d{2}', s) is not None
    i = 0
    while i < n:
        if not is_amt(lines[i]):
            i += 1; continue
        # a run of amount lines (the value block)
        j = i
        vals = []
        while j < n and (is_amt(lines[j]) or lines[j] == ""):
            if is_amt(lines[j]):
                vals.append(j)
            j += 1
        # the label block: consecutive TOTAL-LABEL lines directly above i. Only lines
        # that look like a totals label count (so a stray 'IRN NO.:' or address line
        # above the box does not shift the positions).
        k = i - 1
        labels = []
        while k >= 0 and lines[k] and not is_amt(lines[k]) and _LABELISH.search(lines[k]):
            labels.append(k); k -= 1
        labels = labels[::-1]
        if len(labels) >= 2 and len(vals) >= 2:
            # which label matches THIS field's tax word, and its position in the block
            pos = next((p for p, li in enumerate(labels) if re.search(kw, lines[li], re.I)), None)
            if pos is not None and pos < len(vals):
                if lines[vals[pos]].replace(",", "") == want:
                    anchor = re.escape(lines[labels[-1]])   # last label, right before values
                    A = r'[\d,]+\.\d{2}'
                    yield {"name": field, "datatype": "amount",
                           "regex": [anchor + r'\s*\n(?:\s*' + A + r'\s*\n){' + str(pos) + r'}\s*(' + A + r')']}
                    return
        i = j if j > i else i + 1


# ---------- document_type (a fixed title, resolved deterministically) ----------
# The document title ("TAX INVOICE", "CREDIT NOTE", ...) is STABLE TEMPLATE TEXT —
# identical on every invoice of the template — so anchoring the regex on the title
# itself is correct, not overfit. (The overfit guard, which rejects a regex that
# copies the value, must be BYPASSED for this one field, since here the value IS a
# stable label.) We emit a regex that captures exactly the known title.
_DOC_TITLES = ["CREDIT NOTE", "DEBIT NOTE", "TAX INVOICE", "GST INVOICE",
               "BILL OF SUPPLY", "BILL OF SALE", "INVOICE"]


def build_doctype_specs(value, text):
    """Yield a stable regex that captures the document title. Prefers the known
    title the discovered value maps to; else captures the discovered phrase."""
    v = (str(value) or "").strip().upper()
    chosen = next((t for t in _DOC_TITLES if t in v), None)
    phrase = chosen or v
    if not phrase:
        return
    pat = r'(' + r'\s+'.join(re.escape(w) for w in phrase.split()) + r')'
    yield {"name": "document_type", "datatype": "string", "regex": [pat]}
