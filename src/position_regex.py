"""
position_regex.py  —  build candidate regexes from WHERE the value sits.

Why: discovery (the LLM) reliably finds each field's value on every invoice of a
template, but the LLM often cannot turn that into a working regex — it miscounts
lines in a values block, writes a 16-char GSTIN pattern, or cannot reach an amount
printed far below its label. Those are mechanical tasks, so code does them:

  1. locate the value on EVERY invoice of the template (primary + siblings whose
     values were discovered), in whatever form it is printed (48203 / 48,203.00 /
     20,80,494.68, 05-JAN-26 / 05/01/2026, ...);
  2. find an ANCHOR: a line whose text is IDENTICAL on every invoice and sits at the
     SAME line distance from the value (above or below), or a label printed on the
     value's own line;
  3. emit   anchor -> skip N lines -> (generic token)   — the token is built from
     the value's SHAPE (amount / GSTIN / date / identifier class), never from its
     characters.

This is general (no vendor rules). Every candidate is then validated exactly like
the LLM's: it must extract the correct value on every invoice, must not be overfit,
must be safe to run, and a tax amount must carry its tax's label word.
"""
import re
from src.field_tester import clean_amount, parse_date, date_variants, _indian_group

AMT_TOKEN = r"\d[\d,]*(?:\.\d+)?"
GSTIN_TOKEN = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]"
MAX_DIST = 15                 # how many lines above/below to look for an anchor
MAX_CANDIDATES = 6
MAX_KEEP = 3                  # working regexes kept per field (primary + fallbacks)

_NEGATIVE_LABEL = re.compile(
    r"(bill\s*to|ship\s*to|detail\s*of\s*receiver|detail\s*of\s*consignee|buyer|consignee)",
    re.I,
)
_POSITIVE_LABEL = re.compile(r"(gstin|gstn\s*no|gst\s*no|supplier|seller)", re.I)


def gstin_from_text(raw_text: str) -> str | None:
    """Fallback GST-number detection when no QR code is available. Finds every
    GSTIN-shaped candidate, and for each one finds the NEAREST preceding label
    (positive or negative) anywhere earlier in the text — not a fixed window —
    so a label far from its GSTIN (e.g. across address lines) still counts."""
    text = raw_text or ""
    candidates = list(re.finditer(GSTIN_TOKEN, text))
    if not candidates:
        return None

    neg_positions = [m.start() for m in _NEGATIVE_LABEL.finditer(text)]
    pos_positions = [m.start() for m in _POSITIVE_LABEL.finditer(text)]

    def nearest_before(label_positions, idx):
        before = [p for p in label_positions if p < idx]
        return max(before) if before else None   # the CLOSEST preceding label

    scored = []
    for m in candidates:
        idx = m.start()
        nearest_neg = nearest_before(neg_positions, idx)
        nearest_pos = nearest_before(pos_positions, idx)
        # Reject only if the CLOSEST preceding label is a negative one
        # (i.e. this GSTIN is textually "inside" a buyer/consignee block).
        if nearest_neg is not None and (nearest_pos is None or nearest_neg > nearest_pos):
            continue
        # Prefer candidates with a positive label closer to them, then earlier
        # position in the document (seller block is almost always first).
        dist = (idx - nearest_pos) if nearest_pos is not None else float("inf")
        scored.append((dist, idx, m.group(0)))

    if not scored:
        return None
    scored.sort()   # smallest label-distance wins; ties broken by earliest position
    return scored[0][2]

_TAX_WORDS = {                # tax label words to carry in a tax regex (specific first)
    "total_invoice_cgst_tax_value": ("CGST", "CENTRAL"),
    "total_invoice_sgst_tax_value": ("SGST", "UTGST", "STATE"),
    "total_invoice_igst_tax_value": ("IGST", "INTEGRATED"),
    "total_invoice_cess_tax_value": ("CESS", "TCS"),
}


# ---------- where is the value printed? ----------
def _amount_forms(v):
    s = str(v).strip()
    a = clean_amount(s)
    if a is None:
        return {s}
    forms = {s, f"{a:.2f}", f"{a:,.2f}", _indian_group(a)}
    if a == int(a):
        forms |= {str(int(a)), f"{int(a):,}"}
    else:
        forms.add(f"{a:.2f}".rstrip("0"))            # 257.40 -> 257.4
    return {f for f in forms if f}


def _occurrences(value, datatype, name, text):
    """[(line_index, start_col, end_col, printed_text)] of the value in `text`."""
    lines = text.split("\n")
    s = str(value).strip()
    if datatype == "amount":
        forms = sorted(_amount_forms(s), key=len, reverse=True)
        pat = re.compile(r"(?<![\d.,])(?:" + "|".join(map(re.escape, forms)) + r")(?![\d]|\.\d)")
    elif datatype == "date":
        forms = sorted(set(date_variants(s)) | {s}, key=len, reverse=True)
        pat = re.compile(r"(?<![0-9A-Za-z])(?:" + "|".join(map(re.escape, forms)) + r")(?![0-9A-Za-z])", re.I)
    else:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])")
    out = []
    for i, ln in enumerate(lines):
        for m in pat.finditer(ln):
            out.append((i, m.start(), m.end(), m.group(0)))
    return out


# ---------- the capture token, from the value's SHAPE ----------
def _date_shape(p):
    parts = []
    for run in re.findall(r"\d+|[A-Za-z]+|[^\dA-Za-z]", p):
        if run.isdigit():
            parts.append(r"\d{1,2}" if len(run) <= 2 else rf"\d{{{len(run)}}}")
        elif run.isalpha():
            parts.append(r"[A-Za-z]{3}" if len(run) == 3 else r"[A-Za-z]+")
        elif run.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(run))
    return "".join(parts)


def _token(datatype, name, printed_forms):
    if datatype == "amount":
        return AMT_TOKEN
    if datatype == "gstin":
        return GSTIN_TOKEN
    if datatype == "date":
        shapes = sorted({_date_shape(p) for p in printed_forms})
        return shapes[0] if len(shapes) == 1 else "(?:" + "|".join(shapes) + ")"
    # identifiers: 64-hex IRN, or a character class covering every printed value
    if all(re.fullmatch(r"[0-9A-Fa-f]{64}", p) for p in printed_forms):
        return r"[0-9A-Fa-f]{64}"
    words = {len(p.split()) for p in printed_forms}
    if len(words) != 1:
        return None                                   # word count varies -> no safe shape
    chars = "".join(printed_forms)
    cls = ""
    if re.search(r"[A-Z]", chars): cls += "A-Z"
    if re.search(r"[a-z]", chars): cls += "a-z"
    if re.search(r"\d", chars):    cls += "0-9"
    for c in sorted(set(re.sub(r"[A-Za-z0-9\s]", "", chars))):
        cls += re.escape(c)
    if not cls:
        return None
    return r"[ \t]+".join([f"[{cls}]+"] * words.pop())


# ---------- literal text -> pattern (spaces flexible) ----------
def _lit(s):
    return r"[ \t]+".join(re.escape(w) for w in s.split())


_AMOUNT_WORDS = re.compile(r"\b(thousand|hundred|lakh|lakhs|crore|rupees|paise|only)\b", re.I)


def _labelish(s, strict):
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    if letters < 2 or len(s) > 80:
        return False
    if _AMOUNT_WORDS.search(s):        # amount-in-words line: changes on every invoice
        return False
    if strict and digits > 3:          # single invoice: avoid dates/numbers as anchors
        return False
    return True


def build_candidates(name, datatype, members):
    """members: [(text, value), ...] — PRIMARY FIRST, then siblings whose value for
    this field is known. Returns candidate regexes, best first."""
    if datatype not in ("amount", "gstin", "date", "string") or name == "document_type":
        return []
    occ = [_occurrences(v, datatype, name, t) for t, v in members]
    if not occ or not occ[0] or any(not o for o in occ):
        return []                                  # value not located on some invoice
    lines = [t.split("\n") for t, _ in members]
    strict = len(members) == 1
    printed = {o[3] for oc in occ for o in oc}
    tok = _token(datatype, name, printed)
    if not tok:
        return []

    def same_line_prefix(k, o):
        return lines[k][o[0]][:o[1]].strip()

    scored = []
    for o0 in occ[0]:
        li0 = o0[0]
        # ---- value-line prefix: a stable label, a fixed number of tokens, or nothing
        p0 = same_line_prefix(0, o0)
        pref_label = None
        pref_generic = None
        matches_p = [[o for o in occ[k] if same_line_prefix(k, o) == p0] for k in range(len(members))]
        if p0 and all(matches_p) and _labelish(p0, strict):
            pref_label = r"[ \t]*" + _lit(p0) + r"[ \t]*"
        ntok = len(p0.split()) if p0 else 0
        if ntok:
            pref_generic = r"[ \t]*(?:\S+[ \t]+){%d}" % ntok
        else:
            pref_generic = r"[ \t]*"
        if pref_label:
            scored.append((0, 0, r"\n" + pref_label + f"({tok})"))
        # ---- anchors above / below at the same distance on every invoice
        for d in range(1, MAX_DIST + 1):
            for direction in ("above", "below"):
                ai = li0 - d if direction == "above" else li0 + d
                if not (0 <= ai < len(lines[0])):
                    continue
                a = lines[0][ai].strip()
                if not _labelish(a, strict):
                    continue
                ok, unique = True, True
                for k in range(1, len(members)):
                    hit = False
                    for o in occ[k]:
                        j = o[0] - d if direction == "above" else o[0] + d
                        if 0 <= j < len(lines[k]) and lines[k][j].strip() == a:
                            if len(p0.split()) == len(same_line_prefix(k, o).split()):
                                hit = True; break
                    if not hit:
                        ok = False; break
                if not ok:
                    continue
                for L in lines:
                    if sum(1 for x in L if x.strip() == a) != 1:
                        unique = False
                pref = pref_label or pref_generic
                skip = r"(?:[^\n]*\n){%d}" % (d - 1) if d > 1 else ""
                if direction == "above":
                    rx = r"\n[ \t]*" + _lit(a) + r"[ \t]*\n" + skip + pref + f"({tok})"
                else:
                    rx = (r"\n" + pref + f"({tok})" + r"[^\n]*\n" + skip
                          + r"[ \t]*" + _lit(a) + r"[ \t]*(?:\n|$)")
                score = ((0 if unique else 50) + d + (0 if direction == "above" else 1)
                         + (0 if pref_label else 2) - (2 if ":" in a else 0))
                scored.append((score, d, rx))

    # a TAX amount must carry its tax's label word: put the first one found BEFORE
    # the value in front (lazy skip), or else after it
    words = _TAX_WORDS.get(name)
    out, seen = [], set()
    for _, _, rx in sorted(scored, key=lambda x: (x[0], x[1])):
        if words:
            t0 = members[0][0]
            pos = sum(len(x) + 1 for x in lines[0][:occ[0][0][0]])
            before = [w for w in words if re.search(re.escape(w), t0[:pos], re.I)]
            after = [w for w in words if re.search(re.escape(w), t0[pos:], re.I)]
            if before:
                rx = re.escape(before[0]) + r".*?" + rx
            elif after:
                rx = rx + r".*?" + re.escape(after[0])
            else:
                continue
        if rx not in seen:
            seen.add(rx); out.append(rx)
        if len(out) >= MAX_CANDIDATES:
            break
    return out
