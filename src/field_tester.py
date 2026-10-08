"""
field_tester.py  —  THE CORE of the config agent.

Validates ONE candidate field spec against invoice raw text. CRITICAL: the
extraction logic below is a FAITHFUL PORT of the production pipeline
(header_extraction.py extract_field) — same strategy order (alias-first for gstin,
regex-first otherwise), same alias line scan (right/below/above), same cleaners and
'looks-like' validators, same negative_alias handling. This is what makes the
agent's PASS/FAIL predict production: a config only passes here if it will extract
the same value in your real pipeline. If the two ever drift, configs marked
"solved" fail in production (wrong GSTIN, empty totals) — so keep this in sync with
header_extraction.py.
"""
import re, os
from datetime import datetime
from typing import Optional, Any

# ---------- SAFE execution of LLM-written regexes ----------
# An LLM regex with nested quantifiers, e.g. (?:\s*[\d,]+\s*)+X, backtracks
# catastrophically: one re.search on a few KB can run for hours, and because
# Python's `re` holds the GIL while matching, it FREEZES EVERY worker thread and the
# main thread with it. Two layers of protection:
#   1. static: reject patterns with an unbounded repeat nested in an unbounded repeat
#      (also protects production, whose `re` has no timeout);
#   2. dynamic: run patterns with the `regex` module (same syntax/semantics as `re`
#      for these patterns) under a timeout; a timeout rejects the pattern.
REGEX_TIMEOUT = float(os.environ.get("REGEX_TIMEOUT", "2.0"))   # seconds per search


class RegexTimeout(Exception):
    """An LLM regex took too long (catastrophic backtracking) — reject it."""


def regex_is_dangerous(pat: str) -> bool:
    """True if `pat` has an UNBOUNDED repeat (+, *, {n,}) that contains another
    unbounded repeat — the classic exponential-backtracking shape."""
    try:
        import re._parser as sp          # Python 3.11+
        import re._constants as sc
    except ImportError:                  # older Pythons
        import sre_parse as sp
        import sre_constants as sc
    try:
        tree = sp.parse(pat)
    except Exception:
        return False                      # invalid patterns are rejected elsewhere
    REPEATS = {sc.MAX_REPEAT, sc.MIN_REPEAT}
    if hasattr(sc, "POSSESSIVE_REPEAT"):
        REPEATS.add(sc.POSSESSIVE_REPEAT)
    UNBOUNDED = sc.MAXREPEAT

    def children(op, av):
        if op in REPEATS:
            return [av[2]]
        if op == sc.SUBPATTERN:
            return [av[-1]]
        if op == sc.BRANCH:
            return list(av[1])
        if op in (sc.ASSERT, sc.ASSERT_NOT):
            return [av[1]]
        if hasattr(sc, "ATOMIC_GROUP") and op == sc.ATOMIC_GROUP:
            return [av]
        return []

    def has_unbounded(sub):
        for op, av in sub:
            if op in REPEATS and av[1] == UNBOUNDED:
                return True
            if any(has_unbounded(c) for c in children(op, av)):
                return True
        return False

    def walk(sub):
        for op, av in sub:
            if op in REPEATS and av[1] == UNBOUNDED and has_unbounded(av[2]):
                return True
            if any(walk(c) for c in children(op, av)):
                return True
        return False

    return walk(tree)


# The check must use the SAME engine production uses (plain `re`): the `regex`
# module optimises some patterns that still hang `re`, e.g. (?:\d|\d\d)+X. A stuck
# `re` search cannot be interrupted inside a process (it holds the GIL), so each
# worker thread gets a tiny helper PROCESS (plain `python -c`, no project imports,
# starts in milliseconds, same on Windows) that runs the searches. If one exceeds
# REGEX_TIMEOUT the helper is killed and restarted and the pattern is rejected.
import sys as _sys
import json as _json
import queue as _queue
import struct as _struct
import subprocess as _sp
import threading as _th

_CHILD = r"""
import sys, re, json, struct
inp, out = sys.stdin.buffer, sys.stdout.buffer
def send(o):
    b = json.dumps(o).encode(); out.write(struct.pack('>I', len(b)) + b); out.flush()
send("ready")
while True:
    h = inp.read(4)
    if len(h) < 4: break
    pat, text, flags = json.loads(inp.read(struct.unpack('>I', h)[0]))
    try:
        m = re.search(pat, text, flags)
        if m is None: send(["none", None])
        else:
            n = m.re.groups
            send(["ok", {"lastindex": m.lastindex, "groups": [m.group(i) for i in range(n + 1)],
                         "spans": [list(m.span(i)) for i in range(n + 1)]}])
    except re.error as e:
        send(["error", str(e)])
"""


class _Match:
    """The parts of re.Match the extractor uses."""
    def __init__(self, d):
        self._d, self.lastindex = d, d["lastindex"]

    def group(self, i=0):
        return self._d["groups"][i]

    def groups(self):
        return tuple(self._d["groups"][1:])

    def span(self, i=0):
        return tuple(self._d["spans"][i])


class _ReWorker:
    def __init__(self):
        self._start()

    def _start(self):
        flags = getattr(_sp, "CREATE_NO_WINDOW", 0)          # no console flash on Windows
        self.proc = _sp.Popen([_sys.executable, "-c", _CHILD], stdin=_sp.PIPE,
                              stdout=_sp.PIPE, stderr=_sp.DEVNULL, creationflags=flags)
        self.q = _queue.Queue()
        _th.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        if self._get(30) != "ready":
            raise RuntimeError("regex helper process did not start")

    @staticmethod
    def _reader(proc, q):
        f = proc.stdout
        while True:
            h = f.read(4)
            if len(h) < 4:
                q.put(("__dead__", None)); return
            q.put(_json.loads(f.read(_struct.unpack(">I", h)[0])))

    def _get(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except _queue.Empty:
            return None

    def search(self, pat, text, flags, timeout):
        b = _json.dumps([pat, text, int(flags)]).encode()
        self.proc.stdin.write(_struct.pack(">I", len(b)) + b)
        self.proc.stdin.flush()
        msg = self._get(timeout)
        if msg is None:                                     # too slow -> kill + restart
            self.proc.kill()
            try:
                self.proc.wait(5)
            except Exception:
                pass
            self._start()
            raise RegexTimeout(pat)
        kind, payload = msg
        if kind == "none":
            return None
        if kind == "error":
            raise re.error(payload)
        if kind == "__dead__":
            raise RuntimeError("regex helper died")
        return _Match(payload)


_local = _th.local()
_HELPER_OK = [True]


def _helper():
    w = getattr(_local, "w", None)
    if w is None or w.proc.poll() is not None:
        w = _local.w = _ReWorker()
    return w


def safe_search(pat: str, text: str, flags=0):
    """re.search for an UNTRUSTED (LLM) pattern, run by production's own engine
    (`re`) under a hard time limit. Raises RegexTimeout instead of hanging, re.error
    if the pattern is invalid."""
    if regex_is_dangerous(pat):
        raise RegexTimeout(pat)
    re.compile(pat, flags)                        # invalid -> re.error, no round-trip
    if _HELPER_OK[0]:
        try:
            return _helper().search(pat, text, flags, REGEX_TIMEOUT)
        except (RegexTimeout, re.error):
            raise
        except Exception:                         # helper unavailable -> in-process
            _HELPER_OK[0] = False
    return re.search(pat, text, flags)


# ---------- datatype cleaners (faithful port of header_extraction.py) ----------
def clean_amount(raw):
    if not raw:
        return None
    s = re.sub(r'[₹$€£\s]', '', str(raw).strip())
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg, s = True, s[1:-1]
    if s.startswith('-'):
        neg, s = True, s[1:]
    s = s.replace(',', '')
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None

def clean_date(raw):
    if not raw:
        return None
    s = re.sub(r'\s+', '', str(raw).strip())
    s = re.sub(r'\d{1,2}:\d{2}(:\d{2})?(AM|PM|am|pm)?', '', s).strip()
    for fmt in ["%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
                "%d%b%Y", "%d%B%Y", "%d.%m.%y", "%d-%m-%y", "%d/%m/%y"]:
        try:
            return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return s or None

def clean_gstin(raw):
    if not raw:
        return None
    m = re.search(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]', str(raw).upper())
    return m.group(0) if m else str(raw).strip()

def clean_string(raw):
    if not raw:
        return None
    s = str(raw).strip().strip(':').strip()
    return s or None

def clean_boolean(raw):
    if not raw:
        return "No"
    return "Yes" if str(raw).strip().lower() in ("yes", "true", "y", "1", "applicable") else "No"

def clean_percentage(raw):
    if not raw:
        return None
    try:
        return float(str(raw).strip().replace('%', '').replace(',', '').strip())
    except ValueError:
        return None

CLEANERS = {
    "string": clean_string, "amount": clean_amount, "date": clean_date,
    "gstin": clean_gstin, "boolean": clean_boolean, "percentage": clean_percentage,
}

VALIDATORS = {
    "date":       re.compile(r'\d{1,4}[\.\-/]\d{1,2}[\.\-/]\d{1,4}'),
    "amount":     re.compile(r'^[\-₹$€£\s\(]?[\d,]+\.?\d*\)?$'),
    "gstin":      re.compile(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]', re.I),
    "percentage": re.compile(r'^\d+\.?\d*%?$'),
}

def _looks_like(value, datatype):
    v = VALIDATORS.get(datatype)
    return bool(v.search(value)) if v else True


# ---------- alias-based extraction (faithful port of header_extraction.py) ----------
def _build_alias_pattern(alias):
    e = re.escape(alias)
    return rf'(?:\*\*|__)?{e}\.?(?:\s*\([^)]*\))?(?:\*\*|__)?\s*[:\-|]?\s*'

def _lines(text):
    return text.split('\n')

def _extract_right(line, alias):
    m = re.search(_build_alias_pattern(alias), line, re.I)
    if not m:
        return None
    rest = line[m.end():].strip()
    rest = re.sub(r'\s*\|.*$', '', rest).strip()
    nxt = re.search(r'\s{2,}\S+[\.\s]*[:\-]\s', rest)
    if nxt:
        rest = rest[:nxt.start()].strip()
    rest = rest.strip('*').strip('_').strip()
    return rest or None

def _extract_below(lines, idx, datatype):
    for i in range(idx + 1, min(idx + 4, len(lines))):
        c = lines[i].strip()
        if not c or c.startswith('#') or c == '---':
            continue
        c = c.strip('|').strip('*').strip('_').strip(':').strip()
        if c and _looks_like(c, datatype):
            return c
    return None

def _extract_above(lines, idx, datatype):
    _date = re.compile(r'^\d{1,4}[\.\-/]\d{1,2}[\.\-/]\d{1,4}$')
    _label = re.compile(r'.+\s*:\s*.+')
    for i in range(idx - 1, max(idx - 4, -1), -1):
        c = lines[i].strip()
        if not c or c.startswith('#') or c == '---':
            continue
        c = c.strip('|').strip('*').strip('_').strip()
        if not c:
            continue
        if datatype != "date" and _date.match(c):
            continue
        if datatype == "string" and _label.match(c):
            continue
        if _looks_like(c, datatype):
            return c
    return None

def _is_negative(line, neg_aliases):
    return any(re.search(re.escape(n), line, re.I) for n in (neg_aliases or []))


# ---------- the extractor: run ONE field spec against raw text (production port) ----------
def extract_field(spec: dict, raw_text: str):
    """Faithful port of header_extraction.extract_field. Returns the CLEANED value
    (float/str) or None. Strategy order matches production: gstin -> alias then
    regex; everything else -> regex then alias."""
    aliases   = spec.get("alias", []) or []
    neg_alias = spec.get("negative_alias", []) or []
    direction = spec.get("direction", "right")
    datatype  = spec.get("datatype", "string")
    regexes   = spec.get("regex", []) or []
    lines = _lines(raw_text)

    order = ["alias", "regex"] if datatype == "gstin" else ["regex", "alias"]
    for strategy in order:
        if strategy == "regex":
            for rgx in regexes:
                if not rgx:
                    continue
                try:
                    m = safe_search(rgx, raw_text, re.I | re.DOTALL)
                except re.error:
                    continue              # (RegexTimeout propagates -> pattern rejected)
                if m:
                    raw = m.group(1) if m.lastindex else m.group(0)
                    val = CLEANERS.get(datatype, clean_string)(raw)
                    if val is not None:
                        return val
        elif strategy == "alias":
            best, best_pri = None, -1
            for i, line in enumerate(lines):
                for ai, alias in enumerate(aliases):
                    if not re.search(re.escape(alias), line, re.I):
                        continue
                    if neg_alias and _is_negative(line, neg_alias):
                        continue
                    raw = None
                    if direction == "right":
                        raw = _extract_right(line, alias)
                        if not raw:
                            raw = _extract_above(lines, i, datatype)
                        if not raw and datatype in ("date", "amount"):
                            raw = _extract_below(lines, i, datatype)
                    elif direction == "bottom":
                        raw = _extract_below(lines, i, datatype)
                        if not raw:
                            raw = _extract_right(line, alias)
                        if not raw:
                            raw = _extract_above(lines, i, datatype)
                    elif direction == "above":
                        raw = _extract_above(lines, i, datatype)
                        if not raw:
                            raw = _extract_right(line, alias)
                    else:
                        raw = _extract_right(line, alias)
                    if not raw:
                        continue
                    val = CLEANERS.get(datatype, clean_string)(raw)
                    if val is None:
                        continue
                    if datatype == "string" and isinstance(val, str):
                        if len(val) < 2 or not re.search(r'[a-zA-Z0-9]', val):
                            continue
                    pri = len(aliases) - ai
                    if pri > best_pri:
                        best, best_pri = val, pri
            if best is not None:
                return best
    return None


def clean_value(raw, datatype):
    return CLEANERS.get(datatype, clean_string)(raw) if raw is not None else None


# ---------- the TESTER: does this spec extract the trusted value in PRODUCTION? ----------
def same_value(name: str, datatype: str, got, expected, *, tol: float = 0.01) -> bool:
    """Is the EXTRACTED (cleaned) value the same as the trusted value? Shared by the
    validator (test_field) and validate.py so both judge correctness identically.
    NOTE: acceptance logic, not extraction — extract_field mirrors production exactly.
    document_type / irn_number legitimately differ only by whitespace/case between the
    printed text and the canonical value ("Tax\nINVOICE" == "TAX INVOICE"; a 64-hex
    IRN wrapped over two lines)."""
    if datatype == "amount":
        exp = clean_amount(str(expected))
        return isinstance(got, (int, float)) and exp is not None and abs(got - exp) <= tol
    if datatype == "date":
        pg, pe = parse_date(got), parse_date(clean_date(str(expected)))
        if pg is not None and pe is not None:
            return pg == pe
        return (got is not None) and (str(got).strip() == str(clean_date(str(expected))).strip())
    if datatype == "gstin":
        exp = clean_gstin(str(expected))
        return (got is not None) and (str(got).strip().upper() == str(exp).strip().upper())
    g = "" if got is None else str(got)
    e = str(expected)
    if name == "document_type":
        _n = lambda x: re.sub(r'\s+', ' ', x).strip().upper()
        return bool(g.strip()) and _n(g) == _n(e)
    if name == "irn_number":
        _n = lambda x: re.sub(r'\s+', '', x).lower()
        return bool(g.strip()) and _n(g) == _n(e)
    return (got is not None) and (g.strip() == e.strip())


def test_field(spec: dict, raw_text: str, expected: Any, *, tol: float = 0.01) -> dict:
    """Run the spec through the production-faithful extractor and compare to the
    trusted value. PASS here means it will extract the same value in your pipeline."""
    datatype = spec.get("datatype", "string")
    try:
        got = extract_field(spec, raw_text)      # already cleaned
    except RegexTimeout:
        return {"field": spec.get("name", "?"), "passed": False, "expected": expected,
                "got": None, "status": "TIMEOUT", "timeout": True}
    passed = same_value(spec.get("name", ""), datatype, got, expected, tol=tol)
    return {
        "field": spec.get("name", "?"),
        "passed": passed,
        "expected": expected,
        "got": got,
        "status": "PASS" if passed else "FAIL",
    }

# ---------- locate where a trusted value appears (for LLM context) ----------
def value_context(raw_text: str, expected, ctx: int = 1, max_snips: int = 4):
    """
    Return raw-text snippets (with +/- ctx lines) where `expected` appears.
    Handles float formatting: 70837.23 also matches '70,837.23' and '70837.23'.
    Lets the LLM anchor a regex on the surrounding LABEL, not guess blindly.
    """
    variants = set()
    s = str(expected).strip()
    variants.add(s)
    for dv in date_variants(expected):     # date may appear in another format
        variants.add(dv)
    # AMOUNTS: the value may be printed plain (2141.20), US-grouped (2,141.20) or
    # Indian-grouped (20,80,494.68) — search for all forms, whatever form the
    # discovered value came in, so the LLM always gets the location context.
    amt = clean_amount(s) if re.fullmatch(r'[\d,]+(\.\d+)?', s) else None
    if amt is not None:
        if s.endswith(".0"):
            variants.add(s[:-2])
        variants.add(f"{amt:.2f}")
        variants.add(f"{amt:,.2f}")
        variants.add(_indian_group(amt))
    lines = raw_text.splitlines()
    snips, seen = [], set()
    for i, ln in enumerate(lines):
        if any(v and v in ln for v in variants):
            lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
            block = "\n".join(lines[lo:hi])
            if block not in seen:
                seen.add(block)
                snips.append(block)
            if len(snips) >= max_snips:
                break
    return snips


def _indian_group(x: float) -> str:
    """2080494.68 -> '20,80,494.68' (lakh/crore grouping)."""
    neg = x < 0
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + "." + frac


# ---------- match span of a spec (for position-distinctness checks) ----------
def match_span(spec: dict, text: str):
    """Return (start,end) of the captured group for a spec's first working regex."""
    for pat in spec.get("regex", []) or []:
        try:
            m = safe_search(pat, text, re.I | re.DOTALL)
        except (re.error, RegexTimeout):
            continue
        if m:
            return m.span(1) if m.groups() else m.span(0)
    return None


def matches(spec: dict, text: str) -> bool:
    """True if the spec's regex/alias extracts ANY value from `text`.
    Used for GENERALIZATION: a regex must still match on sibling invoices of the
    same template (a part-name-anchored regex will fail here)."""
    try:
        return extract_field(spec, text) is not None
    except RegexTimeout:
        return False


# ---------- overfit guard: regex that embeds a literal copied from the value ----------
def is_value_overfit(pattern: str, value) -> bool:
    """True if `pattern` contains a LITERAL character-run copied from `value` (e.g.
    value "6000123456" and pattern r"\\b(6000\\d{6})\\b" -> literal "6000"). The
    sibling-generalization test cannot catch this when every sample shares the
    overfit feature, so this guard does. Strips escaped classes, quantifier braces
    and char classes, leaving only literal text, then looks for digit/alnum runs
    that are a prefix of (or equal to) the value."""
    if value is None:
        return False
    v = str(value).strip()
    v_digits = re.sub(r'\D', '', v)
    scrub = re.sub(r'\\.', '', pattern)         # drop \d \s \. \b etc.
    scrub = re.sub(r'\{[^}]*\}', '', scrub)     # drop {6} {2,4} quantifiers
    scrub = re.sub(r'\[[^\]]*\]', '', scrub)    # drop [A-Z0-9] classes
    if len(v_digits) >= 4:
        for lit in re.findall(r'\d{2,}', scrub):
            if len(lit) >= 3 and (v_digits.startswith(lit) or v_digits == lit):
                return True
    v_alnum = re.sub(r'[^A-Za-z0-9]', '', v).upper()
    if len(v_alnum) >= 5:
        for lit in re.findall(r'[A-Za-z0-9]{4,}', scrub):
            L = lit.upper()
            if len(L) >= 4 and (v_alnum.startswith(L) or v_alnum == L):
                return True
    return False


def spec_is_overfit(spec: dict, value) -> bool:
    """True if ANY regex in the spec embeds a literal copied from `value`."""
    return any(is_value_overfit(p, value) for p in (spec.get("regex") or []))


# ---------- date-aware helpers (QR gives 01/03/2026, text may say 01-Mar-2026) ----------
_MONTHS = {m: i+1 for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])}
_MON_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def parse_date(s):
    """Return (d, m, y) from many formats, or None."""
    if s is None:
        return None
    s = str(s).strip()
    m = re.match(r'(\d{1,2})[./\-\s]+(\d{1,2})[./\-\s]+(\d{2,4})', s)   # 01/03/2026
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        return (d, mo, y)
    m = re.match(r'(\d{1,2})[./\-\s]+([A-Za-z]{3,})[./\-\s]+(\d{2,4})', s)  # 01-Mar-2026
    if m and m.group(2)[:3].lower() in _MONTHS:
        d, mo, y = int(m.group(1)), _MONTHS[m.group(2)[:3].lower()], int(m.group(3))
        y = y + 2000 if y < 100 else y
        return (d, mo, y)
    return None

def date_variants(value):
    """Text forms the same date might appear as, for searching raw text."""
    p = parse_date(value)
    if not p:
        return [str(value)]
    d, mo, y = p
    mon = _MON_ABBR[mo-1]
    outs = []
    for sep in ("/", "-", "."):
        outs.append(f"{d:02d}{sep}{mo:02d}{sep}{y}")
        outs.append(f"{d}{sep}{mo}{sep}{y}")
    for sep in ("-", " ", "/"):
        outs.append(f"{d:02d}{sep}{mon}{sep}{y}")
        outs.append(f"{d}{sep}{mon}{sep}{y}")
    # invoices also print UPPERCASE months and 2-digit years: 05-JAN-26, 05-JAN-2026,
    # 05.01.26, 05/01/26 — without these a real invoice date is rejected as absent
    yy = f"{y % 100:02d}"
    for sep in ("-", " ", "/", "."):
        outs.append(f"{d:02d}{sep}{mo:02d}{sep}{yy}")
    for sep in ("-", " ", "/"):
        for m in (mon.upper(), mon):
            outs.append(f"{d:02d}{sep}{m}{sep}{y}")
            outs.append(f"{d:02d}{sep}{m}{sep}{yy}")
            outs.append(f"{d}{sep}{m}{sep}{yy}")
    return list(dict.fromkeys(outs))
