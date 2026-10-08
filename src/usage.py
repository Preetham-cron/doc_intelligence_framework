"""
usage.py  —  counts every Gemini API call and its tokens.

Every call in the project goes through generate.generate_with_retry, which reports
here. Each call is tagged with the template being built (e.g. 29AAAAA0000A1Z5:TAXINVOICE:V1)
and what it was for (discover / regex / verify), added to the running totals, and
appended to token_usage.csv so the cost of a run can be looked at afterwards.

Token fields come from Gemini's usage_metadata:
  input    = prompt_token_count
  output   = candidates_token_count + thoughts_token_count  (thinking is billed as output)
  total    = total_token_count
"""
import csv
import datetime
import os
import threading

_LOCK = threading.Lock()
_LOCAL = threading.local()
_ROWS = []                       # one dict per API call, in order
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "token_usage.csv")
_HEADER = ["time", "gstin", "template", "purpose", "model", "status",
           "input_tokens", "output_tokens", "thinking_tokens", "total_tokens"]


def set_tag(tag):
    """Name the template this thread is working on (None to clear)."""
    _LOCAL.tag = tag


def get_tag():
    return getattr(_LOCAL, "tag", None)


def record(resp, model="", purpose="", status="ok"):
    """Count one API call. `resp` is the Gemini response (None when the call failed)."""
    um = getattr(resp, "usage_metadata", None) if resp is not None else None
    n = lambda a: int(getattr(um, a, 0) or 0) if um is not None else 0
    inp, cand, think, total = (n("prompt_token_count"), n("candidates_token_count"),
                               n("thoughts_token_count"), n("total_token_count"))
    out = cand + think
    tag = get_tag() or ""
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "gstin": tag.split(":")[0], "template": tag, "purpose": purpose or "",
           "model": model or "", "status": status,
           "input_tokens": inp, "output_tokens": out, "thinking_tokens": think,
           "total_tokens": total or (inp + out)}
    with _LOCK:
        _ROWS.append(row)
        if os.environ.get("TOKEN_LOG", "1") != "0":
            try:
                new = not os.path.exists(LOG_PATH)
                with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=_HEADER)
                    if new:
                        w.writeheader()
                    w.writerow(row)
            except Exception:
                pass                                     # logging must never stop a run


def mark():
    """A position in the call log; pass it to totals()/by_gstin() to count from here."""
    with _LOCK:
        return len(_ROWS)


def _sum(rows):
    return {"calls": len(rows),
            "input": sum(r["input_tokens"] for r in rows),
            "output": sum(r["output_tokens"] for r in rows),
            "thinking": sum(r["thinking_tokens"] for r in rows),
            "total": sum(r["total_tokens"] for r in rows)}


def totals(since=0, template=None):
    with _LOCK:
        rows = [r for r in _ROWS[since:] if template is None or r["template"] == template]
    return _sum(rows)


def by_gstin(since=0):
    with _LOCK:
        rows = list(_ROWS[since:])
    out = {}
    for r in rows:
        out.setdefault(r["gstin"] or "(other)", []).append(r)
    return {g: _sum(rs) for g, rs in out.items()}


def fmt(t):
    return (f"{t['calls']} API call(s) | input {t['input']:,} | output {t['output']:,} "
            f"| total {t['total']:,} tokens")


def print_summary(title, since=0):
    """One line: API calls and tokens used since `since` (all GST numbers together).
    The per-call detail stays in token_usage.csv."""
    tot = totals(since)
    print(f"TOKEN USAGE — {title}: {fmt(tot)}")
    return tot