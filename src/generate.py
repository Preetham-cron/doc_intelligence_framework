"""
generate.py  —  Step 3+4: propose candidate field specs, test them, keep winners.

Uses the NEW google-genai SDK (google.generativeai is deprecated).
    pip install google-genai

Pieces:
  1) call_gemini(...)      - real gemini-2.5-flash call (JSON output).
  2) build_field_specs(...) - generate -> test -> retry loop, judged by field_tester.
                              LLM-agnostic (pass a mock in tests).
"""
import json, re
from src.field_tester import test_field, value_context, match_span


PROMPT = """You generate extraction patterns for an invoice header config.
Return ONLY a JSON object: {{"specs": [ ... ]}} and nothing else.

Each spec extracts ONE field's value from the plain TEXT below (this is the EXACT
text the extraction pipeline feeds the regex; real newlines, no <br>, no pipes).
Regex are applied with re.DOTALL. For EACH field, return the `regex` field as an
ORDERED LIST of 2-3 INDEPENDENT candidate patterns: a PRIMARY plus 1-2 FALLBACKS.
The extraction pipeline tries them IN ORDER and uses the first that matches, so:
- The PRIMARY should be your best, most precise anchor.
- Each FALLBACK must anchor on a DIFFERENT stable cue than the primary (e.g. a
  different nearby label, or a structural position) so that if the primary's
  wording is absent on some invoice, a fallback still catches the value.
- Every candidate must be GENERIC (capture the value with a token pattern, never
  the literal value). Do not repeat the same pattern twice.
You may also give `alias` + `direction`. Use `negative_alias` to AVOID wrong matches.

CRITICAL — THE REGEX MUST GENERALIZE. This config is REUSED for MANY invoices of
the same template, which differ in their values and line items. Therefore:
- Anchor ONLY on STABLE TEMPLATE TEXT: fixed labels and column headers such as
  "Total Invoice Value (in figures)", "GSTN NO.", "Invoice No:", "IRN NO.:",
  "Purchase Order", "Invoice Date", "Taxable Value", "CGST Amount", etc.
- NEVER anchor on LINE-ITEM CONTENT that changes per invoice: part names, part
  numbers, product descriptions (e.g. "Seat kit 874WLE 3R"), HSN of a specific
  line, or a specific line quantity/amount. A regex anchored on such text will
  fail on every other invoice.
- Do NOT hard-code the specific value. Capture it generally, e.g. ([\\d,]+\\.\\d{{2}}).
- AMOUNT DECIMALS VARY: some PDFs drop trailing zeros, printing 48203 for 48203.00 and
  257.4 for 257.40. Unless EVERY invoice shown (this one AND the other invoices listed
  below) prints exactly 2 decimals, capture an amount with ([\\d,]+(?:\\.\\d{{1,2}})?)
  and skip N amount lines with (?:\\s*[\\d,]+(?:\\.\\d{{1,2}})?\\s*\\n){{N}}.
- A totals line can hold TWO numbers, e.g. "22 12.84": a quantity total then the
  amount. Capture the amount that belongs to the field's own column, not the count.
- For IDENTIFIERS (invoice no, PO/PDS number, credit note no): capture the WHOLE
  token, never a single digit. Require length, e.g. (\\d{{4,}}) or ([A-Z0-9][A-Z0-9/-]{{3,}}),
  and stop at whitespace/'Dtd'/'Date'. A regex that could return just "6" is WRONG.
- NEVER copy any characters FROM THE VALUE ITSELF into the pattern — not even a
  prefix. If the PO number is 6000123456, do NOT write (6000\\d{{6}}); the "6000"
  is part of the value and will make the regex fail on invoices whose PO starts
  differently (e.g. 4500…, 7100…). Instead anchor on the printed LABEL and capture
  a generic token: e.g.  (?:P\\.?\\s?O\\.?|PDS)\\s*(?:No\\.?|Number)?\\s*[:.\\-]?\\s*(\\d{{6,}}) .
  po_pds_number: PO and PDS are the SAME field — the label may be "PO No", "P.O. No",
  "PDS No", "PDS NO." or "Purchase Order"; your regex should accept BOTH the PO and
  PDS wordings (e.g. anchor on (?:P\\.?\\s?O\\.?|PDS)).
  The label is stable across invoices; the value is not. This rule applies to ALL
  fields: the capturing pattern must be built from LABELS and generic token shapes
  only, never from the specific digits/letters of the value you were given.
- For a TAX AMOUNT, the regex MUST contain one of that tax's own LABEL words as the
  anchor, exactly as printed on the invoice:
     CGST -> "CGST" / "Central ... Tax" / a "... GST ..." header
     SGST -> "SGST" / "UTGST" / "State ... Tax" / a "... GST ..." header
     IGST -> "IGST" / "Integrated ... Tax" / a "... GST ..." header
     CESS -> "CESS" or "TCS"  (many invoices print it as "TCS", e.g. "TOTAL TCS VALUE")
  e.g. CGST[^\\n]*\\n?\\s*([\\d,]+\\.\\d{{2}}). A purely positional tax regex (no tax
  label in it) is rejected: it grabs the WRONG amount on an invoice with another tax.
- For gstin_number, set "alias": [] (EMPTY). The extraction pipeline tries the ALIAS
  first for GSTIN, and a "GSTIN" alias will grab the BUYER's GSTIN (under "Details of
  Receiver"/"Bill To"). Rely on a REGEX that anchors on the SELLER's GSTIN near the
  top / supplier name / supplier PAN, so the SELLER (not buyer) is captured.
- NEVER nest a repeat inside a repeated group, e.g. (?:\\s*[\\d,]+\\s*)+ or (a+)*: such
  patterns backtrack catastrophically and are REJECTED. To skip N amounts use a FIXED
  count (?:\\s*[\\d,]+\\.\\d{{2}}\\s*\\n){{2}}, and to skip text use a single lazy .*? .
- NEVER use a start-of-string anchor '^'. The text begins with "# Page 1", so a
  '^' anchor will never match. Anchor on the field's own label instead (e.g.
  capture 'GST INVOICE' as "(GST INVOICE)", not "^GST INVOICE").

For TAX TOTALS (taxable / CGST / SGST / IGST / CESS / total): work out the layout
family from the raw text and anchor accordingly. CGST and SGST are often EQUAL, so
each MUST capture a DIFFERENT position — never by value alone.

FAMILY A — tax word directly beside its amount:
   "CGST\\n27960.64"      -> (?<![A-Za-z])CGST[^\\n]*\\n\\s*([\\d,]+\\.\\d{{2}})
   "SGST Value 9.00 %\\n1,274.41" -> SGST[^\\n]*\\n\\s*([\\d,]+\\.\\d{{2}})
   Each tax's regex CONTAINS its own tax word, so it can't grab another tax's number.

FAMILY B — a BLOCK of labels then a BLOCK of values (Toyota Boshoku), e.g.
   TOTAL BASIC VALUE / TOTAL CGST PAYABLE / TOTAL SGST PAYABLE / TOTAL IGST PAYABLE / TCS ON SCRAP
   1,569.20 / 192.70 / 192.70 / 0.00 / 0.00 / TOTAL AMOUNT / 1,954.60
   Anchor on the LAST label before the values (here "TCS ON SCRAP") and count to the
   target's position (labels map 1:1 to values, in order). For a TAX field, START the
   pattern at that tax's own label inside the label block and run to the values
   (DOTALL lets .*? cross lines), so the regex carries the tax word:
     basic: TCS ON SCRAP\\s*\\n\\s*([\\d,]+\\.\\d{{2}})
     cgst:  CGST PAYABLE.*?TCS ON SCRAP\\s*\\n\\s*[\\d,]+\\.\\d{{2}}\\s*\\n\\s*([\\d,]+\\.\\d{{2}})
     sgst:  SGST PAYABLE.*?TCS ON SCRAP\\s*\\n(?:\\s*[\\d,]+\\.\\d{{2}}\\s*\\n){{2}}\\s*([\\d,]+\\.\\d{{2}})
   (also seen with labels "TOTAL CGST VALUE / TOTAL SGST VALUE / TOTAL IGST VALUE".)

FAMILY C — values under ONE generic label, ordered by the COLUMN HEADER (Asahi):
   header reads "SGST/UTGST" then "CGST"; the summary is
   "Total GST value (in figure)\\n1,059.68\\n1,059.68"  -> first value is SGST, second CGST:
     sgst: Total GST value \\(in figure\\)\\s*\\n\\s*([\\d,]+\\.\\d{{2}})
     cgst: Total GST value \\(in figure\\)\\s*\\n\\s*[\\d,]+\\.\\d{{2}}\\s*\\n\\s*([\\d,]+\\.\\d{{2}})

FAMILY D — a single totals row before a stable label like "Total Invoice Value":
   count the newline-separated amounts backward/forward from that label to the target.

FAMILY E — labels printed in one place, their VALUES printed much later, after
   unrelated boilerplate (declaration text, "Prepared By", signatures). e.g.
     Total :\\nTotal Invoice value (in figure) :\\n ... Declaration ... \\nPrepared By\\n 5228.28\\n 941.09\\n 6169.37
   Anchor on the STABLE line directly ABOVE the value block (here "Prepared By")
   and count values:  taxable: Prepared By\\s*\\n\\s*([\\d,]+\\.\\d{{2}})
                      total:   Prepared By\\s*\\n(?:\\s*[\\d,]+\\.\\d{{2}}\\s*\\n){{2}}\\s*([\\d,]+\\.\\d{{2}})
   For a TAX in this layout, start at the tax's column header and run (.*?) to the
   same anchor:       igst:    IGST\\s*\\n\\s*%.*?Prepared By\\s*\\n\\s*[\\d,]+\\.\\d{{2}}\\s*\\n\\s*([\\d,]+\\.\\d{{2}})
   Which position holds which amount: taxable + taxes = total (check the arithmetic).

INTER- vs INTRA-STATE: the trusted values already carry the correct tax (IGST when
the seller's and buyer's GSTIN states differ, CGST+SGST when they are the same). A
header like "CGST % | SGST/UTGST % | IGST %" with ONE amount under it belongs to the
tax that is in the trusted values — anchor that tax's regex on ITS header word.

A tax amount that is 0.00 / absent on this invoice: do NOT emit a regex for it
(leave it out); it defaults to 0. Only emit a tax regex for a NON-ZERO tax.

ALWAYS ALSO GIVE AN ALIAS FALLBACK FOR AMOUNTS. For every amount field (taxable /
basic / CGST / SGST / IGST / CESS / total), in ADDITION to the regex, set:
  "alias":     the field's exact printed LABEL as it appears (e.g. ["Taxable Value"],
               ["Assessable Value"], ["CGST Amount"], ["Total Invoice Value"],
               ["CESS Amount"]) — for a tax amount the alias MUST contain that tax word,
  "direction": where the number sits RELATIVE to that label — use EXACTLY one of:
               "right"  (value on the SAME line as the label),
               "bottom" (value on a FOLLOWING line, e.g. under a column header),
               "above"  (value on a PRECEDING line).
               Do NOT use "below" — the pipeline only understands right/bottom/above.
If your regex misses, the pipeline scans from this label in that direction to pull the
number — so a summary amount under a distant column header is still caught. Pick the
direction from the raw text (column header with the value beneath it -> "bottom").

document_type: the title ("TAX INVOICE", "CREDIT NOTE", "GST INVOICE") is stable
template text, so capture it literally: (TAX\\s+INVOICE) / (CREDIT\\s+NOTE). Allow a
line break inside it (\\s+ matches a newline under DOTALL). This is the ONE field
where the captured text equals the value — that is correct here.

total_invoice_taxable_value: this is the PRE-TAX taxable/assessable GRAND TOTAL — it
is one of the LARGEST amounts on the invoice and roughly equals (total invoice value
− all taxes). It is NEVER a small number, a tax RATE ("1.50", "9.00", "18.00"), or a
single line-item amount — if your candidate returns one of those it is WRONG. Anchor
on the summary label, treating ALL these as the SAME field: "Taxable Value",
"Assessable Value", "Asse. Value", "Asse Value", "Total Taxable Value", "Total
Assessable Value". If the number sits in a values-block under a column header (no
inline label), anchor on the nearest STABLE label BEFORE the block and count the
newline-separated amounts to the taxable column's position (see FAMILY B/C).

irn_number: the 64-char hex IRN is frequently SPLIT across two lines. Capture EXACTLY
64 hex chars while tolerating whitespace between them, so BOTH a one-line and a
wrapped IRN match and NO trailing text is grabbed:
   IRN[^\\n]*?[:\\-]?\\s*((?:[0-9A-Fa-f]\\s*){{64}})
The value is cleaned to contiguous hex afterward. Do NOT anchor on the specific hex,
and do NOT use an open-ended run like ([0-9A-Fa-f]+(?:\\s*[0-9A-Fa-f]+)*) — it will
swallow letters from the next label (e.g. "Ack No").

Spec shape — return the COMPLETE field block for EACH field (you generate the whole
block; code only validates it against the schema afterwards):
  {{"name": str,
    "alias": [str],                 // label words near the value, or []
    "direction": "right|bottom|above",   // bottom = value on a following line; NEVER "below"
    "datatype": "amount|gstin|date|percentage|string",
    "scope": "vendor|invoice",
    "priority": int (1-10),
    "required": true|false,
    "validation": [str],            // e.g. ["numeric"], ["gstin","gstin_checksum"], ["date"], ["non_empty"]
    "regex": [str],                 // ORDERED candidate patterns (primary + fallbacks)
    "prompt": [str],                // one short human hint
    "fallback_alias": [str],
    "negative_alias": [str]}}

Use these FIXED schema values per field (they are the same on every invoice — emit
them exactly):
{schema_table}

Known TRUSTED values for THIS invoice (your specs must extract EXACTLY these):
{trusted}

WHERE each trusted value appears in the raw text (anchor your regex on the
surrounding LABEL text — do NOT hardcode the value itself; capture it with a
general pattern like ([\\d,]+\\.\\d{{2}}) so it works on other invoices too):
{contexts}

Hints:
{hints}

{feedback}

The SAME fields on OTHER invoices of this SAME template. Your regex is validated on
these too and MUST extract exactly these values from them. Look at how each value is
printed there (a total may be '48203' with NO decimals, or '257.4' with ONE decimal,
where this invoice shows '5532.38') and write ONE pattern that fits ALL of them:
{siblings}

Fields to generate specs for: {fields}

RAW TEXT:
------
{raw_text}
------
"""


def _repair_escapes(text: str) -> str:
    _valid = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|(\\)')
    return _valid.sub(lambda m: m.group(0) if m.group(1) is not None else '\\\\', text)


def _salvage_specs(text: str) -> list:
    """Pull COMPLETE {...} objects out of a (possibly truncated) specs array by
    bracket-matching, honouring strings/escapes. Keeps the specs that fully
    arrived, drops the one cut off mid-stream. Never raises."""
    out, i = [], text.find('[')
    if i == -1:
        return out
    depth = 0; start = None; in_str = False; esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == '{':
            if depth == 0: start = j
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                frag = text[start:j+1]
                for cand in (frag, _repair_escapes(frag)):
                    try:
                        out.append(json.loads(cand)); break
                    except json.JSONDecodeError:
                        continue
                start = None
    return out


def _extract_json(text: str) -> dict:
    """Never raises. Full parse -> escape-repaired parse -> salvage complete specs
    from a truncated/garbled response. A dropped spec just goes to retry/fallback,
    so ONE bad LLM response costs one field, never the whole batch."""
    text = re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.M).strip()
    for cand in (text, _repair_escapes(text)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return {"specs": _salvage_specs(text)}


def _fast_json_config(types, think=0):
    """Kept for compatibility: see json_config."""
    return json_config(types, think)


def make_client():
    """Gemini client WITH a request timeout (default 180 s, env GEMINI_TIMEOUT_S).
    Without one, a dropped connection waits forever and hangs that worker."""
    import os
    from google import genai
    from google.genai import types
    from src.config import load_env
    ms = int(float(os.environ.get("GEMINI_TIMEOUT_S", "180")) * 1000)
    try:
        return genai.Client(api_key=load_env(), http_options=types.HttpOptions(timeout=ms))
    except Exception:
        return genai.Client(api_key=load_env())


def json_config(types, think=0, max_tokens=8192):
    """temperature 0, JSON output, thinking budget `think`, and automatic function
    calling OFF (we pass no tools; it only triggers the 'Direct use of AFC' warning)."""
    kw = dict(temperature=0.0, response_mime_type="application/json",
              max_output_tokens=max_tokens + int(think or 0))
    for extra in (dict(thinking_config=types.ThinkingConfig(thinking_budget=int(think or 0)),
                       automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
                  dict(thinking_config=types.ThinkingConfig(thinking_budget=int(think or 0))),
                  {}):
        try:
            return types.GenerateContentConfig(**kw, **extra)
        except Exception:
            continue


_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
              "DEADLINE", "timed out", "Timeout", "timeout", "Connection", "overloaded", "ReadError")


_TIMEOUT_WORDS = ("timed out", "timeout", "deadline", "readtimeout", "read operation timed out")


def generate_with_retry(client, tries=4, **kw):
    """client.models.generate_content with backoff on TRANSIENT errors. A rate limit
    (429) or 'overloaded' (503) is waited out and retried (templates run in parallel;
    a hiccup must not become a 'failed field'). A TIMEOUT is retried only ONCE: a
    request that already ran the full timeout will most likely do so again, and
    retrying it many times is what made a single template run for half an hour.
    Non-transient errors raise immediately."""
    import time, random
    from src import usage
    purpose = kw.pop("purpose", "")
    timeouts = 0
    for k in range(tries):
        try:
            resp = client.models.generate_content(**kw)
            usage.record(resp, model=kw.get("model", ""), purpose=purpose)
            return resp
        except Exception as e:
            usage.record(None, model=kw.get("model", ""), purpose=purpose, status="failed")
            msg = f"{type(e).__name__}: {e}".lower()
            if any(w in msg for w in _TIMEOUT_WORDS):
                timeouts += 1
                if timeouts > 1 or k == tries - 1:
                    raise
                continue                                   # one immediate retry
            if k == tries - 1 or not any(t.lower() in msg for t in _TRANSIENT):
                raise
            time.sleep(min(20, 2 ** (k + 1)) + random.random())


def call_gemini(raw_text, trusted, fields, hints=None, feedback="", contexts=None,
                model_name="gemini-2.5-flash", think=0, siblings_info=None):
    from google import genai
    from google.genai import types
    from src.config import load_env
    client = make_client()
    from src.field_schema import generate_schema_table_text
    prompt = PROMPT.format(
        schema_table=generate_schema_table_text(),
        trusted=json.dumps(trusted, indent=2),
        contexts=json.dumps(contexts or {}, indent=2),
        hints=json.dumps(hints or {}, indent=2),
        feedback=("Previous attempt problems (fix these):\n" + feedback) if feedback else "",
        fields=json.dumps(fields),
        raw_text=raw_text,
        siblings=(json.dumps(siblings_info, indent=1) if siblings_info else
                  "(no other invoices of this template were checked)"),
    )
    try:
        resp = generate_with_retry(
            client, model=model_name, contents=prompt, config=_fast_json_config(types, think),
            purpose="regex")
    except Exception as e:
        print(f"   [generate] Gemini call failed ({type(e).__name__}); "
              f"treating as no specs this attempt")
        return []
    return _extract_json(resp.text or "").get("specs", [])


def build_field_specs(raw_text, trusted, generator, hints=None, max_retries=2, distinct_groups=None):
    """
    trusted:   {field_name: expected_value}
    generator: fn(raw_text, trusted, fields, hints=, feedback=) -> list[spec]
    Returns (passing_specs, info).
    """
    passing, report, pending = {}, [], list(trusted.keys())
    feedback = ""

    for attempt in range(max_retries + 1):
        if not pending:
            break
        want = list(pending)
        contexts = {k: value_context(raw_text, trusted[k]) for k in want}
        try:
            specs = generator(raw_text, {k: trusted[k] for k in want}, want,
                              hints=hints, feedback=feedback, contexts=contexts)
        except TypeError:
            specs = generator(raw_text, {k: trusted[k] for k in want}, want)  # mock-friendly
        by_name = {s["name"]: s for s in specs}

        still, fails = [], []
        for name in pending:
            spec = by_name.get(name)
            if not spec:
                still.append(name); continue
            r = test_field(spec, raw_text, trusted[name])
            report.append({"attempt": attempt, **r})
            if r["passed"]:
                passing[name] = spec
            else:
                still.append(name)
                fails.append(f"- {name}: your spec extracted {r['got']!r}, expected {trusted[name]!r}")
        pending = still
        feedback = "\n".join(fails)

    # position-distinctness: fields in a group must capture DIFFERENT offsets
    for group in (distinct_groups or []):
        seen = {}
        for name in group:
            if name not in passing:
                continue
            sp = match_span(passing[name], raw_text)
            if sp is not None and sp in seen.values():
                del passing[name]
                if name not in pending:
                    pending.append(name)
            else:
                seen[name] = sp

    return passing, {"report": report, "unresolved": pending}