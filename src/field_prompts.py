"""
field_prompts.py  —  writes the per-field "prompt" stored in the vendor JSON.

Why: production sends a field's "prompt" to Gemini when the regex for that field
fails on an invoice. A generic line ("Taxable value total in the summary row.") does
not help on a vendor whose invoice calls it 'Transaction Value'. This step writes an
instruction that is specific to THIS vendor layout, from what the agent has already
proven: the invoice text, the validated values and the patterns that extract them.

One Gemini call per layout (purpose "prompts" in token_usage.csv). If it fails, the
generic prompts are kept — a missing prompt never blocks a config.
"""
import json
import os

PROMPT = """You write extraction instructions for ANOTHER AI model.

That model will be given one invoice from this supplier (same layout as the sample
below) and asked for ONE field. It only gets your instruction for that field, so the
instruction must tell it exactly where the value is on THIS supplier's invoices.

You are given:
- the text of a sample invoice,
- the CORRECT value of each field on that sample (verified),
- the CORRECT values on other invoices of the same layout (so you can see what changes
  from invoice to invoice and what stays the same),
- the regex patterns that extract each value (they show the label / anchor used),
- fields that this layout does NOT print.

For EACH field listed in "fields", write 3 to 5 plain, complete SENTENCES (one per
line), in this order:
1. Where to look and the exact printed LABEL: "The <field in words> is printed after
   the label '<label as printed>' in the <block / part of the page>, just below / next
   to '<what is printed beside it>'." Mention a second label if the same value is
   printed twice, and the LAST page for totals on multi-page invoices.
2. The FORMAT of the value: "It is in DD.MM.YYYY format." / "It is a 10-digit number."
   / "It is an amount with two decimals and thousands commas."
3. What NOT to return, naming the nearby values that are easy to confuse with it on
   this layout: "Do not return the 'PO Date', which is a different date on the same
   invoice." / "Do not return 'Grand Total', which includes tax." / "Do not return the
   number in the 'GST %' column, which is a rate."
4. LAST line, exactly in this form (it shows the shape of the value only):
   "Format example only, do not return this value: <value from the sample>."
For a field in "absent_fields", write ONE sentence saying this layout does not print it
and what to return (0.0 for amounts, an empty string for text) unless a clearly
labelled value is present.

Rules:
- Full sentences. No headings such as "LABEL:", "WHERE:", "FORMAT:"; no bullet marks,
  no markdown, no regex.
- Describe the LAYOUT, not this one invoice: the last line is the only place a value
  from the sample may appear, and it must carry the "do not return this value" wording.
- Use only labels that are literally printed in the sample text. Do not invent labels.
- Do not describe things that are not visible in plain text (bold, colour, font size).
- Never tell the model to calculate, add or derive a value.

Return ONLY JSON: {{"prompts": {{"<field>": ["line", "line", ...], ...}}}}

fields: {fields}
absent_fields: {absent}

Correct values on the sample:
{values}

Correct values on other invoices of this layout:
{siblings}

Patterns that extract them:
{patterns}

=== SAMPLE INVOICE TEXT ===
{raw_text}
=== END ===
"""

MAX_LINES, MAX_CHARS = 8, 1500


_EXAMPLE = __import__("re").compile(r"^\s*(format\s+)?example(\s+only)?\b[^:]*:\s*(.+?)\s*$",
                                    __import__("re").I)


def _clean(lines):
    """Tidy the lines Gemini wrote. An example line is always rewritten into the safe
    form, so the model that later reads the prompt never takes it as the answer."""
    out = []
    for ln in lines if isinstance(lines, list) else []:
        ln = " ".join(str(ln).split())
        m = _EXAMPLE.match(ln)
        if m:
            value = m.group(3).rstrip(".")
            ln = f"Format example only, do not return this value: {value}."
        if ln:
            out.append(ln)
    out = out[:MAX_LINES]
    while out and sum(len(x) for x in out) > MAX_CHARS:
        out.pop()
    return out


def write_field_prompts(raw_text, values, patterns, absent=(), siblings=None,
                        model_name="gemini-2.5-flash"):
    """{field: [instruction lines]} for the fields in `values` and `absent`.
    Returns {} when switched off (FIELD_PROMPTS=0) or when the call fails."""
    if os.environ.get("FIELD_PROMPTS", "1") == "0" or not values:
        return {}
    try:
        from google.genai import types
        from src.generate import make_client, generate_with_retry, _fast_json_config, _extract_json
        prompt = PROMPT.format(
            fields=json.dumps(sorted(values)),
            absent=json.dumps(sorted(absent)),
            values=json.dumps(values, indent=1, ensure_ascii=False),
            siblings=json.dumps(siblings or [], indent=1, ensure_ascii=False),
            patterns=json.dumps(patterns, indent=1, ensure_ascii=False),
            raw_text=raw_text,
        )
        resp = generate_with_retry(make_client(), model=model_name, contents=prompt,
                                   config=_fast_json_config(types, 0), purpose="prompts")
        data = _extract_json(resp.text or "").get("prompts") or {}
    except Exception as e:
        print(f"   [prompts] could not write field prompts ({type(e).__name__}); "
              f"generic prompts kept")
        return {}
    allowed = set(values) | set(absent)
    out = {}
    for field, lines in data.items():
        lines = _clean(lines)
        if field in allowed and lines:
            out[field] = lines
    return out