"""
discover.py  —  evidence-grounded field discovery (STRICT).

Implements the spec's core rule:   Schema ∩ Evidence-in-Raw-Text = Extractable.

The LLM inspects the RAW TEXT against the target schema and, for EACH field,
reports whether there is EXPLICIT evidence in the raw text, the exact value as
it appears, the label/context, and any ambiguity. It must NOT infer, calculate,
derive, normalize, or hallucinate. A field present in the schema but absent from
the raw text is returned present=false and gets NO extraction rule downstream.

QR data is NOT used here (raw text is the sole source of truth). QR is only a
later cross-check.
"""
import os, json, re

# target schema = the canonical header fields (same names as the old config)
SCHEMA_FIELDS = {
    "document_type":                "The document TITLE at the top. One of: 'TAX INVOICE', 'GST INVOICE', 'CREDIT NOTE', 'DEBIT NOTE'. (A credit note also carries a 'CREDIT NOTE NO' label.)",
    "gstin_number":                 "The SUPPLIER's 15-char GSTIN — the seller near the top/supplier name. Labels: 'GSTIN:', 'GSTN NO.', 'GST No.:', 'GSTIN/UIN'. NOT the buyer/consignee GSTIN (under 'Bill To', 'Ship To', 'Detail of Receiver', 'Buyer', 'Consignee').",
    "po_pds_number":                "The Purchase Order / PDS number — PO and PDS are the SAME field here, just different labels on different vendors. Capture whichever appears. Labels (all mean this field): 'Purchase Order', 'PO No', 'PO NO:', 'P.O NO:', 'P.O. No.', 'PO Number', 'PDS No', 'PDS NO.', 'PDS Number', 'Order No'. If BOTH a PO and a PDS number are printed, prefer the PO number. Value is the number, not the date. Do NOT report the e-invoice 'Ack No' (a ~15-digit acknowledgement number), the IRN, an E-way bill number or the invoice number as the PO. The text is linearised from a PDF, so a label can be followed by values that belong to OTHER nearby labels (e.g. 'PO. No. & Date :' followed by the IRN, the Ack No and the Ack Date) — pick the value whose format and position fit a PO; the PO often also appears in a 'PO No.' column of the line items.",
    "order_date":                   "The purchase-order date (a DATE). Labels (all mean this): 'Order Date', 'PO Date', 'P.O DATE', 'P.O.DATE', 'Purchase Order Date', or 'Dtd:'/'Dt:' shown next to the PO number. NOT the e-invoice 'Ack Date' and NOT the invoice date. If the only PO date printed is a placeholder such as 30-12-1899 or 01-01-1900, order_date is ABSENT.",
    "vendor_code":                  "The supplier/vendor code, when the invoice has a 'Vendor Code' / 'Supplier Code' / 'Supplier ID' label with a value next to it (e.g. 'Vendor Code : D002' -> D002). Report that value. Do NOT confuse it with a longer code it may be a prefix of (e.g. a part number or DI/E-Nagare number like 'D002AA1G8028'); report the standalone vendor code only.",
    "supplier_document_number":     "The document number of THIS invoice/note. Labels: 'Invoice No', 'Invoice Number', 'INVOICE NO :', 'INVOICE NO.:', 'Bill No', 'Document No'. On a CREDIT NOTE it is the 'CREDIT NOTE NO' value.",
    "supplier_document_date":       "The invoice/document date (a DATE). Labels (all mean this): 'Invoice Date', 'Date of Invoice', 'Date of Issue', 'DATE OF ISSUE', 'Date & Time of Issue of Invoice'.",
    "invoice_total_basic_value":    "The 'Total Sales Amount' column TOTAL (pre-tax sales, before assessable/taxable adjustments). A DIFFERENT column from 'Taxable Value'. Only if a 'Total Sales Amount' / 'Basic Amount' column literally exists. 'Transaction Value' and 'Sub Total' are NOT this field: they are the taxable value.",
    "total_invoice_taxable_value":  "The TAXABLE value TOTAL from the summary row. This field has MANY label variants that ALL mean the same thing — treat them as equal: 'Taxable Value', 'Assessable Value', 'Asse. Value', 'Asse Value', 'Assess. Value', 'TOTAL TAXABLE VALUE', 'Total Assessable Value', 'Transaction Value', 'Taxable Amount', and a pre-tax 'Sub Total' printed just above the tax totals. It is the amount the GST is charged on: taxable + IGST/CGST/SGST = invoice total (unless amortisation is involved). It is NEVER a rate or percentage such as 18.00 from a 'GST %' column, and it is always larger than the tax amount. The label may be an ABBREVIATION (e.g. 'Asse.') and may be a COLUMN HEADER far from the value, with the total sitting in a summary/total row. Different column from 'Total Sales Amount' — read the header.",
    "total_invoice_igst_tax_value": "The IGST amount TOTAL (summary row). Labels: 'IGST Amount', 'TOTAL IGST VALUE', 'IGST Value'. Often 0 on intra-state invoices.",
    "total_invoice_cgst_tax_value": "The CGST amount TOTAL (summary row). Labels: 'CGST Amount', 'TOTAL CGST VALUE', 'CGST Value Rs.', 'CGST'. Take the SUMMARY total, not a single line item.",
    "total_invoice_sgst_tax_value": "The SGST/UTGST amount TOTAL (summary row). Labels: 'SGST Amount', 'SGST/UTGST Amount', 'TOTAL SGST VALUE', 'SGST Value Rs.'. Frequently EQUAL to CGST — still read it from its own column.",
    "total_invoice_cess_tax_value": "The CESS/TCS amount TOTAL (summary row). Labels: 'CESS Amount', 'CESS / TCS Amount', 'TOTAL TCS VALUE', 'Comp Cess'. Often 0. Do NOT confuse with amortization/rounding amounts.",
    "total_invoice_value":          "The final GRAND TOTAL of the invoice (usually the largest amount, at the bottom). Labels: 'Total Invoice Value (in figures)', 'Total Invoice Value (In Figure)', 'TOTAL AMOUNT', 'Invoice Value', 'Grand Total'.",
    "irn_number":                   "The 64-character hexadecimal IRN. Labels: 'IRN NO.:', 'IRN :', 'IRN No.'. It may be split across two lines — join them into 64 hex chars.",
}

DISCOVER_PROMPT = """You are an evidence-grounded extraction analyst. You are
given a TARGET SCHEMA (fields that COULD exist) and the RAW TEXT of ONE invoice
(the source of truth).

Your job: for each schema field, decide whether it has EXPLICIT evidence in the
RAW TEXT, and if so report its exact value, label, and context.

ABSOLUTE RULES (do not violate):
- The RAW TEXT is the ONLY source of truth. A field existing in the schema does
  NOT mean it exists in the document.
- Report a value ONLY if it LITERALLY appears in the raw text. Copy it verbatim.
- NEVER infer, calculate, derive, split, normalize, convert, or hallucinate a
  value. Do not use a total to produce a basic/tax value. Do not populate a
  field just because a similar field exists.
- If there is no explicit evidence for a field, set present=false, value=null.
- Distinguish the SUPPLIER's values from the BUYER/CONSIGNEE's (e.g. GSTIN).
- If several values could match one field and you cannot safely choose, set
  present=true, ambiguous=true, and list candidates with a short reason.
- If a field appears multiple times (header vs line-item), say which occurrence
  is the intended header/summary value.
- Map amounts to fields by their COLUMN HEADER, left-to-right. Do not swap
  'Total Sales Amount'/'Basic' with 'Taxable Value'/'Assessable'; they are
  DIFFERENT columns with DIFFERENT headers. Read each from its OWN column.
- Do NOT copy the SAME number into invoice_total_basic_value AND
  total_invoice_taxable_value. If the invoice has only ONE of those columns, mark
  only THAT field present and the other absent. Report the same value for both
  ONLY if each column INDEPENDENTLY shows that amount.
- Recognise label variants: Purchase Order = PO No = PDS NO (po_pds_number).
- Labels are often ABBREVIATED or spaced oddly by the PDF: 'Asse. Value' =
  'Assessable Value' = 'Taxable Value'; 'Inv. No' = 'Invoice No'; 'Qty.' =
  'Quantity'; 'Doc No' = 'Document No'; 'Amt' = 'Amount'. Expand abbreviations and
  match by MEANING. A trailing '.' or extra spaces inside a label do not change it.
- Labels VARY across vendors — match each field by MEANING using the label hints
  in the schema, not by an exact string. A field is present when a recognisable
  label carries a value, or (for totals) a clearly-headed summary column/row exists.
- A value may sit on the SAME line as its label OR on a NEARBY line (the text is
  linearised) — look at both; report the value that belongs to the label.
- For the tax totals (IGST/CGST/SGST/CESS/taxable), use the SUMMARY / TOTAL row,
  never a single line-item amount. CGST and SGST are frequently the SAME number —
  report each from its own column; do not drop one because it equals the other.
- For IDENTIFIERS (invoice no, PO/PDS number, credit note no, IRN), report the
  COMPLETE value exactly as printed (e.g. "6000011782"), never a single digit or
  a fragment. Stop before a following 'Dtd'/'Date'/'Amend'.
- Be decisive: if a clear label+value exists, mark present=true (do not leave a
  clearly-present field as ambiguous). Reserve ambiguous=true for genuine ties.

Return ONLY JSON:
{{
  "fields": {{
    "<field>": {{
      "present": true/false,
      "value": <verbatim string/number from raw text, or null>,
      "label": "<the label/keyword seen next to it, or null>",
      "context": "<where it appears: header / totals row / etc., or null>",
      "ambiguous": true/false,
      "candidates": [<optional list when ambiguous>],
      "reason_absent": "<short reason if present=false, optional>"
    }}
  }}
}}

TARGET SCHEMA (fields, with meaning):
{schema}

RAW TEXT (source of truth):
------
{raw_text}
------
"""


def _extract_json(text):
    text = re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.M).strip()
    _valid = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|(\\)')
    for cand in (text, _valid.sub(lambda m: m.group(0) if m.group(1) is not None else '\\\\', text)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return {}   # never raise: a bad discovery response -> no fields, not a crash


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "cache", "discover")


def _cache_key(raw_text, model_name):
    import hashlib
    h = hashlib.sha1()
    for part in (DISCOVER_PROMPT, json.dumps(SCHEMA_FIELDS, sort_keys=True), model_name, raw_text):
        h.update(part.encode("utf-8")); h.update(b"\x00")
    return h.hexdigest()


def discover(raw_text, model_name="gemini-2.5-flash", use_cache=True):
    """
    Returns:
      present  : {field: value}   -- only fields with explicit evidence
      evidence : {field: full record}  -- labels, context, ambiguity, reasons
    Results are cached on disk (cache/discover/) per invoice text + prompt version, so
    the repair pass and validate.py never pay for the same discovery twice. Editing
    the discovery prompt or schema invalidates the cache automatically.
    """
    path = os.path.join(_CACHE_DIR, _cache_key(raw_text, model_name) + ".json")
    if use_cache and os.path.exists(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            return d["present"], d["evidence"]
        except Exception:
            pass
    present, evidence = _discover_uncached(raw_text, model_name)
    if use_cache and (present or evidence):
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            import threading
            tmp = path + f".{os.getpid()}.{threading.get_ident()}.tmp"
            json.dump({"present": present, "evidence": evidence}, open(tmp, "w", encoding="utf-8"))
            os.replace(tmp, path)
        except Exception:
            pass
    return present, evidence


def _discover_uncached(raw_text, model_name="gemini-2.5-flash"):
    from google import genai
    from google.genai import types
    from src.config import load_env
    from src.generate import make_client, json_config
    client = make_client()
    schema_desc = "\n".join(f"- {k}: {v}" for k, v in SCHEMA_FIELDS.items())
    # thinking OFF: discovery is fully specified by the prompt
    cfg = json_config(types, think=0)
    from src.generate import generate_with_retry
    resp = generate_with_retry(
        client,
        model=model_name,
        contents=DISCOVER_PROMPT.format(schema=schema_desc, raw_text=raw_text),
        config=cfg,
        purpose="discover",
    )
    data = _extract_json(resp.text or "").get("fields", {})

    present, evidence = {}, {}
    for f, rec in data.items():
        if f not in SCHEMA_FIELDS:
            continue
        evidence[f] = rec
        # ONLY fields with explicit evidence and a concrete value become extractable
        if rec.get("present") and not rec.get("ambiguous") and rec.get("value") not in (None, "", []):
            present[f] = rec["value"]
    return present, evidence


# ---- deterministic verification that a discovered value truly is in the raw text ----
def verify_in_text(value, raw_text):
    """Guard against hallucination: the value must occur in the raw text
    (dates may appear in a different format, so accept any date-variant)."""
    if value is None:
        return False
    s = str(value).strip()
    if s and s in raw_text:
        return True
    try:
        from src.field_tester import date_variants, parse_date
        if parse_date(s):
            low = raw_text.lower()
            if any(v.lower() in low for v in date_variants(s)):
                return True
    except Exception:
        pass
    # amounts may appear with commas
    if re.match(r'^\d+(\.\d+)?$', s):
        try:
            return f"{float(s):,.2f}" in raw_text or f"{float(s):.2f}" in raw_text
        except ValueError:
            return False
    # 64-char hex identifiers (IRN) are frequently split across a line break /
    # whitespace in the raw text, so the CONTIGUOUS value Gemini reports won't
    # appear as a literal substring — check it with whitespace tolerated between
    # hex characters instead.
    if re.fullmatch(r'[0-9A-Fa-f]{64}', s):
        spaced_pattern = r'\s*'.join(re.escape(ch) for ch in s)
        if re.search(spaced_pattern, raw_text, re.I):
            return True
    return False


# ---------------------------------------------------------------------------
# REVIEW: re-verify specific fields by reading the actual PDFs (multimodal)
# ---------------------------------------------------------------------------
VERIFY_PROMPT = """You are VERIFYING invoice field values that an earlier text-only step may
have read WRONG.

The {n} invoices below come from the SAME supplier and the SAME document template:
every field is printed with the SAME label, in the SAME place and in the SAME format on
each of them. Use that — the value you report for a field must come from the same
label / column / position on every invoice.

For each invoice you get the PDF (use its visual layout to see which label or column a
value belongs to) and its extracted TEXT (what the extraction regexes run on).
For each requested field and each invoice, return the value EXACTLY as it appears in
that invoice's TEXT (same digits, commas, decimals, letter case), the label it belongs
to, and the full TEXT line that contains it. If the field is genuinely not printed on
an invoice, return null for it.

Fields (meaning):
{fields}

Values reported earlier (they may be WRONG — check each one against the PDF):
{previous}

Rules:
- A PO / PDS number is never the invoice number, the e-invoice Ack No, the IRN, an
  E-way bill number, an HSN code, or a vehicle / transporter / dispatch reference.
  In linearised text its label may be printed after the value.
- A totals line such as "22 12.84" is usually a quantity total followed by an amount:
  report the amount of the field's own column.
- The taxable / assessable value can differ from the basic / sales value (e.g.
  amortisation added): take the column headed Taxable / Assessable.
- CGST and SGST are usually equal; report each from its own column.
- The taxable / assessable value is the BASE the GST was computed on: CGST (= SGST)
  = taxable x rate (2.5/6/9/14 %), IGST = taxable x rate (5/12/18/28 %). Use this to
  pick the right column — it can differ from the sales value and from (invoice total
  - taxes) when amortisation is part of the assessable value. If earlier values for
  different invoices came from DIFFERENT columns, pick the column that satisfies this
  on every invoice — even if that means changing the FIRST invoice's value.
- Do NOT compute, round or reformat anything: copy the value from the TEXT.

Return ONLY JSON:
{{"invoices": {{"<invoice name>": {{"<field>": {{"value": <string or null>, "label": "<label>",
  "line": "<text line containing the value>"}}}}}}}}
"""


def verify_fields(members, fields, previous=None, model_name="gemini-2.5-pro", think=4096):
    """Re-read `fields` on ALL invoices of one template in ONE request, giving Gemini
    each invoice's PDF (layout) and extracted text. members: [{"name", "pdf_path",
    "raw"}]. Returns {invoice_name: {field: value or None}}."""
    from google.genai import types
    from src.generate import make_client, json_config, generate_with_retry
    client = make_client()
    desc = "\n".join(f"- {f}: {SCHEMA_FIELDS.get(f, '')}" for f in fields)
    prompt = VERIFY_PROMPT.format(n=len(members), fields=desc,
                                  previous=json.dumps(previous or {}, indent=1))
    contents = [prompt]
    for m in members:
        contents.append(f"\n=== INVOICE {m['name']} : PDF ===")
        try:
            blob = open(m["pdf_path"], "rb").read()
            contents.append(types.Part.from_bytes(data=blob, mime_type="application/pdf"))
        except Exception:
            contents.append("(PDF not available — use the text)")
        contents.append(f"=== INVOICE {m['name']} : extracted TEXT ===\n{m['raw']}\n"
                        f"=== END OF INVOICE {m['name']} ===")
    resp = generate_with_retry(client, model=model_name, contents=contents,
                               config=json_config(types, think=think), purpose="verify")
    data = (_extract_json(resp.text or "").get("invoices") or {})
    out = {}
    for m in members:
        rec = data.get(m["name"]) or {}
        vals = {}
        for f in fields:
            r = rec.get(f)
            vals[f] = r.get("value") if isinstance(r, dict) else r
        out[m["name"]] = vals
    return out