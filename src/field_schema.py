"""
field_schema.py  —  SINGLE SOURCE OF TRUTH for the 14 canonical header fields.

Before this file existed, the same per-field metadata (datatype, scope, priority,
required, validation) was hardcoded independently in THREE places:
    - src/discover.py   SCHEMA_FIELDS       (field -> description, for the discovery prompt)
    - src/generate.py   PROMPT              (a literal "FIXED schema values" table, for the
                                              regex-generation prompt)
    - src/assemble.py   FIELD_TEMPLATE      (field -> metadata, used to validate/correct
                                              whatever Gemini returns)

Those three copies could silently drift apart (e.g. someone changes a field's
`priority` in assemble.py but forgets generate.py, and the prompt now tells Gemini
something that doesn't match what the pipeline will actually enforce). This module
defines each field exactly once; the three files above now import from here instead
of hardcoding their own copy.

Nothing about the ACTUAL PIPELINE BEHAVIOR changes — discover.py, generate.py, and
assemble.py still do exactly what they did before. Only where the metadata values
live has changed: one dataclass per field, in one dict, in one file.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FieldSpec:
    name: str                                  # canonical field key, e.g. "gstin_number"
    description: str                           # long-form guidance for discover's DISCOVER_PROMPT
    datatype: str                               # "amount" | "gstin" | "date" | "percentage" | "string"
    scope: str                                  # "vendor" | "invoice"
    priority: int                               # 1-10
    required: bool
    validation: List[str] = field(default_factory=list)
    default_prompt: List[str] = field(default_factory=list)        # fallback "prompt" hint text
    default_negative_alias: List[str] = field(default_factory=list)  # only gstin_number uses this today


# Order here IS the canonical field order (CANONICAL_ORDER in assemble.py used to be a
# separately maintained list — it is now just list(FIELD_SCHEMA.keys())).
FIELD_SCHEMA = {
    "document_type": FieldSpec(
        name="document_type",
        description=("The document TITLE at the top. One of: 'TAX INVOICE', 'GST INVOICE', "
                      "'CREDIT NOTE', 'DEBIT NOTE'. (A credit note also carries a "
                      "'CREDIT NOTE NO' label.)"),
        datatype="string", scope="invoice", priority=10, required=True,
        validation=["non_empty"],
        default_prompt=["Look for the document type at the top."],
    ),
    "gstin_number": FieldSpec(
        name="gstin_number",
        description=("The SUPPLIER's 15-char GSTIN — the seller near the top/supplier name. "
                      "Labels: 'GSTIN:', 'GSTN NO.', 'GST No.:', 'GSTIN/UIN'. NOT the "
                      "buyer/consignee GSTIN (under 'Bill To', 'Ship To', 'Detail of Receiver', "
                      "'Buyer', 'Consignee')."),
        datatype="gstin", scope="vendor", priority=10, required=True,
        validation=["gstin", "gstin_checksum"],
        default_prompt=["Find the GSTIN associated with the supplier's address at the top."],
        default_negative_alias=["Ship-From", "Bill To", "Ship To"],
    ),
    "po_pds_number": FieldSpec(
        name="po_pds_number",
        description=("The Purchase Order / PDS number — PO and PDS are the SAME field here, "
                      "just different labels on different vendors. Capture whichever appears. "
                      "Labels (all mean this field): 'Purchase Order', 'PO No', 'PO NO:', "
                      "'P.O NO:', 'P.O. No.', 'PO Number', 'PDS No', 'PDS NO.', 'PDS Number', "
                      "'Order No'. If BOTH a PO and a PDS number are printed, prefer the PO "
                      "number. Value is the number, not the date. Do NOT report the e-invoice "
                      "'Ack No' (a ~15-digit acknowledgement number), the IRN, an E-way bill "
                      "number or the invoice number as the PO. The text is linearised from a "
                      "PDF, so a label can be followed by values that belong to OTHER nearby "
                      "labels (e.g. 'PO. No. & Date :' followed by the IRN, the Ack No and the "
                      "Ack Date) — pick the value whose format and position fit a PO; the PO "
                      "often also appears in a 'PO No.' column of the line items."),
        datatype="string", scope="invoice", priority=5, required=False,
        validation=[],
    ),
    "order_date": FieldSpec(
        name="order_date",
        description=("The purchase-order date (a DATE). Labels (all mean this): 'Order Date', "
                      "'PO Date', 'P.O DATE', 'P.O.DATE', 'Purchase Order Date', or 'Dtd:'/'Dt:' "
                      "shown next to the PO number. NOT the e-invoice 'Ack Date' and NOT the "
                      "invoice date. If the only PO date printed is a placeholder such as "
                      "30-12-1899 or 01-01-1900, order_date is ABSENT."),
        datatype="date", scope="invoice", priority=5, required=False,
        validation=[],
    ),
    "vendor_code": FieldSpec(
        name="vendor_code",
        description=("The supplier/vendor code, when the invoice has a 'Vendor Code' / "
                      "'Supplier Code' / 'Supplier ID' label with a value next to it (e.g. "
                      "'Vendor Code : D002' -> D002). Report that value. Do NOT confuse it with "
                      "a longer code it may be a prefix of (e.g. a part number or DI/E-Nagare "
                      "number like 'D002AA1G8028'); report the standalone vendor code only."),
        datatype="string", scope="invoice", priority=5, required=False,
        validation=[],
    ),
    "supplier_document_number": FieldSpec(
        name="supplier_document_number",
        description=("The document number of THIS invoice/note. Labels: 'Invoice No', "
                      "'Invoice Number', 'INVOICE NO :', 'INVOICE NO.:', 'Bill No', "
                      "'Document No'. On a CREDIT NOTE it is the 'CREDIT NOTE NO' value."),
        datatype="string", scope="invoice", priority=10, required=True,
        validation=["non_empty"],
        default_prompt=["Find the supplier invoice number near the top supplier details."],
    ),
    "supplier_document_date": FieldSpec(
        name="supplier_document_date",
        description=("The invoice/document date (a DATE). Labels (all mean this): "
                      "'Invoice Date', 'Date of Invoice', 'Date of Issue', 'DATE OF ISSUE', "
                      "'Date & Time of Issue of Invoice'."),
        datatype="date", scope="invoice", priority=10, required=True,
        validation=["date"],
        default_prompt=["Look for the invoice date near the top / IRN number."],
    ),
    "invoice_total_basic_value": FieldSpec(
        name="invoice_total_basic_value",
        description=("The 'Total Sales Amount' column TOTAL (pre-tax sales, before "
                      "assessable/taxable adjustments). A DIFFERENT column from 'Taxable "
                      "Value'. Only if a 'Total Sales Amount' / 'Basic Amount' column "
                      "literally exists. 'Transaction Value' and 'Sub Total' are NOT this "
                      "field: they are the taxable value."),
        datatype="amount", scope="invoice", priority=8, required=False,
        validation=["numeric"],
        default_prompt=["Total sales amount in the summary row."],
    ),
    "total_invoice_taxable_value": FieldSpec(
        name="total_invoice_taxable_value",
        description=("The TAXABLE value TOTAL from the summary row. This field has MANY "
                      "label variants that ALL mean the same thing — treat them as equal: "
                      "'Taxable Value', 'Assessable Value', 'Asse. Value', 'Asse Value', "
                      "'Assess. Value', 'TOTAL TAXABLE VALUE', 'Total Assessable Value', "
                      "'Transaction Value', 'Taxable Amount', and a pre-tax 'Sub Total' "
                      "printed just above the tax totals. It is the amount the GST is charged "
                      "on: taxable + IGST/CGST/SGST = invoice total (unless amortisation is "
                      "involved). It is NEVER a rate or percentage such as 18.00 from a "
                      "'GST %' column, and it is always larger than the tax amount. The label "
                      "may be an ABBREVIATION (e.g. 'Asse.') and may be a COLUMN HEADER far "
                      "from the value, with the total sitting in a summary/total row. "
                      "Different column from 'Total Sales Amount' — read the header."),
        datatype="amount", scope="invoice", priority=8, required=False,
        validation=["numeric"],
        default_prompt=["Taxable value total in the summary row."],
    ),
    "total_invoice_igst_tax_value": FieldSpec(
        name="total_invoice_igst_tax_value",
        description=("The IGST amount TOTAL (summary row). Labels: 'IGST Amount', "
                      "'TOTAL IGST VALUE', 'IGST Value'. Often 0 on intra-state invoices."),
        datatype="amount", scope="invoice", priority=9, required=False,
        validation=["numeric"],
        default_prompt=["IGST amount total in the summary row."],
    ),
    "total_invoice_cgst_tax_value": FieldSpec(
        name="total_invoice_cgst_tax_value",
        description=("The CGST amount TOTAL (summary row). Labels: 'CGST Amount', "
                      "'TOTAL CGST VALUE', 'CGST Value Rs.', 'CGST'. Take the SUMMARY total, "
                      "not a single line item."),
        datatype="amount", scope="invoice", priority=9, required=False,
        validation=["numeric"],
        default_prompt=["CGST amount total in the summary row."],
    ),
    "total_invoice_sgst_tax_value": FieldSpec(
        name="total_invoice_sgst_tax_value",
        description=("The SGST/UTGST amount TOTAL (summary row). Labels: 'SGST Amount', "
                      "'SGST/UTGST Amount', 'TOTAL SGST VALUE', 'SGST Value Rs.'. Frequently "
                      "EQUAL to CGST — still read it from its own column."),
        datatype="amount", scope="invoice", priority=9, required=False,
        validation=["numeric"],
        default_prompt=["SGST amount total in the summary row."],
    ),
    "total_invoice_cess_tax_value": FieldSpec(
        name="total_invoice_cess_tax_value",
        description=("The CESS/TCS amount TOTAL (summary row). Labels: 'CESS Amount', "
                      "'CESS / TCS Amount', 'TOTAL TCS VALUE', 'Comp Cess'. Often 0. Do NOT "
                      "confuse with amortization/rounding amounts."),
        datatype="amount", scope="invoice", priority=8, required=False,
        validation=["numeric"],
        default_prompt=["CESS/TCS amount total in the summary row."],
    ),
    "total_invoice_value": FieldSpec(
        name="total_invoice_value",
        description=("The final GRAND TOTAL of the invoice (usually the largest amount, at "
                      "the bottom). Labels: 'Total Invoice Value (in figures)', "
                      "'Total Invoice Value (In Figure)', 'TOTAL AMOUNT', 'Invoice Value', "
                      "'Grand Total'."),
        datatype="amount", scope="invoice", priority=10, required=True,
        validation=["numeric"],
        default_prompt=["Find 'Total Invoice Value (in figures)' and capture the number."],
    ),
    "irn_number": FieldSpec(
        name="irn_number",
        description=("The 64-character hexadecimal IRN. Labels: 'IRN NO.:', 'IRN :', "
                      "'IRN No.'. It may be split across two lines — join them into 64 hex "
                      "chars."),
        datatype="string", scope="invoice", priority=10, required=False,
        validation=["non_empty"],
        default_prompt=["Find 'IRN NO.:' near the top."],
    ),
}

# Preserves the exact same ordering the old, separately-maintained CANONICAL_ORDER list had.
CANONICAL_ORDER = list(FIELD_SCHEMA.keys())

_DATATYPES = {"amount", "gstin", "date", "percentage", "string"}


def discover_schema_text() -> str:
    """The '- field: description' bullet list discover.py embeds in DISCOVER_PROMPT."""
    return "\n".join(f"- {f.name}: {f.description}" for f in FIELD_SCHEMA.values())


def generate_schema_table_text() -> str:
    """The 'FIXED schema values per field' table generate.py embeds in PROMPT."""
    lines = []
    for f in FIELD_SCHEMA.values():
        extra = ""
        if f.default_negative_alias:
            extra = f", negative_alias={f.default_negative_alias!r}"
        lines.append(
            f"  {f.name + ':':30s} datatype={f.datatype:10s} scope={f.scope:8s} "
            f"priority={f.priority:<3d} required={str(f.required).lower():5s} "
            f"validation={f.validation!r}{extra}"
        )
    return "\n".join(lines)


def validate_and_correct(name: str, spec: dict) -> dict:
    """What assemble.py's _field_block used to do inline against its own local
    FIELD_TEMPLATE dict: keep Gemini's value for a fixed-schema key only if it's
    valid, otherwise fall back to this field's canonical value. Returns the
    corrected (datatype, scope, priority, required, validation, prompt) tuple-as-dict;
    the invoice-specific keys (alias/direction/regex/fallback_alias/negative_alias)
    are NOT handled here — assemble.py still owns those, since they're per-invoice,
    not schema metadata.
    """
    tpl = FIELD_SCHEMA.get(name)
    spec = spec or {}
    if tpl is None:
        # Unknown field name — nothing to validate against; behave like the old code's
        # tpl = {} fallback (defaults baked in below).
        datatype = spec.get("datatype") if spec.get("datatype") in _DATATYPES else "string"
        return {
            "datatype": datatype,
            "scope": spec.get("scope") if spec.get("scope") in ("vendor", "invoice") else "invoice",
            "priority": spec["priority"] if isinstance(spec.get("priority"), int) and 1 <= spec["priority"] <= 10 else 5,
            "required": spec["required"] if isinstance(spec.get("required"), bool) else False,
            "validation": spec["validation"] if isinstance(spec.get("validation"), list) else [],
            "prompt": spec["prompt"] if isinstance(spec.get("prompt"), list) and spec.get("prompt") else [],
            "default_negative_alias": [],
        }

    datatype = spec.get("datatype")
    if datatype not in _DATATYPES:
        datatype = tpl.datatype
    scope = spec.get("scope") if spec.get("scope") in ("vendor", "invoice") else tpl.scope
    priority = spec.get("priority")
    if not (isinstance(priority, int) and 1 <= priority <= 10):
        priority = tpl.priority
    required = spec.get("required")
    if not isinstance(required, bool):
        required = tpl.required
    validation = spec.get("validation")
    if not isinstance(validation, list):
        validation = list(tpl.validation)
    prompt = spec.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        prompt = list(tpl.default_prompt)

    return {
        "datatype": datatype, "scope": scope, "priority": priority, "required": required,
        "validation": validation, "prompt": prompt,
        "default_negative_alias": list(tpl.default_negative_alias),
    }