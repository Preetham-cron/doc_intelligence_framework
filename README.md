# Invoice Extraction Config Agent (LangGraph)

Evidence-grounded agent: reads each invoice's RAW TEXT, finds which schema fields
are ACTUALLY present, and generates regex extraction rules only for those — in the
old config format, versioned by layout (V1, V2, ...).

## Core rule (strict)
    Schema ∩ Evidence-in-Raw-Text = Extractable fields
  - Raw text is the ONLY source of truth. QR is used only to CROSS-CHECK.
  - Never infer, calculate, derive, split, normalize, or hallucinate a value.
  - A hallucination guard drops any value not found verbatim in the raw text.
  - Fields with no explicit evidence get NO rule.

## Flow (LangGraph nodes)
    upload -> ocr -> qr -> discover -> generate -> test -> [retry] -> crosscheck -> assemble
  - ocr / qr : the EXACT main-pipeline code (src/markdownextractor, src/qrcode)
  - discover : LLM finds evidence-backed present fields + values (raw text only)
  - generate : LLM writes regex; test validates it extracts the discovered value
  - crosscheck: compares to QR for confidence (never changes decisions)
  - assemble : old config format; ONLY present fields filled

## Versioning
  Same layout (fixed-vocabulary fingerprint) -> reuse the same version (no LLM).
  New/different layout for a vendor -> new version V1, V2, ...

## Setup
    pip install -r requirements.txt
    # key in .envs/.local :  GEMINI_API_KEY=your_key_here
    # pdf2image needs poppler; pyzbar needs zbar.

## Run
    # put invoice PDFs in  invoices/
    python main.py
    # configs written to  output/<GSTIN>_<DOCTYPE>_V<n>.json

## Knowledge base (self-correction at scale)
The agent stores every SOLVED template in knowledge_base.json, keyed by
GSTIN:DOCTYPE:layout_hash.
- New template  -> the agent solves it (LLM) and SAVES the working config.
- Known template -> REUSED from the KB with NO LLM call.
So each of your 1000s of templates is solved once and remembered forever.
Delete knowledge_base.json to force a full rebuild.

## Review queue
A template is stored in the KB ONLY when ALL present fields resolve. If any field
is unresolved, the partial config is written to  review/<gstin>_<doctype>_<hash>.json
(with the list of unresolved fields) and is NOT remembered as broken.

## One command, settings in `.envs/.local`

    python main.py

`INVOICE_SOURCE` in `.envs/.local` decides what it does:

| Setting | What happens |
|---|---|
| `INVOICE_SOURCE=database` | Invoices come from `invoicecheck_invoiceextracted` + Azure, 5 GST numbers per batch; solved JSON is stored in `master_vendorocrprompt.rscp_prompt`, unsolved ones go to `review_db/` |
| `INVOICE_SOURCE=folder` | Invoices come from the local `invoices/` folder; JSON is written to `output/` (`review/` for unsolved) |
| `GST_NUMBERS=A,B` | Database mode: build only these GST numbers |
| `MAX_BATCHES=1` | Database mode: stop after one batch (trial run) |
| `CONFIG_OVERWRITE=1` | Database mode: rebuild GST numbers that already have a JSON |
| `REPAIR=1` | Folder mode: re-work only the templates in `review/` |

All other options are listed in `.envs/.local.example`. There is no `run_db.py` any more.
