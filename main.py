"""
main.py  —  the ONE entry point: build per-vendor JSON configs from invoices.

    python main.py

What it does is set in .envs/.local (see .envs/.local.example):
  INVOICE_SOURCE=database  -> invoices from the database + Azure, 5 GST numbers per batch,
                              JSON stored in master_vendorocrprompt.rscp_prompt
  INVOICE_SOURCE=folder    -> invoices from the local invoices/ folder, JSON written to
                              output/ (review/ for unsolved), remembered in knowledge_base.json

How a config is built (both modes):
  - Each invoice -> text + QR + layout_hash (cheap).
  - Group by (gstin, doctype, layout_hash) = template.
  - A template already in the knowledge base is REUSED (no LLM).
  - A NEW one is solved by the agent and validated on every invoice of that template.
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# settings first: everything below reads its options from .envs/.local
from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, ".envs", ".local"), override=False)

from src.pipeline_text import pipeline_text
from src.qrcode.qr_extractor import QRCodeExtractor
from src.fingerprint import layout_hash
from src.kb import (load_kb, save_kb, kb_get, kb_put, kb_bump_seen,
                    kb_versions_for, kb_configs_for_gstin)
from graph import AGENT, _doc_type
from src import usage

INPUT_DIR  = os.path.join(HERE, "invoices")
OUTPUT_DIR = os.path.join(HERE, "output")
_QR = QRCodeExtractor()

# A QR code is decoded once per PDF: choosing the sample invoices already reads it, and
# the agent would otherwise read the same PDFs again (1-3 seconds each).
_QR_CACHE = {}
_qr_uncached = QRCodeExtractor.extract_from_blob


def _qr_cached(self, blob_data, invoice_id="?", *args, **kwargs):
    import hashlib
    key = hashlib.sha1(blob_data).hexdigest()
    if key not in _QR_CACHE:
        if len(_QR_CACHE) > 500:
            _QR_CACHE.clear()
        _QR_CACHE[key] = _qr_uncached(self, blob_data, invoice_id, *args, **kwargs)
    return _QR_CACHE[key]


QRCodeExtractor.extract_from_blob = _qr_cached
REVIEW_DIR = os.path.join(HERE, "review")   # templates not fully solved land here


class _ThreadOut:
    """stdout router: while a worker thread has begun(), its prints go to its own
    buffer; everything else goes straight to the console. Lets templates build in
    parallel while each one's log is printed as a single, readable block."""
    def __init__(self, real):
        import threading
        self.real, self.local = real, threading.local()

    def begin(self):
        import io
        self.local.buf = io.StringIO()

    def end(self):
        buf = getattr(self.local, "buf", None)
        self.local.buf = None
        return buf.getvalue() if buf else ""

    def write(self, s):
        buf = getattr(self.local, "buf", None)
        return (buf if buf is not None else self.real).write(s)

    def flush(self):
        if getattr(self.local, "buf", None) is None:
            self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


_OUT = _ThreadOut(sys.stdout)
sys.stdout = _OUT


def _decide(job, final, err, kb, vendor_review):
    """Main thread only: store a solved template in the KB, or file it for review."""
    gstin, doc_type, lhash = job["gstin"], job["doc_type"], job["lhash"]
    members, version, cfg_key = job["members"], job["version"], job["cfg_key"]
    primary = members[0]
    if err:
        vendor_review.setdefault(gstin, {})[cfg_key] = {
            "config": None,
            "review": {"doc_type": doc_type, "layout_hash": lhash, "sample": primary["name"],
                       "status": "error", "error": err[0], "trace": err[1],
                       "raw_text_files": _dump_raw(gstin, doc_type, lhash, members)},
        }
        print(f"   -> ERROR building {cfg_key}: {err[0]}; continuing")
        return

    inner = list(final["config"].values())[0]
    # does this layout carry an e-invoice QR code? (first key of the block)
    inner = {"qr_code": qr_code_flag(members), **{k: v for k, v in inner.items() if k != "qr_code"}}
    filled = len(final.get("passing", {}))
    total = len(final.get("present", {})) or 1
    unresolved = list(final.get("pending", []))

    # BEST-EFFORT fields: label-detached on some layouts; their absence must NOT block
    # an otherwise-good config (irn is required=false in the schema).
    BEST_EFFORT = {"po_pds_number", "vendor_code", "order_date", "irn_number"}
    blocking = [f for f in unresolved if f not in BEST_EFFORT]
    # CORE fields every usable config must extract — an empty config (unreadable /
    # garbled PDF) must never be stored as solved.
    passing_now = final.get("passing", {}) or {}
    # REPAIR safety net: a required field that was unresolved in review must end up
    # with a VALIDATED rule — it may never be "solved" by disappearing
    for f in job.get("reverify") or []:
        if f not in BEST_EFFORT and f not in passing_now and f not in blocking:
            blocking.append(f)
            if f not in unresolved:
                unresolved.append(f)
    for f in ("gstin_number", "supplier_document_number", "total_invoice_value"):
        if f not in passing_now and f not in blocking:
            blocking.append(f)
            if f not in unresolved:
                unresolved.append(f)
    # an invoice that carries a tax amount always has a taxable value: a config that
    # extracts the tax but not the taxable value is not complete
    _TAXES = ("total_invoice_igst_tax_value", "total_invoice_cgst_tax_value",
              "total_invoice_sgst_tax_value")
    f = "total_invoice_taxable_value"
    if any(t in passing_now for t in _TAXES) and f not in passing_now and f not in blocking:
        blocking.append(f)
        if f not in unresolved:
            unresolved.append(f)

    if blocking and job.get("from_kb"):
        # repairing an OPTIONAL gap of a working stored template: never demote it
        print(f"   -> gap repair did not produce a valid config ({blocking}); the stored "
              f"config is kept unchanged")
        return
    if not blocking:
        vendor_review.get(gstin, {}).pop(cfg_key, None)     # drop a stale review entry (repair)
        kb_put(kb, gstin, doc_type, lhash, version, cfg_key, inner, filled, total)
        # remember optional fields left empty, so `--repair` can re-read them from the PDFs
        kb_get(kb, gstin, doc_type, lhash)["unresolved"] = list(unresolved)
        note = "none pending" if not unresolved else f"best-effort left empty: {unresolved}"
        print(f"   -> stored in KB ({filled} fields resolved; {note})")
        if unresolved:
            vendor_review.setdefault(gstin, {})[cfg_key] = {
                "config": None,
                "review": {"doc_type": doc_type, "layout_hash": lhash, "sample": primary["name"],
                           "status": "stored", "unresolved_best_effort": unresolved},
            }
    else:
        # partial -> NOT stored in KB (never remember broken); filed in the vendor's
        # single review file WITH its partial config, expected values and raw text.
        trusted = final.get("trusted", {}) or {}
        vendor_review.setdefault(gstin, {})[cfg_key] = {
            "config": inner,
            "review": {"doc_type": doc_type, "layout_hash": lhash, "sample": primary["name"],
                       "status": "partial", "fields_resolved": filled, "fields_present": total,
                       "unresolved_fields": unresolved, "blocking_fields": blocking,
                       "expected_values": {f: trusted.get(f) for f in unresolved},
                       "invoices": [m["name"] for m in members],
                       "raw_text_files": _dump_raw(gstin, doc_type, lhash, members)},
        }
        print(f"   -> NOT stored ({filled}/{total} resolved; blocking={blocking})")
        print(f"      flagged in review/{gstin}.json  (raw text: review/raw/)")


def _dump_raw(gstin, doc_type, lhash, members):
    """Write the exact regex text of each invoice in an unsolved template to
    review/raw/, so the missing field can be fixed by looking at the real text
    (the same text production runs regexes on). Returns the relative paths."""
    out_dir = os.path.join(REVIEW_DIR, "raw")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for m in members:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in m["name"])
        fn = f"{gstin}_{doc_type}_{lhash}_{safe}.txt"
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            f.write(m["raw"])
        paths.append(f"review/raw/{fn}")
    return paths


def write_vendor_files(kb, vendor_review, gstin):
    """ONE file per vendor, in ONE place (never split across output AND review): fully
    solved -> output/GSTIN.json; any template needing attention -> review/GSTIN.json,
    which then carries ALL the vendor's templates (solved ones included). Best-effort
    gaps (po/order_date/irn/vendor_code) do not send a vendor to review."""
    solved = kb_configs_for_gstin(kb, gstin)
    rev = vendor_review.get(gstin) or {}
    needs = {k: v for k, v in rev.items()
             if (v.get("review") or {}).get("status") in ("partial", "error")}
    out_path = os.path.join(OUTPUT_DIR, f"{gstin}.json")
    rev_path = os.path.join(REVIEW_DIR, f"{gstin}.json")
    if needs:
        os.makedirs(REVIEW_DIR, exist_ok=True)
        templates = {k: {"status": "solved", "config": inner} for k, inner in solved.items()}
        templates.update(rev)
        json.dump({"gstin": gstin, "templates": templates},
                  open(rev_path, "w", encoding="utf-8"), indent=4)
        if os.path.exists(out_path):
            os.remove(out_path)
        return (f"  review/{gstin}.json ->  {len(templates)} template(s) "
                f"({len(solved)} solved + {len(needs)} need attention): {sorted(templates)}")
    if solved:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        json.dump(solved, open(out_path, "w", encoding="utf-8"), indent=4)
        if os.path.exists(rev_path):
            os.remove(rev_path)
        return f"  output/{gstin}.json  ->  {len(solved)} solved template(s): {sorted(solved)}"
    return None


SCANNED_SKIPPED = []     # invoice names skipped by the last scan (no text layer)
GSTIN_HINT = {}          # invoice name -> GSTIN known from the database (used when the
                         # invoice has no QR code, so it is not lost under "UNKNOWN")
MAX_VERSIONS = int(os.environ.get("MAX_VERSIONS", "10"))   # V1..V10 per vendor + doc type


def qr_code_flag(members) -> str:
    """'Yes' when the invoices of this layout carry a readable e-invoice QR code."""
    with_qr = sum(1 for m in members if (m.get("qr") or {}).get("seller_gstin"))
    return "Yes" if with_qr * 2 >= len(members) and with_qr else "No"


def scan_invoices(pdfs):
    """Text + QR + template key for each PDF, grouped by template
    (seller GSTIN, doc type, layout hash). Sequential on purpose: PDF libraries are
    not thread-safe."""
    groups = {}
    SCANNED_SKIPPED.clear()
    skip_scanned = os.environ.get("SKIP_SCANNED", "1") != "0"
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        blob = open(pdf, "rb").read()
        # the SAME text the primary is validated on (and production extracts from),
        # so sibling generalization checks compare like with like
        raw, scanned, _md = pipeline_text(blob)
        # A scanned / image-only PDF has no text layer: a regex config cannot work on it
        # (production also skips the regex level for scanned invoices), so it is not
        # sent to Gemini at all.
        if skip_scanned and (scanned or len((raw or "").strip()) < 50):
            SCANNED_SKIPPED.append(name)
            continue
        qr = _QR.extract_from_blob(blob, invoice_id=name) or {}
        gstin = qr.get("seller_gstin") or GSTIN_HINT.get(name) or gstin_from_text(raw) or "UNKNOWN"
        key = (gstin, _doc_type(raw), layout_hash(raw))
        groups.setdefault(key, []).append({"name": name, "path": pdf, "raw": raw,
                                           "qr": qr, "scanned": scanned})
    return groups


def load_review_templates():
    """Templates waiting in review/: {(gstin, doc_type, layout_hash): {...}} plus the
    raw per-vendor entries (to carry over anything the repair run does not rebuild)."""
    todo, carried = {}, {}
    if not os.path.isdir(REVIEW_DIR):
        return todo, carried
    for fn in sorted(os.listdir(REVIEW_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(REVIEW_DIR, fn), encoding="utf-8"))
        except Exception:
            continue
        gstin = data.get("gstin") or fn[:-5]
        for cfg_key, entry in (data.get("templates") or {}).items():
            rv = entry.get("review") or {}
            if entry.get("status") == "solved":
                continue                                  # lives in the KB already
            carried.setdefault(gstin, {})[cfg_key] = entry
            if rv.get("status") in ("partial", "error") and rv.get("layout_hash"):
                parts = cfg_key.split(":")
                todo[(gstin, rv.get("doc_type") or parts[1], rv["layout_hash"])] = {
                    "cfg_key": cfg_key, "version": parts[-1],
                    "config": (entry.get("config") or {}).get("header", {}),
                    "reverify": list(rv.get("unresolved_fields") or []),
                }
    return todo, carried


# stronger settings for the repair pass (only the review templates, only their
# unsolved fields): a stronger model, more attempts, thinking on every attempt
_BEST_EFFORT = ("po_pds_number", "vendor_code", "order_date", "irn_number")


def load_kb_gap_templates(kb):
    """Stored (solved) templates whose OPTIONAL fields were left empty — e.g. a PO the
    build could not pin down. `--repair` re-reads just those fields from the PDFs.
    (For KB entries written before gaps were recorded, a best-effort field without
    any regex/alias is treated as a gap once; the repair then records the outcome.)"""
    out = {}
    for e in kb.values():
        header = (e.get("config") or {}).get("header", {})
        if "unresolved" in e:
            gaps = [f for f in e["unresolved"] if f in header]
        else:
            gaps = [f for f in _BEST_EFFORT
                    if f in header and not (header[f].get("regex") or header[f].get("alias"))]
        if gaps:
            out[(e["gstin"], e["doc_type"], e["layout_hash"])] = {
                "cfg_key": e["config_key"], "version": e["version"], "config": header,
                "reverify": gaps, "from_kb": True}
    return out


REPAIR_SETTINGS = {
    "model": os.environ.get("REPAIR_MODEL", "gemini-2.5-pro"),
    "max_retries": int(os.environ.get("REPAIR_RETRIES", "4")),
    "think_first": 4096, "think_retry": 4096,
    "verify_model": os.environ.get("VERIFY_MODEL", "gemini-2.5-pro"),
    "positional_first": True,
}


def _list_pdfs(folder):
    return sorted({p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(".pdf")})


def run_folder(repair=False):
    os.makedirs(INPUT_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdfs = _list_pdfs(INPUT_DIR)
    if not pdfs:
        print(f"No PDFs. Put invoices in: {INPUT_DIR}"); return

    kb = load_kb()
    review_todo, review_carried = load_review_templates() if repair else ({}, {})
    if repair:
        for key, t in load_kb_gap_templates(kb).items():
            review_todo.setdefault(key, t)
    if repair:
        n_kb = sum(1 for t in review_todo.values() if t.get("from_kb"))
        print(f"REPAIR PASS: {len(review_todo) - n_kb} template(s) in review/ + {n_kb} stored "
              f"template(s) with empty optional fields | model "
              f"{REPAIR_SETTINGS['model']}, up to {REPAIR_SETTINGS['max_retries']+1} attempts\n")
        if not review_todo:
            print("Nothing to repair (review/ is empty and no stored template has gaps)."); return
    print(f"Processing {len(pdfs)} invoice(s)   |   knowledge base: {len(kb)} known template(s)\n")

    # --- Phase 1: identity + layout, then group ---
    groups = scan_invoices(pdfs)
    # most common layout first, so it gets V1, the next V2, ...
    groups = dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))
    if SCANNED_SKIPPED:
        print(f"Skipped {len(SCANNED_SKIPPED)} scanned / image-only invoice(s) with no text "
              f"layer (a regex config cannot work on them).")

    print(f"Found {len(groups)} distinct template(s):")
    for (g, dt, h), members in groups.items():
        known = "IN KB (reuse)" if kb_get(kb, g, dt, h) else "NEW (will build)"
        print(f"  {g} [{dt}] layout {h}: {len(members)} invoice(s)  -> {known}")
    print()

    touched_gstins = set()
    vendor_review = {}   # gstin -> {cfg_key: {"config":inner, "review":{...}}}  (ONE per vendor)
    if repair:           # keep review entries this run does not rebuild (e.g. PDFs removed)
        for g, entries in review_carried.items():
            vendor_review[g] = dict(entries)
            touched_gstins.add(g)
    local_ver = {}       # (gstin, doc_type) -> highest Vn used so far (KB + this run)

    def _next_version(gstin, doc_type):
        # span BOTH the KB (already-solved) and templates built earlier in THIS run,
        # so two unsolved layouts of the same vendor never collide on V1/V1.
        base = local_ver.get((gstin, doc_type))
        if base is None:
            nums = [int(str(x).lstrip("Vv")) for x in kb_versions_for(kb, gstin, doc_type).values()
                    if str(x).lstrip("Vv").isdigit()]
            base = max(nums, default=0)
        n = base + 1
        local_ver[(gstin, doc_type)] = n
        return f"V{n}"

    # ---- Phase 2a (sequential): KB hits, version numbers, build the job list ----
    jobs = []
    for (gstin, doc_type, lhash), members in groups.items():
        if repair and (gstin, doc_type, lhash) not in review_todo:
            continue                               # repair touches review templates only
        touched_gstins.add(gstin)
        entry = kb_get(kb, gstin, doc_type, lhash)
        if entry and not (repair and (gstin, doc_type, lhash) in review_todo):   # KB HIT: reuse
            kb_bump_seen(kb, gstin, doc_type, lhash)
            try:                                   # keep the version counter above reused ones
                vn = int(str(entry.get("version", "V0")).lstrip("V"))
                local_ver[(gstin, doc_type)] = max(local_ver.get((gstin, doc_type), 0), vn)
            except ValueError:
                pass
            print(f"[{entry['config_key']}] REUSED FROM KB  (seen {entry['seen_count']}x, no LLM)\n")
            continue
        rt = review_todo.get((gstin, doc_type, lhash)) if repair else None
        # repair keeps the version the template already has in its review file
        version = rt["version"] if rt else _next_version(gstin, doc_type)
        if not rt and int(version.lstrip("V")) > MAX_VERSIONS:
            local_ver[(gstin, doc_type)] = MAX_VERSIONS
            print(f"[{gstin}:{doc_type}] layout {lhash} ({len(members)} invoice(s)) skipped: "
                  f"this vendor already has {MAX_VERSIONS} versions (V{MAX_VERSIONS} is the limit)\n")
            continue
        # UNION of tax fields across ALL members (handles IGST-on-some, CGST/SGST-on-others)
        from src.autofallback import find_tax_value
        forced = {}
        for fld in ("total_invoice_taxable_value", "total_invoice_cgst_tax_value",
                    "total_invoice_sgst_tax_value", "total_invoice_igst_tax_value",
                    "total_invoice_cess_tax_value"):
            for m in members:
                hit = find_tax_value(fld, m["raw"])
                if hit:
                    forced[fld] = hit; break
        jobs.append({"gstin": gstin, "doc_type": doc_type, "lhash": lhash, "members": members,
                     "version": version, "cfg_key": f"{gstin}:{doc_type}:{version}",
                     "forced": forced,
                     "pre_passing": ({f: b for f, b in rt["config"].items()
                                      if (b or {}).get("regex") or (b or {}).get("alias")}
                                     if rt else {}),
                     "reverify": (rt.get("reverify") or []) if rt else [],
                     "from_kb": bool(rt and rt.get("from_kb"))})

    # ---- Phase 2b (PARALLEL): run the agent per template ----
    # Only Gemini calls + regex validation run in the workers (text and QR were
    # built above), so no PDF library is touched from a thread. Each template's log
    # is buffered and printed as ONE block when it finishes (no interleaving).
    run_mark = usage.mark()
    workers = max(1, int(os.environ.get("AGENT_WORKERS", "4")))
    print(f"Building {len(jobs)} new template(s) with {min(workers, max(1, len(jobs)))} "
          f"parallel worker(s)  (set AGENT_WORKERS to change)\n")

    def _run(job):
        import time, traceback
        primary, members = job["members"][0], job["members"]
        t0 = time.time()
        _OUT.begin()
        usage.set_tag(job["cfg_key"])                 # every Gemini call below counts for this template
        final, err = None, None
        try:
            print(f"[{job['cfg_key']}] {'REPAIR' if repair else 'NEW template'} -> building from {primary['name']} "
                  f"(validating on {len(members)-1} sibling(s); tax union: {sorted(job['forced'])})")
            final = AGENT.invoke({"pdf_path": primary["path"],
                                  "raw_text": primary["raw"], "is_scanned": primary["scanned"],
                                  "qr": primary["qr"],
                                  "sibling_texts": [m["raw"] for m in members[1:]],
                                  "sibling_qrs": [m["qr"] for m in members[1:]],
                                  "sibling_names": [m["name"] for m in members[1:]],
                                  "forced_present": job["forced"],
                                  "pre_passing": job.get("pre_passing") or {},
                                  "reverify": job.get("reverify") or [],
                                  "primary_name": primary["name"],
                                  "member_paths": [m["path"] for m in members],
                                  "settings": REPAIR_SETTINGS if repair else {},
                                  "deadline": t0 + float(os.environ.get(
                                      "REPAIR_DEADLINE_S" if repair else "TEMPLATE_DEADLINE_S",
                                      "600" if repair else "300"))})
        except Exception as e:                     # isolate: one bad template never kills the batch
            err = (f"{type(e).__name__}: {e}", traceback.format_exc())
        finally:
            log = _OUT.end()
        return job, final, err, log, time.time() - t0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, j) for j in jobs]
        for fut in as_completed(futures):
            job, final, err, log, secs = fut.result()
            done += 1
            sys.stdout.write(log)
            _decide(job, final, err, kb, vendor_review)
            # write results NOW (not only at the very end): a slow or stuck template
            # can never hold back the output of templates that already finished
            save_kb(kb)
            write_vendor_files(kb, vendor_review, job["gstin"])
            print(f"   [{done}/{len(jobs)} templates done — {job['cfg_key']} took {secs:.0f}s; "
                  f"files updated]\n")

    save_kb(kb)
    print("=== OUTPUT (one file per vendor) ===")
    for gstin in sorted(touched_gstins):
        line = write_vendor_files(kb, vendor_review, gstin)
        if line:
            print(line)

    print(f"\nknowledge_base.json now holds {len(kb)} template(s).")

    # ---- AUTO REPAIR: re-read the unresolved fields from the PDFs right away ----
    # (same as running `python main.py --repair`; set AUTO_REPAIR=0 to skip it)
    if not repair and os.environ.get("AUTO_REPAIR", "1") != "0":
        todo, _ = load_review_templates()
        gaps = load_kb_gap_templates(kb)
        if todo or gaps:
            print(f"\n{'=' * 60}\nAUTO REPAIR: {len(todo)} template(s) in review/ + {len(gaps)} "
                  f"stored template(s) with empty optional fields -> re-reading those fields "
                  f"from the PDFs\n{'=' * 60}\n")
            run_folder(repair=True)


# ══════════════════════════════════════════════════════════════════════════════════
# DATABASE + AZURE MODE   (INVOICE_SOURCE=database in .envs/.local)
#
#   Azure -> fetch invoices -> build the JSON -> store in master_vendorocrprompt -> next batch
#
# Invoices come from invoicecheck_invoiceextracted (GST number + file path, read only).
# Per cycle (GST_BATCH_SIZE, default 5 GST numbers):
#   1. pick the next <=5 GST numbers that still need a JSON
#   2. download up to MAX_INVOICES_PER_GSTIN invoices WITH TEXT for each (older ones are
#      looked at when the newest are scanned)
#   3. build with run_folder() in a temp folder (fresh knowledge base: the DB is the store)
#   4. fully solved -> master_vendorocrprompt.rscp_prompt ; not solved -> review_db/<GSTIN>.json
#   5. delete the temp folder, next 5 GST numbers
# ══════════════════════════════════════════════════════════════════════════════════
import csv, datetime, gc, random, shutil, tempfile
import src.kb as kbmod
from src.kb import order_blocks


def _db_source():
    """DB + Azure access (imported only in database mode, so folder mode needs neither
    the MySQL nor the Azure package)."""
    os.chdir(HERE)                  # sql_connector loads ".envs/.local" relative to CWD
    from data_source.vendor_config_source import VendorConfigSource
    return VendorConfigSource()


CHECKPOINT = os.path.join(HERE, "config_checkpoint.csv")
REVIEW_KEEP = os.path.join(HERE, "review_db")
UNSAVED = os.path.join(HERE, "unsaved_db")      # built JSON the DB refused (kept, retried next run)
KB_DIR = os.path.join(HERE, "knowledge_base")   # database mode: one knowledge-base file per GST number


_HASH_IN_NOTE = __import__("re").compile(r"layout_hash=([0-9a-f]{6,})")


def _block_hash(block):
    """The layout fingerprint an agent-built block carries in its prompt_note."""
    for note in (block or {}).get("prompt_note") or []:
        m = _HASH_IN_NOTE.search(str(note))
        if m:
            return m.group(1)
    return None


def _split_key(key):
    """'GSTIN:TAXINVOICE:V3' -> ('GSTIN:TAXINVOICE', 3); None for other keys."""
    parts = str(key).split(":")
    if len(parts) == 3 and parts[2][:1] in "Vv" and parts[2][1:].isdigit():
        return f"{parts[0]}:{parts[1]}", int(parts[2][1:])
    return None


def align_versions(new_cfg: dict, old_cfg) -> dict:
    """Give every newly built block the RIGHT version name next to what the vendor
    already has, so nothing existing is ever replaced by a different layout:
      * same layout as an existing block (same fingerprint)  -> keeps that block's name
      * a layout the vendor did not have                     -> the next free version
    A new layout beyond V<MAX_VERSIONS> is left out (and reported)."""
    if not isinstance(old_cfg, dict) or not old_cfg:
        return new_cfg
    by_hash = {_block_hash(b): k for k, b in old_cfg.items() if _block_hash(b)}
    used = {}
    for k in old_cfg:
        sk = _split_key(k)
        if sk:
            used.setdefault(sk[0], set()).add(sk[1])
    out = {}
    for key, block in new_cfg.items():
        h = _block_hash(block)
        sk = _split_key(key)
        if h and h in by_hash:
            out[by_hash[h]] = block                      # same layout: same name
            continue
        if not sk:
            out[key] = block
            continue
        prefix, n = sk
        taken = used.setdefault(prefix, set())
        if n in taken:                                   # name belongs to another layout
            n = next((i for i in range(1, MAX_VERSIONS + 1) if i not in taken), None)
            if n is None:
                print(f"  [{prefix}] a new layout was left out: the vendor already has "
                      f"{MAX_VERSIONS} versions")
                continue
        taken.add(n)
        out[f"{prefix}:V{n}"] = block
    return out


def _kb_seed_from_db(src, v) -> dict:
    """Knowledge-base entries for the versions a GST number ALREADY has in the
    database. With them in the knowledge base, a layout the vendor already has is
    recognised and reused as it is (no Gemini call, no change), and a new layout is
    numbered after the highest existing version."""
    kb = {}
    for vid in v["vendor_ids"]:
        cfg = src.load_config(vid)[1]
        if not isinstance(cfg, dict):
            continue
        for key, block in cfg.items():
            h, sk = _block_hash(block), _split_key(key)
            if not (h and sk and isinstance(block, dict)):
                continue
            gstin, doc_type = sk[0].split(":")
            k = f"{gstin}:{doc_type}:{h}"
            if k not in kb:
                kb[k] = {"config_key": key, "gstin": gstin, "doc_type": doc_type,
                         "layout_hash": h, "version": f"V{sk[1]}", "config": block,
                         "fields_filled": 0, "fields_total": 0, "seen_count": 1,
                         "created": "", "updated": "", "unresolved": [], "from_db": True}
    return kb


def _add_new_formats() -> bool:
    """ADD_NEW_FORMATS=1: the batch run also visits GST numbers that already have a
    JSON, to add a new invoice format as the next version (existing versions stay)."""
    return os.getenv("ADD_NEW_FORMATS", "0") == "1"


def _ids_to_write(src, vendor_ids, touch_existing):
    """Which vendor ids of a GST number are worked on: all of them when existing
    JSONs may be extended (new formats / rebuild), otherwise only the ones that have
    no JSON yet."""
    return [i for i in vendor_ids if touch_existing or not src.load_config(i)[1]]


def _store_for_all(src, v, cfg, overwrite):
    """Store the JSON of one GST number against EVERY vendor id it belongs to.
    A vendor id that already has a JSON is MERGED, never replaced: its existing
    versions stay, a layout it already has keeps its version name, and a new layout
    is added as the next version (with the vendor's line items / prompts inherited).
    A row whose JSON would not change is not written. Returns (log text, changed?)."""
    done, changed = [], False
    for vid in v["write_ids"]:
        old = src.load_config(vid)[1]
        if old:
            fresh = align_versions(cfg, old)
            if not overwrite:
                # a version this vendor id already has is left EXACTLY as it is;
                # only versions it does not have yet are added
                fresh = {k: b for k, b in fresh.items() if k not in old}
            mine = merge_config(fresh, old) if fresh else old
        else:
            mine = cfg
        if old and mine == old:
            done.append(f"{vid} no change")
            continue
        added = [k for k in mine if not old or k not in old]
        done.append(f"{vid} {src.save_config(vid, mine)}"
                    + (f" (+{', '.join(a.split(':')[-1] for a in added)})" if old and added else ""))
        changed = True
    kept = [i for i in v["vendor_ids"] if i not in v["write_ids"]]
    text = f"vendor_id(s): {', '.join(done)}"
    if kept:
        text += f" | already had a JSON, left as is: {kept}"
    return text, changed


def _kb_load_vendors(gstins) -> dict:
    """Knowledge-base entries of these GST numbers (knowledge_base/<GSTIN>.json each)."""
    kb = {}
    for g in gstins:
        f = os.path.join(KB_DIR, f"{g}.json")
        if os.path.exists(f):
            try:
                kb.update(json.load(open(f, encoding="utf-8")))
            except Exception:
                pass
    return kb


def _kb_save_vendors(kb: dict, gstins):
    """Write the entries back, one small file per GST number (never one huge file)."""
    os.makedirs(KB_DIR, exist_ok=True)
    for g in gstins:
        mine = {k: v for k, v in kb.items() if v.get("gstin") == g}
        if mine:
            json.dump(mine, open(os.path.join(KB_DIR, f"{g}.json"), "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def load_checkpoint() -> dict:
    """{gstin: last status} for every GST number processed so far (the last line of a
    GST number wins, so a review that is later stored counts as stored)."""
    seen = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    seen[row[0]] = row[1]
    return seen


def checkpoint_summary(seen: dict) -> str:
    """'37 GST number(s): stored 21, review 9, scanned 4, ...' from the last status of each."""
    if not seen:
        return "empty"
    counts = {}
    for st in seen.values():
        st = "stored" if st == "unchanged" else st     # re-checked, nothing new: still stored
        counts[st] = counts.get(st, 0) + 1
    order = ["stored", "review", "scanned", "no_invoices", "no_result", "error"]
    parts = [f"{k} {counts[k]}" for k in order if k in counts]
    parts += [f"{k} {v}" for k, v in sorted(counts.items()) if k not in order]
    return f"{len(seen)} GST number(s): " + ", ".join(parts)


def append_checkpoint(gstin, status, note=""):
    with open(CHECKPOINT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [gstin, status, datetime.datetime.now().isoformat(timespec="seconds"), note])


def next_batch(src, after, size, skip, overwrite, skipped):
    """Up to `size` GST numbers after `after` that still need a JSON. Each query asks
    the DB for at most `size` values. `skipped` counts why others were passed over.
    Returns (batch [{raw, gstin, vendor_ids, write_ids}], new_after, exhausted)."""
    batch = []
    while len(batch) < size:
        values = src.fetch_gst_numbers(after, size)
        if not values:
            return batch, after, True
        for raw in values:
            after = raw
            g = str(raw).strip().upper()
            if not g or g in skip or any(b["gstin"] == g for b in batch):
                continue
            skip.add(g)                               # decided once per run
            ids = src.vendor_ids_for(g)
            if not ids:
                skipped["not in master_vendors"] = skipped.get("not in master_vendors", 0) + 1
                continue
            write = _ids_to_write(src, ids, overwrite or _add_new_formats())
            if not write:
                skipped["already has a JSON"] = skipped.get("already has a JSON", 0) + 1
                continue
            batch.append({"raw": raw, "gstin": g, "vendor_ids": ids, "write_ids": write})
            if len(batch) == size:
                break
    return batch, after, False


def _rich_prompt(p):
    """A hand-written field prompt (not just the field name the agent puts there)."""
    return isinstance(p, list) and (len(p) > 1 or (len(p) == 1 and len(str(p[0])) > 40))


def merge_config(new_cfg: dict, old_cfg) -> dict:
    """Only used with CONFIG_OVERWRITE=1 (vendor already has a config).

    Production tries the blocks whose key carries the document type first
    (GSTIN:TAXINVOICE:V1 ...), so the agent's new blocks take over from an older
    plain-GSTIN block. The agent only builds header regexes, so each new block
    inherits from the vendor's existing config what it does not produce:
      * line_items   (the coordinate config for the line-item table)
      * prompt_note  (the vendor's Gemini instructions), kept in front of the agent's note
      * a field's hand-written "prompt", where the agent only has the field name
    Existing blocks the agent did not rebuild are kept as they are."""
    if not isinstance(old_cfg, dict):
        return new_cfg
    merged = dict(old_cfg)
    for key, block in new_cfg.items():
        block = dict(block)
        gstin = key.split(":")[0]
        donors = [old_cfg.get(key), old_cfg.get(gstin)] + [
            b for k, b in old_cfg.items() if k.startswith(gstin)]
        donors = [d for d in donors if isinstance(d, dict)]

        if "line_items" not in block:
            for d in donors:
                if d.get("line_items"):
                    block["line_items"] = d["line_items"]
                    break
        for d in donors:
            notes = [n for n in (d.get("prompt_note") or [])
                     if _rich_prompt([n]) and not str(n).startswith("Evidence-grounded")]
            if notes:
                mine = block.get("prompt_note") or []
                block["prompt_note"] = notes + [n for n in mine if n not in notes]
                break
        header = dict(block.get("header") or {})
        for fname, fdef in header.items():
            if not isinstance(fdef, dict) or _rich_prompt(fdef.get("prompt")):
                continue
            for d in donors:
                dp = ((d.get("header") or {}).get(fname) or {}).get("prompt")
                if _rich_prompt(dp):
                    header[fname] = dict(fdef, prompt=dp)
                    break
        block["header"] = header
        merged[key] = block
    return order_blocks(merged)


def _has_text(path) -> bool:
    """Does this PDF carry a real text layer? (fast: plain PyMuPDF text, no OCR)"""
    try:
        import pymupdf
        doc = pymupdf.open(path)
        try:
            n = sum(len(doc[i].get_text("text").strip()) for i in range(doc.page_count))
        finally:
            doc.close()
        return n > _env_int("SCANNED_TEXT_THRESHOLD", 50)
    except Exception:
        return False


def _tax_kind(path):
    """'intra' (CGST + SGST), 'inter' (IGST) or 'unknown', from the QR code: seller and
    buyer in the same state -> intra, different states -> inter. No Gemini call."""
    try:
        qr = _QR.extract_from_blob(open(path, "rb").read(),
                                   invoice_id=os.path.splitext(os.path.basename(path))[0]) or {}
        s, b = str(qr.get("seller_gstin") or "").strip(), str(qr.get("buyer_gstin") or "").strip()
        if len(s) >= 2 and len(b) >= 2 and s[:2].isdigit() and b[:2].isdigit():
            return "intra" if s[:2] == b[:2] else "inter"
    except Exception:
        pass
    return "unknown"


def _tax_kinds(paths):
    """_tax_kind for several PDFs at once (reading a QR code takes 1-3 seconds each)."""
    workers = max(1, _env_int("OCR_MAX_WORKERS", 5))
    if workers > 1 and len(paths) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_tax_kind, paths))
    return [_tax_kind(p) for p in paths]


_GSTIN_NEGATIVE_LABEL = None  # compiled lazily inside gstin_from_text (keeps `re` import local)
_GSTIN_POSITIVE_LABEL = None


def gstin_from_text(raw_text):
    """Fallback GST-number detection for when the QR code is missing/unreadable (only
    reached in scan_invoices() after qr.get("seller_gstin") and GSTIN_HINT.get(name)
    have both failed — see the call site below). Finds every GSTIN-shaped candidate
    in the text and, for each one, looks at whichever label (positive: 'GSTIN'/
    'supplier'/'seller', or negative: 'buyer'/'consignee'/'bill to'/'ship to') is
    NEAREST before it ANYWHERE earlier in the text — not a fixed window — so a label
    far from its GSTIN (e.g. separated by address lines) is still correctly matched.
    A candidate whose nearest preceding label is negative is rejected outright. Among
    the remaining candidates, the one closest to a positive label wins (the seller's
    GSTIN is almost always right next to an explicit 'GSTIN:' label). Returns None if
    nothing passes — callers should then fall back to "UNKNOWN", same as today."""
    import re
    from src.position_regex import GSTIN_TOKEN
    global _GSTIN_NEGATIVE_LABEL, _GSTIN_POSITIVE_LABEL
    if _GSTIN_NEGATIVE_LABEL is None:
        _GSTIN_NEGATIVE_LABEL = re.compile(
            r"(bill\s*to|ship\s*to|detail\s*of\s*receiver|detail\s*of\s*consignee|buyer|consignee)",
            re.I)
        _GSTIN_POSITIVE_LABEL = re.compile(r"(gstin|gstn\s*no|gst\s*no|supplier|seller)", re.I)

    text = raw_text or ""
    candidates = list(re.finditer(GSTIN_TOKEN, text))
    if not candidates:
        return None

    neg_positions = [m.start() for m in _GSTIN_NEGATIVE_LABEL.finditer(text)]
    pos_positions = [m.start() for m in _GSTIN_POSITIVE_LABEL.finditer(text)]

    def nearest_before(label_positions, idx):
        before = [p for p in label_positions if p < idx]
        return max(before) if before else None

    scored = []
    for m in candidates:
        idx = m.start()
        nearest_neg = nearest_before(neg_positions, idx)
        nearest_pos = nearest_before(pos_positions, idx)
        if nearest_neg is not None and (nearest_pos is None or nearest_neg > nearest_pos):
            continue  # this GSTIN sits textually "inside" a buyer/consignee block
        dist = (idx - nearest_pos) if nearest_pos is not None else float("inf")
        scored.append((dist, idx, m.group(0)))

    if not scored:
        return None
    scored.sort()  # smallest label-distance wins; ties broken by earliest position
    return scored[0][2]


_KIND_NAME = {"intra": "CGST/SGST", "inter": "IGST", "unknown": "no QR"}


def collect_invoices(src, gst_value, want, folder):
    """Up to `want` invoices WITH TEXT for one GST number.

    - The candidates are the newest MAX_LOOKBACK_INVOICES invoices (ids and paths only).
    - RANDOM_SAMPLE=1 (default): they are taken in RANDOM order, so the samples are
      spread over time instead of being the newest 20. RANDOM_SAMPLE=0: newest first.
    - Scanned / image-only PDFs are skipped and deleted straight away.
    - MIX_TAX_SAMPLES=1 (default): when all the samples are one tax kind (only IGST, or
      only CGST + SGST), up to MIX_TAX_EXTRA_LOOK more invoices are checked for the
      other kind and up to MIN_OTHER_TAX_SAMPLES of them replace some samples, so the
      JSON learns both.
    Returns (paths, looked_at, scanned, note)."""
    lookback = max(want, _env_int("MAX_LOOKBACK_INVOICES", 200))
    refs = src.fetch_invoice_refs(gst_value, lookback)
    if os.getenv("RANDOM_SAMPLE", "1") != "0":
        seed = os.getenv("RANDOM_SEED", "")
        (random.Random(int(seed)) if seed.lstrip("-").isdigit() else random).shuffle(refs)
    mix = os.getenv("MIX_TAX_SAMPLES", "1") != "0"
    min_other = max(1, min(_env_int("MIN_OTHER_TAX_SAMPLES", 5), want // 2 or 1))
    extra_look = max(0, _env_int("MIX_TAX_EXTRA_LOOK", 20))

    kept, others = [], []                    # [(path, kind)]
    seen = scanned = extra_seen = pos = 0
    while pos < len(refs):
        filling = len(kept) < want
        kinds = {k for _, k in kept if k != "unknown"}
        hunting = (mix and not filling and len(kinds) == 1
                   and len(others) < min_other and extra_seen < extra_look)
        if not (filling or hunting):
            break
        size = want if filling else min(want, extra_look - extra_seen)
        chunk, pos = refs[pos:pos + size], pos + size
        seen += len(chunk)
        if hunting:
            extra_seen += len(chunk)
        readable = []
        for p in src.download_to_dir(chunk, folder):
            if _has_text(p):
                readable.append(p)
            else:
                scanned += 1
                os.remove(p)
        use = readable[:want - len(kept)] if filling else readable
        for p in readable[len(use):]:
            os.remove(p)                                  # more than needed
        found = _tax_kinds(use) if mix else ["unknown"] * len(use)
        for p, kind in zip(use, found):
            if filling:
                kept.append((p, kind))
            elif kind != "unknown" and kind not in kinds and len(others) < min_other:
                others.append((p, kind))
            else:
                os.remove(p)
    # swap the other-kind invoices in for the same number of majority-kind ones
    for item in others:
        for i in range(len(kept) - 1, -1, -1):
            if kept[i][1] != item[1] and kept[i][1] != "unknown":
                os.remove(kept[i][0])
                kept[i] = item
                break
    count = {}
    for _, k in kept:
        count[k] = count.get(k, 0) + 1
    note = ""
    if mix and kept:
        note = ", ".join(f"{n} {_KIND_NAME[k]}" for k, n in sorted(count.items(), key=lambda x: -x[1]))
        real = [k for k in count if k != "unknown"]
        if len(real) == 1 and extra_seen:
            note += f" (no other tax kind in {extra_seen} more invoice(s) checked)"
    return [p for p, _ in kept], seen, scanned, note


def process_batch(src, batch, per_gstin, overwrite):
    """Download -> agent -> store, for one batch of GST numbers. Returns {gstin: status}."""
    outcome = {}
    work = tempfile.mkdtemp(prefix="cfg_batch_")
    inv_dir, out_dir, rev_dir = (os.path.join(work, d) for d in ("invoices", "output", "review"))
    try:
        # 1. invoices of these GST numbers: DB refs -> Azure -> temp folder
        have, names = [], {}
        for v in list(batch):
            # a JSON built earlier that the DB refused: store it now, no rebuild
            kept = os.path.join(UNSAVED, f"{v['gstin']}.json")
            if os.path.exists(kept) and not overwrite:
                try:
                    cfg = json.load(open(kept, encoding="utf-8"))
                    where, _changed = _store_for_all(src, v, cfg, False)
                    os.remove(kept)                 # it is in the database now
                    outcome[v["gstin"]] = "stored"
                    print(f"  [DB] {v['gstin']}: stored the JSON built earlier "
                          f"({len(cfg)} template(s)) -> {where}")
                except Exception as e:
                    outcome[v["gstin"]] = "error"
                    print(f"  [ERROR] {v['gstin']}: still cannot store the JSON built earlier: {e}")
                batch.remove(v)
        for v in batch:
            paths, seen, scanned, mix_note = collect_invoices(src, v["raw"], per_gstin, inv_dir)
            names[v["gstin"]] = {os.path.splitext(os.path.basename(p))[0] for p in paths}
            print(f"  {v['gstin']}: looked at {seen} invoice(s) -> {len(paths)} with text kept"
                  + (f", {scanned} scanned skipped" if scanned else "")
                  + (f" | {mix_note}" if mix_note else ""))
            if paths:
                have.append(v)
            elif scanned:
                outcome[v["gstin"]] = "scanned"
                print(f"  [SCANNED] {v['gstin']}: the {seen} invoice(s) checked are all "
                      f"scanned / image-only PDFs — a regex JSON cannot be built "
                      f"(raise MAX_LOOKBACK_INVOICES to look further back)")
            else:
                outcome[v["gstin"]] = "no_invoices"
        if not have:
            return outcome

        # 2. the agent, pointed at the temp folder (fresh KB: the DB is the store)
        global INPUT_DIR, OUTPUT_DIR, REVIEW_DIR, GSTIN_HINT
        INPUT_DIR, OUTPUT_DIR, REVIEW_DIR = inv_dir, out_dir, rev_dir
        # an invoice without a QR code still belongs to the GST number the table says
        GSTIN_HINT = {n: g for g, ns in names.items() for n in ns}
        # knowledge base: the batch works on a small temp file holding only these GST
        # numbers' entries; they are kept per vendor in knowledge_base/ afterwards.
        # A layout solved in an earlier run is reused (no Gemini call) unless
        # CONFIG_OVERWRITE=1 asks for a rebuild.
        batch_gstins = [v["gstin"] for v in have]
        kbmod.KB_PATH = os.path.join(work, "knowledge_base.json")
        seed = {}
        if not overwrite:
            for v in have:
                from_db = _kb_seed_from_db(src, v)
                # the database is the truth: local knowledge-base files are only used
                # for a GST number that has no agent-built version in the DB yet
                seed.update(from_db or _kb_load_vendors([v["gstin"]]))
                if from_db:
                    print(f"  {v['gstin']}: {len(from_db)} version(s) already in the DB "
                          f"({', '.join(sorted(e['version'] for e in from_db.values()))}) — "
                          f"kept as they are; only a new invoice format is built")
        json.dump(seed, open(kbmod.KB_PATH, "w", encoding="utf-8"))
        try:
            run_folder()
        finally:
            _kb_save_vendors(kbmod.load_kb(), batch_gstins)

        # 3. store solved GST numbers; keep unsolved ones for review
        for v in have:
            g = v["gstin"]
            out_file = os.path.join(out_dir, f"{g}.json")
            rev_file = os.path.join(rev_dir, f"{g}.json")
            try:
                if os.path.exists(out_file):
                    cfg = order_blocks(json.load(open(out_file, encoding="utf-8")))
                    try:
                        where, changed = _store_for_all(src, v, cfg, overwrite)
                    except Exception:
                        os.makedirs(UNSAVED, exist_ok=True)      # do not lose the built JSON
                        json.dump(cfg, open(os.path.join(UNSAVED, f"{g}.json"), "w",
                                            encoding="utf-8"), indent=4, ensure_ascii=False)
                        raise
                    outcome[g] = "stored" if changed else "unchanged"
                    if changed:
                        print(f"  [DB] {g}: rscp_prompt stored ({len(cfg)} template(s): {list(cfg)}) "
                              f"-> {where}")
                    else:
                        print(f"  [DB] {g}: no new invoice format — the JSON in the DB is "
                              f"already up to date ({len(cfg)} template(s)); nothing written")
                elif os.path.exists(rev_file):
                    os.makedirs(REVIEW_KEEP, exist_ok=True)
                    shutil.copy2(rev_file, os.path.join(REVIEW_KEEP, f"{g}.json"))
                    raw_src = os.path.join(rev_dir, "raw")       # the invoice text it failed on
                    if os.path.isdir(raw_src):
                        os.makedirs(os.path.join(REVIEW_KEEP, "raw"), exist_ok=True)
                        for fn in os.listdir(raw_src):
                            if fn.startswith(g):
                                shutil.copy2(os.path.join(raw_src, fn),
                                             os.path.join(REVIEW_KEEP, "raw", fn))
                    outcome[g] = "review"
                    print(f"  [REVIEW] {g}: not fully solved — NOT stored in DB "
                          f"(see review_db/{g}.json)")
                elif names[g] and names[g] <= set(SCANNED_SKIPPED):
                    outcome[g] = "scanned"
                    print(f"  [SCANNED] {g}: all {len(names[g])} invoice(s) are scanned / image-only "
                          f"PDFs with no text — a regex JSON cannot be built for them")
                else:
                    outcome[g] = "no_result"
                    print(f"  [SKIP] {g}: the agent produced nothing for this GST number "
                          f"(no QR on its invoices, or the QR shows a different GST number)")
            except Exception as e:
                outcome[g] = "error"
                print(f"  [ERROR] {g}: {e}")
        return outcome
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()


def run_only(gstins):
    """GST_NUMBERS=... in .envs/.local, or: python main.py <GSTIN> [<GSTIN> ...]  — build + store just these GST numbers
    (ignores the checkpoint). A GST number that already has a JSON keeps its versions;
    only a new invoice format is built and added. CONFIG_OVERWRITE=1 also rebuilds the
    formats it already has, in place (same version names)."""
    per_gstin = max(1, _env_int("MAX_INVOICES_PER_GSTIN", 20))
    overwrite = os.getenv("CONFIG_OVERWRITE", "0") == "1"
    src = _db_source()
    batch = []
    for raw in gstins:
        g = raw.strip().upper()
        ids = src.vendor_ids_for(g)
        if not ids:
            print(f"  {g}: not in master_vendors — there is no vendor row to store a JSON against.")
        else:
            # a GST number named explicitly is always checked: a vendor id with a JSON
            # keeps its versions and only gets a NEW invoice format added
            have_json = [i for i in ids if src.load_config(i)[1]]
            print(f"  {g}: vendor id(s) {ids}"
                  + (f"; {have_json} already have a JSON — existing versions are kept, "
                     f"a new format is added as the next version" if have_json else ""))
            batch.append({"raw": g, "gstin": g, "vendor_ids": ids, "write_ids": list(ids)})
    if not batch:
        print("Nothing to build."); return
    print("=" * 80)
    print("ONLY -> " + ", ".join(v["gstin"] for v in batch))
    print("=" * 80)
    ids_note = {v["gstin"]: f"vendor_ids {' '.join(str(i) for i in v['vendor_ids'])}" for v in batch}
    outcome = process_batch(src, batch, per_gstin, overwrite)
    for g, st in outcome.items():
        append_checkpoint(g, st, ids_note.get(g, ""))
    print("=" * 80)
    print(f"Finished: {outcome}")
    print(f"Checkpoint now: {checkpoint_summary(load_checkpoint())}  "
          f"({os.path.basename(CHECKPOINT)})")
    print("=" * 80)
    usage.print_summary("this run")
    print(f"Every API call is listed in {os.path.basename(usage.LOG_PATH)}")


def run_database():
    size = max(1, _env_int("GST_BATCH_SIZE", 5))
    per_gstin = max(1, _env_int("MAX_INVOICES_PER_GSTIN", 20))
    max_batches = _env_int("MAX_BATCHES", 0)               # 0 = until nothing is left
    overwrite = os.getenv("CONFIG_OVERWRITE", "0") == "1"
    retry_review = os.getenv("RETRY_REVIEW", "0") == "1"

    src = _db_source()
    seen = load_checkpoint()
    # The checkpoint lists EVERY GST number processed, with its last result. Only the
    # ones that could not be solved are skipped because of it; a stored GST number is
    # recognised from the database (and is re-checked when ADD_NEW_FORMATS=1).
    not_solved = {"review", "scanned", "no_invoices", "no_result"}
    skip = {g for g, st in seen.items()
            if (st in not_solved and not retry_review) or (st == "stored" and overwrite)}
    print(f"Config generation from DB + Azure | {size} GST number(s) per batch | "
          f"up to {per_gstin} invoice(s) per GST number | skipping {len(skip)} from checkpoint")
    print(f"Checkpoint so far: {checkpoint_summary(seen)}\n")

    totals, skipped, after, n = {}, {}, "", 0
    while True:
        batch, after, exhausted = next_batch(src, after, size, skip, overwrite, skipped)
        if batch:
            n += 1
            print("=" * 80)
            print(f"BATCH {n} -> " + ", ".join(v["gstin"] for v in batch))
            print("=" * 80)
            batch_mark = usage.mark()
            try:
                outcome = process_batch(src, batch, per_gstin, overwrite)
            except Exception as e:                     # one bad batch never stops the run
                print(f"  [BATCH ERROR] {e}")
                outcome = {v["gstin"]: "error" for v in batch}
            stats = {}
            notes = {v["gstin"]: f"vendor_ids {' '.join(str(i) for i in v['vendor_ids'])}"
                     for v in batch}
            for g, st in outcome.items():
                stats[st] = stats.get(st, 0) + 1
                totals[st] = totals.get(st, 0) + 1
                append_checkpoint(g, st, notes.get(g, ""))   # every GST number is recorded
            print(f"BATCH {n} done: {stats}")
            usage.print_summary(f"batch {n}", since=batch_mark)
            print()
        if exhausted:
            break
        if max_batches and n >= max_batches:
            print(f"MAX_BATCHES={max_batches} reached — stopping here.\n")
            break

    print("=" * 80)
    print(f"Finished — {n} batch(es): {totals or 'nothing to build'}")
    if skipped:
        print(f"GST numbers passed over: {skipped}")
    print(f"Checkpoint now: {checkpoint_summary(load_checkpoint())}  "
          f"({os.path.basename(CHECKPOINT)})")
    print("=" * 80)
    if n > 1:
        usage.print_summary(f"whole run ({n} batches)")
    print(f"Every API call is listed in {os.path.basename(usage.LOG_PATH)}")



def _setting(name, default=""):
    return (os.environ.get(name) or default).strip()


def main(argv=None):
    """ONE entry point. What it does is decided by .envs/.local:

        INVOICE_SOURCE=database   invoices from the DB + Azure, JSON stored in the DB
        INVOICE_SOURCE=folder     invoices from the local invoices/ folder, JSON in output/
                                  (default: database when DB_HOST is set, else folder)
        GST_NUMBERS=A,B           database mode: build only these GST numbers
        REPAIR=1                  folder mode: re-work only the templates in review/

    Command-line values still work and win over the file:
        python main.py 06AAACJ2919A1Z5        only this GST number (database mode)
        python main.py --folder | --database  force a mode
        python main.py --repair               folder mode repair pass
    """
    from src.config import load_env
    load_env()                                              # reads .envs/.local
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in argv if a.startswith("-")}
    gstins = [a for a in argv if not a.startswith("-")]
    gstins = gstins or [g for g in _setting("GST_NUMBERS").replace(";", ",").split(",") if g.strip()]

    source = _setting("INVOICE_SOURCE").lower()
    if "--folder" in flags:
        source = "folder"
    elif "--database" in flags or "--db" in flags:
        source = "database"
    elif source in ("db", "azure"):
        source = "database"
    elif source not in ("database", "folder"):
        source = "database" if (gstins or _setting("DB_HOST")) else "folder"

    if source == "folder":
        print(f"Mode: FOLDER  (invoices from {INPUT_DIR}; set INVOICE_SOURCE=database "
              f"in .envs/.local for the DB + Azure flow)\n")
        run_folder(repair="--repair" in flags or _setting("REPAIR") == "1")
        usage.print_summary("this run")
        print(f"Every API call is listed in {os.path.basename(usage.LOG_PATH)}")
    elif gstins:
        print("Mode: DATABASE + AZURE, only the GST number(s) given\n")
        run_only(gstins)
    else:
        print("Mode: DATABASE + AZURE, all pending GST numbers in batches\n")
        run_database()


if __name__ == "__main__":
    main()