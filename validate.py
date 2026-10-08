"""
validate.py  —  measure the REAL accuracy of the stored configs.

For every invoice in a folder, find its template's config (by seller GSTIN, doc type
and layout hash), extract every field with the production-faithful extractor, and
compare with the TRUE value:
  - signed QR:   seller GSTIN, invoice number, invoice date, IRN, invoice total
  - discovery:   every other field (cached, so re-running costs nothing), with the
                 same rules the build uses (value must be printed on the invoice,
                 tax filed under the right tax by the QR state codes, nil taxes).

Per field it reports OK / WRONG / MISSING (a value exists but nothing extracted) /
UNVERIFIED (extracted, but no true value to compare with). Accuracy =
OK / (OK + WRONG + MISSING).

    python validate.py                     # invoices/ against the solved configs
    python validate.py --dir holdout       # NEW invoices the build never saw (best test)
    python validate.py --include-review    # also score the partial configs in review/
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _compare(name, datatype, got, values, nil):
    from src.field_tester import same_value
    empty = got in (None, "", [])
    if name in values:
        if empty:
            return "MISSING", values[name]
        return ("OK" if same_value(name, datatype, got, values[name]) else "WRONG"), values[name]
    if name in nil:                                   # tax is nil on this invoice
        return ("OK" if (empty or got in (0, 0.0)) else "WRONG"), 0.0
    return ("NOT_ON_INVOICE" if (empty or got in (0, 0.0)) else "UNVERIFIED"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "invoices"))
    ap.add_argument("--include-review", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "validation_report.json"))
    args = ap.parse_args()

    import main as M                       # scanner, review loader (same text + QR as the build)
    from src.kb import load_kb
    from src.discover import discover
    from src.field_tester import extract_field
    from graph import _trusted_values, _canon_dt

    pdfs = M._list_pdfs(args.dir)
    if not pdfs:
        print(f"No PDFs in {args.dir}"); return
    kb = load_kb()
    configs = {(e["gstin"], e["doc_type"], e["layout_hash"]): (e["config_key"], e["config"], "solved")
               for e in kb.values()}
    if args.include_review:
        todo, _ = M.load_review_templates()
        for (g, dt, h), t in todo.items():
            configs.setdefault((g, dt, h), (t["cfg_key"], {"header": t["config"]}, "review"))

    print(f"Validating {len(pdfs)} invoice(s) from {args.dir} against "
          f"{len(configs)} config(s)\n")
    groups = M.scan_invoices(pdfs)

    report, totals = [], {}
    by_vendor, by_field = {}, {}
    for key, members in groups.items():
        gstin = key[0]
        if key not in configs:
            for m in members:
                report.append({"invoice": m["name"], "gstin": gstin, "template": list(key),
                               "status": "NO_CONFIG (template not built / still in review)"})
            print(f"  {gstin} layout {key[2]}: {len(members)} invoice(s) — NO CONFIG for this template")
            continue
        cfg_key, inner, kind = configs[key]
        header = (inner or {}).get("header", {})
        print(f"[{cfg_key}] ({kind}) {len(members)} invoice(s)")
        for m in members:
            present, _ = discover(m["raw"])
            values, nil, _ = _trusted_values(m["raw"], m["qr"], present)
            if m["qr"].get("tot_inv_val"):             # the signed total is the truth
                values["total_invoice_value"] = m["qr"]["tot_inv_val"]
            row, bad = {}, []
            for f, block in header.items():
                spec = {**block, "name": f, "datatype": block.get("datatype") or _canon_dt(f)}
                if spec.get("regex") or spec.get("alias"):
                    try:
                        got = extract_field(spec, m["raw"])
                    except Exception as e:
                        got = None
                else:
                    got = block.get("fallback_value")
                status, exp = _compare(f, spec["datatype"], got, values, nil)
                row[f] = {"status": status, "got": got, "expected": exp}
                if status in ("OK", "WRONG", "MISSING"):
                    for bucket in (totals, by_vendor.setdefault(gstin, {}), by_field.setdefault(f, {})):
                        bucket[status] = bucket.get(status, 0) + 1
                if status in ("WRONG", "MISSING"):
                    bad.append(f"{f}: got {got!r}, expected {exp!r}")
            ok = sum(1 for r in row.values() if r["status"] == "OK")
            scored = sum(1 for r in row.values() if r["status"] in ("OK", "WRONG", "MISSING"))
            print(f"   {m['name']:<22} {ok}/{scored} correct" + ("" if not bad else "   ✗ " + " | ".join(bad)))
            report.append({"invoice": m["name"], "gstin": gstin, "config": cfg_key,
                           "config_kind": kind, "fields": row})

    def pct(d):
        n = d.get("OK", 0) + d.get("WRONG", 0) + d.get("MISSING", 0)
        return (100.0 * d.get("OK", 0) / n) if n else 0.0, n

    print("\n=== ACCURACY BY FIELD ===")
    for f in sorted(by_field):
        p, n = pct(by_field[f])
        print(f"  {f:32} {p:6.1f}%   ({by_field[f].get('OK',0)}/{n}; wrong {by_field[f].get('WRONG',0)}, "
              f"missing {by_field[f].get('MISSING',0)})")
    print("\n=== ACCURACY BY VENDOR ===")
    for g in sorted(by_vendor):
        p, n = pct(by_vendor[g])
        print(f"  {g:18} {p:6.1f}%   ({by_vendor[g].get('OK',0)}/{n})")
    p, n = pct(totals)
    no_cfg = sum(1 for r in report if str(r.get("status", "")).startswith("NO_CONFIG"))
    print(f"\nOVERALL field accuracy: {p:.1f}%  ({totals.get('OK',0)}/{n} fields on "
          f"{len(pdfs) - no_cfg} invoice(s) with a config; {no_cfg} invoice(s) had no config)")
    json.dump({"overall_pct": round(p, 2), "by_field": by_field, "by_vendor": by_vendor,
               "invoices": report}, open(args.out, "w", encoding="utf-8"), indent=2, default=str)
    print(f"Full report -> {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
