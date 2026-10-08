"""
vendor_config_source.py  —  DB + Azure access for the config-generation agent.

Same tables / columns / helpers as invoice_document_intelligence:
  * invoices      : invoicecheck_invoiceextracted  (gst_number + file path)   [read only]
                    PDFs come from Azure via utilities.azure_utils.download_invoice_from_blob
  * vendors       : master_vendors            (id, gst_number)
  * vendor config : master_vendorocrprompt    (vendor_id, rscp_prompt)   <- the JSON that
                    src/fieldextractor/vendor_config_loader.py reads in production

Nothing here loads the invoice table into memory:
  * GST numbers are paged with keyset pagination (WHERE gst_number > last ... LIMIT n)
  * per GST number only (id, file path) of at most MAX_INVOICES_PER_GSTIN rows is read
"""
import json
import os

from mysql.connector import Error

from connector.sql_connector import connect_to_mysql

from concurrent.futures import ThreadPoolExecutor
from utilities.azure_utils import download_invoice_from_blob

_GST_CANDIDATES = ("gst_number", "supplier_gst_number", "supplier_gst", "gstin_number", "gstin")
_PATH_CANDIDATES = ("file_path", "invoice_file_path", "blob_path", "file_url", "invoice_file", "file")

VENDOR_TABLE = "master_vendors"
PROMPT_TABLE = "master_vendorocrprompt"



class VendorConfigSource:

    def __init__(self, db_conn=None) -> None:
        self.db_conn = db_conn or connect_to_mysql()
        if self.db_conn is None:
            raise RuntimeError("Could not connect to MySQL — check DB_* in .envs/.local")
        self._cols = None
        self._pcols = None
        self._told = False

    # ── connection ─────────────────────────────────────────────────────────

    def _conn(self):
        """The agent can work for minutes between DB calls: revive the connection
        first (same ping/reconnect pattern as invoice_document_intelligence/main.py)."""
        try:
            self.db_conn.ping(reconnect=True, attempts=3, delay=1)
        except Exception:
            print("  [DB] connection lost — reconnecting...")
            self.db_conn = connect_to_mysql()
            if self.db_conn is None:
                raise RuntimeError("Could not reconnect to MySQL")
        return self.db_conn

    def _invoice_table(self) -> str:
        return os.getenv("INVOICE_TABLE", "invoicecheck_invoiceextracted")

    # ── the invoice table: invoicecheck_invoiceextracted ──────────────────

    def _resolve_columns(self):
        """Find the GST-number and file-path columns of the invoice table once
        (SHOW COLUMNS), so a different column name only needs an env override."""
        if self._cols:
            return self._cols
        cursor = None
        try:
            cursor = self._conn().cursor()
            cursor.execute(f"SHOW COLUMNS FROM {self._invoice_table()}")
            names = [r[0] for r in cursor.fetchall()]
        except Error as e:
            raise Error(f"[CFG] Cannot read columns of {self._invoice_table()}: {e}")
        finally:
            if cursor:
                cursor.close()
        low = {n.lower(): n for n in names}

        def pick(env, candidates, what):
            want = os.getenv(env)
            if want:
                if want.lower() not in low:
                    raise RuntimeError(f"{env}={want} is not a column of {self._invoice_table()}. "
                                       f"Columns: {names}")
                return low[want.lower()]
            for c in candidates:
                if c in low:
                    return low[c]
            raise RuntimeError(
                f"Could not find the {what} column in {self._invoice_table()}. "
                f"Columns: {names}. Set {env}=<column name> in .envs/.local")

        self._cols = {
            "gst": pick("INVOICE_GST_COLUMN", _GST_CANDIDATES, "GST number"),
            "path": pick("INVOICE_PATH_COLUMN", _PATH_CANDIDATES, "file path"),
            "id": pick("INVOICE_ID_COLUMN", ("id",), "id"),
        }
        print(f"  [DB] {self._invoice_table()}: GST column = {self._cols['gst']}, "
              f"file path column = {self._cols['path']}")
        return self._cols

    def fetch_gst_numbers(self, after: str = "", limit: int = 5) -> list[str]:
        """Next `limit` distinct GST numbers AFTER `after` that have at least one
        invoice file. Keyset pagination: only `limit` small values come back, the
        invoice table is never loaded. (An index on the GST column makes it instant.)"""
        c = self._resolve_columns()
        query = (
            f"SELECT DISTINCT {c['gst']} FROM {self._invoice_table()} "
            f"WHERE {c['gst']} > %s AND {c['path']} IS NOT NULL AND {c['path']} <> '' "
            f"ORDER BY {c['gst']} ASC LIMIT %s"
        )
        cursor = None
        try:
            cursor = self._conn().cursor()
            cursor.execute(query, (after, limit))
            return [r[0] for r in cursor.fetchall()]
        except Error as e:
            raise Error(f"[CFG] Error fetching GST numbers: {e}")
        finally:
            if cursor:
                cursor.close()

    def fetch_invoice_refs(self, gst_value: str, limit: int = 20, before_id=None) -> list[dict]:
        """(id, file_path) of the newest `limit` invoices of this GST number — or, with
        before_id, the next `limit` OLDER ones (id < before_id). Only two small columns
        are read — never the whole table."""
        c = self._resolve_columns()
        older = f"AND {c['id']} < %s " if before_id is not None else ""
        query = (
            f"SELECT {c['id']} AS id, {c['path']} AS file_path FROM {self._invoice_table()} "
            f"WHERE {c['gst']} = %s AND {c['path']} IS NOT NULL AND {c['path']} <> '' "
            f"{older}ORDER BY {c['id']} DESC LIMIT %s"
        )
        params = (gst_value, before_id, limit) if before_id is not None else (gst_value, limit)
        cursor = None
        try:
            cursor = self._conn().cursor(dictionary=True)
            cursor.execute(query, params)
            return cursor.fetchall()
        except Error as e:
            raise Error(f"[CFG] Error fetching invoices for {gst_value}: {e}")
        finally:
            if cursor:
                cursor.close()

    def download_to_dir(self, refs: list[dict], folder: str) -> list[str]:
        """Download the PDFs from Azure (parallel, same helper as RSCP) to disk as
        <invoice id>.pdf. Returns the paths written."""
        os.makedirs(folder, exist_ok=True)

        def _worker(ref):
            fp = str(ref["file_path"])
            if not fp.lower().endswith(".pdf"):
                print(f"  Invoice ID {ref['id']}: not a PDF ({fp}) — skipping.")
                return None
            try:
                data = download_invoice_from_blob(fp)
            except Exception as e:
                print(f"  Invoice ID {ref['id']}: Failed to download blob — {e}")
                return None
            path = os.path.join(folder, f"{ref['id']}.pdf")
            with open(path, "wb") as f:
                f.write(data)
            return path

        workers = int(os.getenv("OCR_MAX_WORKERS", 5))
        if workers > 1 and len(refs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                out = list(ex.map(_worker, refs))
        else:
            out = [_worker(r) for r in refs]
        return [p for p in out if p]

    # ── GSTIN -> vendor (same lookup as vendor_config_loader.py) ───────────

    def vendor_ids_for(self, gstin: str) -> list:
        """EVERY master_vendors.id that carries this GST number (one GST number can be
        registered under several vendor ids), lowest id first. [] when there is none."""
        cursor = None
        try:
            cursor = self._conn().cursor()
            cursor.execute(
                f"SELECT id FROM {VENDOR_TABLE} WHERE UPPER(TRIM(gst_number)) = %s ORDER BY id",
                (gstin.strip().upper(),))
            return [r[0] for r in cursor.fetchall()]
        finally:
            if cursor:
                cursor.close()

    def vendor_id_for(self, gstin: str):
        """The first vendor id of this GST number, or None (kept for older callers)."""
        ids = self.vendor_ids_for(gstin)
        return ids[0] if ids else None

    # ── read / store the vendor config JSON ────────────────────────────────

    def load_config(self, vendor_id: int):
        """(row_exists, parsed rscp_prompt or None)."""
        cursor = None
        try:
            cursor = self._conn().cursor(dictionary=True)
            cursor.execute(
                f"SELECT rscp_prompt FROM {PROMPT_TABLE} WHERE vendor_id = %s LIMIT 1",
                (vendor_id,))
            row = cursor.fetchone()
        finally:
            if cursor:
                cursor.close()
        if not row:
            return False, None
        raw = row.get("rscp_prompt")
        if not raw:
            return True, None
        if not isinstance(raw, str):
            return True, raw
        try:
            return True, json.loads(raw, strict=False)
        except json.JSONDecodeError:
            return True, None

    def _prompt_columns(self):
        """Columns of master_vendorocrprompt (SHOW COLUMNS), read once."""
        if self._pcols is None:
            cursor = None
            try:
                cursor = self._conn().cursor()
                cursor.execute(f"SHOW COLUMNS FROM {PROMPT_TABLE}")
                # (Field, Type, Null, Key, Default, Extra)
                self._pcols = [(r[0], str(r[1]).lower() if not isinstance(r[1], bytes)
                                else r[1].decode().lower(), r[2], r[4], str(r[5] or "").lower())
                               for r in cursor.fetchall()]
            finally:
                if cursor:
                    cursor.close()
        return self._pcols

    def _insert_extras(self):
        """Values for the REQUIRED columns of a new master_vendorocrprompt row that
        the agent does not produce (NOT NULL, no default, not auto-increment), e.g.
        created_on. Returns [(column, sql_expression_or_%s, param_or_None)]."""
        extras = []
        for name, ctype, null, default, extra in self._prompt_columns():
            low = name.lower()
            if low in ("vendor_id", "rscp_prompt") or "auto_increment" in extra:
                continue
            # master_vendorocrprompt's own columns: same values as the existing rows
            known = {
                "upload_type": os.getenv("PROMPT_UPLOAD_TYPE", "invoice"),
                "is_active": 1,
                "application_id": int(os.getenv("PROMPT_APPLICATION_ID", "1")),
                "prompt": "",
            }
            user = os.getenv("PROMPT_USER_ID")              # optional: created_by / updated_by
            if user:
                known["created_by_id"] = known["updated_by_id"] = int(user)
            if low in known:
                extras.append((name, "%s", known[low]))
                continue
            is_time = any(t in ctype for t in ("datetime", "timestamp", "date"))
            stamp = is_time and any(k in low for k in ("creat", "modif", "updat"))
            if not stamp and (null == "YES" or default is not None):
                continue                                   # the DB fills it itself
            if is_time:
                extras.append((name, "NOW()", None))
            elif low.endswith("_by"):
                extras.append((name, "%s", "config_agent"))
            elif low in ("is_active", "active", "status", "is_enabled", "enabled"):
                extras.append((name, "%s", 1))
            elif any(t in ctype for t in ("int", "decimal", "float", "double", "bit")):
                extras.append((name, "%s", 0))
            elif "json" in ctype:
                extras.append((name, "%s", "{}"))
            else:
                extras.append((name, "%s", ""))
        return extras

    def _update_stamps(self):
        """'modified/updated' timestamp columns to refresh on UPDATE."""
        return [name for name, ctype, _n, _d, _e in self._prompt_columns()
                if any(t in ctype for t in ("datetime", "timestamp"))
                and any(k in name.lower() for k in ("modif", "updat"))]

    def save_config(self, vendor_id: int, config: dict) -> str:
        """Store the GSTIN-keyed config in master_vendorocrprompt.rscp_prompt:
        UPDATE the vendor's row, INSERT one if the vendor has none (filling the
        table's other required columns, e.g. created_on). Committed per vendor, so a
        crash never loses finished vendors. Returns 'updated'/'inserted'."""
        payload = json.dumps(config, ensure_ascii=False)
        exists, _ = self.load_config(vendor_id)
        cursor = None
        try:
            conn = self._conn()
            cursor = conn.cursor()
            if exists:
                stamps = "".join(f", {c} = NOW()" for c in self._update_stamps())
                params = [payload]
                user = os.getenv("PROMPT_USER_ID")
                if user and any(c[0].lower() == "updated_by_id" for c in self._prompt_columns()):
                    stamps += ", updated_by_id = %s"
                    params.append(int(user))
                cursor.execute(
                    f"UPDATE {PROMPT_TABLE} SET rscp_prompt = %s{stamps} WHERE vendor_id = %s",
                    (*params, vendor_id))
                action = "updated"
            else:
                extras = self._insert_extras()
                cols = ["vendor_id", "rscp_prompt"] + [c for c, _e, _p in extras]
                exprs = ["%s", "%s"] + [e for _c, e, _p in extras]
                params = [vendor_id, payload] + [p for _c, e, p in extras if e == "%s"]
                cursor.execute(
                    f"INSERT INTO {PROMPT_TABLE} ({', '.join(cols)}) VALUES ({', '.join(exprs)})",
                    tuple(params))
                action = "inserted"
                if extras and not self._told:
                    self._told = True
                    print(f"  [DB] new {PROMPT_TABLE} rows also set: "
                          + ", ".join(f"{c}={'NOW()' if e != '%s' else repr(p)}" for c, e, p in extras))
            conn.commit()
            return action
        except Error as e:
            self.db_conn.rollback()
            raise Error(f"[CFG] Error saving config for vendor_id={vendor_id}: {e}")
        finally:
            if cursor:
                cursor.close()
