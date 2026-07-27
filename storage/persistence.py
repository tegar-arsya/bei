"""Save structured analyses locally and optionally mirror them to Supabase.

Supabase is intentionally optional. Missing credentials or a missing table do
not block the Streamlit analysis, and this module never deletes or migrates
existing tables.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def _configured_value(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


IMPORT_MIGRATION_FILE = "supabase/migrations/20260723_bei_streamlit_imports.sql"
IMPORT_PERMISSIONS_FILE = "supabase/migrations/20260723_bei_streamlit_imports_permissions.sql"


def supabase_config() -> dict[str, str]:
    return {
        "url": _configured_value("SUPABASE_URL").rstrip("/"),
        "key": (
            _configured_value("SUPABASE_SERVICE_ROLE_KEY")
            or _configured_value("SUPABASE_SECRET_KEY")
            or _configured_value("SUPABASE_KEY")
        ).strip(),
        "bucket": _configured_value("SUPABASE_IMPORT_BUCKET", "saham"),
    }


def supabase_is_configured() -> bool:
    config = supabase_config()
    return bool(config["url"] and config["key"])


def _supabase_headers(config: dict[str, str], content_type: str = "application/json") -> dict[str, str]:
    return {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": content_type,
    }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert Excel values to JSON-safe records, including dates and nulls."""
    if frame is None or frame.empty:
        return []
    # pandas' JSON encoder handles NaN, NaT, numpy integers, and timestamps.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _row_ticker(row: dict[str, Any]) -> str:
    for key in ("kode saham", "kode perusahaan", "id instrument", "kode"):
        value = row.get(key)
        if value not in (None, "", "nan"):
            return str(value).strip().upper()
    return ""


def _rest_error(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or body)
    except ValueError:
        pass
    return response.text[:500] or f"HTTP {response.status_code}"


def _get_json_pages(config: dict[str, str], endpoint: str, params: dict[str, str], page_size: int = 1000) -> tuple[list[dict[str, Any]], str | None]:
    """Read a PostgREST collection in pages so stock rows are not truncated."""
    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        while True:
            page_params = {**params, "offset": str(offset), "limit": str(page_size)}
            response = requests.get(endpoint, params=page_params, headers=_supabase_headers(config), timeout=60)
            if not response.ok:
                return rows, _rest_error(response)
            page = response.json()
            if not isinstance(page, list):
                return rows, "Respons Supabase bukan array data."
            rows.extend(item for item in page if isinstance(item, dict))
            if len(page) < page_size:
                break
            offset += page_size
        return rows, None
    except requests.RequestException as exc:
        return rows, str(exc)


def load_import_dates(limit: int = 30) -> tuple[list[str], str | None]:
    """Return trading dates already persisted in Supabase, newest first."""
    config = supabase_config()
    if not config["url"] or not config["key"]:
        return [], "Supabase belum dikonfigurasi."
    endpoint = f"{config['url']}/rest/v1/bei_import_batches"
    rows, error = _get_json_pages(
        config,
        endpoint,
        {"select": "trading_date,status", "order": "trading_date.desc", "limit": str(limit)},
        page_size=max(1, min(limit, 100)),
    )
    dates = [str(row["trading_date"]) for row in rows if row.get("trading_date")]
    return dates[:limit], error


def load_import_bundle(trading_date: str) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Load one persisted date batch back into the DataFrame shapes used by app.py."""
    config = supabase_config()
    if not config["url"] or not config["key"]:
        return None, "Supabase belum dikonfigurasi."
    batches_endpoint = f"{config['url']}/rest/v1/bei_import_batches"
    try:
        response = requests.get(
            batches_endpoint,
            params={"select": "id,trading_date,status", "trading_date": f"eq.{trading_date}", "limit": "1"},
            headers=_supabase_headers(config),
            timeout=30,
        )
        if not response.ok:
            return None, _rest_error(response)
        batches = response.json()
        if not batches:
            return None, f"Data tanggal {trading_date} tidak ditemukan di Supabase."
        batch_id = batches[0].get("id")
        rows_endpoint = f"{config['url']}/rest/v1/bei_import_rows"
        rows, error = _get_json_pages(
            config,
            rows_endpoint,
            {
                "select": "dataset_type,row_number,row_data",
                "batch_id": f"eq.{batch_id}",
                "order": "dataset_type.asc,row_number.asc",
            },
        )
        if error:
            return None, error
        frames: dict[str, pd.DataFrame] = {}
        for dataset_type in {str(row.get("dataset_type")) for row in rows if row.get("dataset_type")}:
            dataset_rows = [row.get("row_data", {}) for row in rows if row.get("dataset_type") == dataset_type]
            frames[dataset_type] = pd.DataFrame(dataset_rows)
        return frames, None
    except requests.RequestException as exc:
        return None, str(exc)


def _get_or_create_batch(config: dict[str, str], trading_date: str, manifest: dict[str, Any], row_counts: dict[str, int]) -> tuple[str | None, str | None]:
    endpoint = f"{config['url']}/rest/v1/bei_import_batches"
    headers = _supabase_headers(config)
    try:
        lookup = requests.get(
            endpoint,
            params={"select": "id", "trading_date": f"eq.{trading_date}", "limit": "1"},
            headers=headers,
            timeout=20,
        )
        if not lookup.ok:
            return None, _rest_error(lookup)
        existing = lookup.json()
        batch_id = existing[0]["id"] if existing else str(uuid.uuid4())
        record = {
            "id": batch_id,
            "trading_date": trading_date,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file_manifest": manifest,
            "row_counts": row_counts,
            "status": "ready",
        }
        if existing:
            response = requests.patch(
                f"{endpoint}?id=eq.{batch_id}",
                headers={**headers, "Prefer": "return=minimal"},
                json=record,
                timeout=20,
            )
        else:
            response = requests.post(
                endpoint,
                headers={**headers, "Prefer": "return=minimal"},
                json=record,
                timeout=20,
            )
        if not response.ok:
            return None, _rest_error(response)
        return batch_id, None
    except requests.RequestException as exc:
        return None, str(exc)


def save_bei_import(
    trading_date: str,
    frames: dict[str, pd.DataFrame],
    uploaded_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Persist one date batch and raw files without requiring Supabase SDK.

    The app uses a server-only Supabase key. Re-uploading a date removes only
    child rows belonging to that same date batch, then inserts the new snapshot.
    No other dates or tables are touched.
    """
    config = supabase_config()
    if not config["url"] or not config["key"]:
        return {"ok": False, "configured": False, "message": "Supabase belum dikonfigurasi."}

    manifest = {
        name: {"filename": item.get("name", ""), "size": len(item.get("bytes", b"")), "storage_path": ""}
        for name, item in uploaded_files.items()
    }
    row_counts = {name: len(_frame_records(frame)) for name, frame in frames.items()}
    batch_id, error = _get_or_create_batch(config, trading_date, manifest, row_counts)
    if error or not batch_id:
        if error and ("permission denied" in error.lower() or "42501" in error):
            return {
                "ok": False,
                "configured": True,
                "permission_required": True,
                "message": f"Tabel Supabase sudah ada, tetapi key tidak punya permission. Jalankan {IMPORT_PERMISSIONS_FILE} di SQL Editor.",
                "detail": error,
            }
        return {
            "ok": False,
            "configured": True,
            "migration_required": True,
            "message": f"Database belum siap. Jalankan {IMPORT_MIGRATION_FILE} di Supabase SQL Editor.",
            "detail": error or "batch id tidak tersedia",
        }

    rows_endpoint = f"{config['url']}/rest/v1/bei_import_rows"
    headers = _supabase_headers(config)
    try:
        deleted = requests.delete(
            rows_endpoint,
            params={"batch_id": f"eq.{batch_id}"},
            headers={**headers, "Prefer": "return=minimal"},
            timeout=30,
        )
        if not deleted.ok:
            return {"ok": False, "configured": True, "message": "Baris tanggal lama gagal dibersihkan.", "detail": _rest_error(deleted)}

        for dataset_type, frame in frames.items():
            records = _frame_records(frame)
            for start in range(0, len(records), 500):
                chunk = [
                    {
                        "batch_id": batch_id,
                        "trading_date": trading_date,
                        "dataset_type": dataset_type,
                        "row_number": start + index + 1,
                        "ticker": _row_ticker(row),
                        "row_data": row,
                    }
                    for index, row in enumerate(records[start:start + 500])
                ]
                inserted = requests.post(
                    rows_endpoint,
                    headers={**headers, "Prefer": "return=minimal"},
                    json=chunk,
                    timeout=60,
                )
                if not inserted.ok:
                    return {"ok": False, "configured": True, "message": f"Gagal menyimpan dataset {dataset_type}.", "detail": _rest_error(inserted)}

        raw_status = "not_uploaded"
        for dataset_type, item in uploaded_files.items():
            filename = str(item.get("name", f"{dataset_type}.xlsx"))
            safe_name = filename.replace("/", "_").replace("\\", "_")
            storage_path = f"bei/{trading_date}/{safe_name}"
            uploaded = requests.post(
                f"{config['url']}/storage/v1/object/{config['bucket']}/{storage_path}",
                headers={**_supabase_headers(config, item.get("mime", "application/octet-stream")), "x-upsert": "true"},
                data=item.get("bytes", b""),
                timeout=120,
            )
            if not uploaded.ok:
                raw_status = f"bucket_error_{uploaded.status_code}"
                continue
            manifest[dataset_type]["storage_path"] = storage_path
            raw_status = "uploaded"

        requests.patch(
            f"{config['url']}/rest/v1/bei_import_batches?id=eq.{batch_id}",
            headers={**headers, "Prefer": "return=minimal"},
            json={"file_manifest": manifest, "status": "complete"},
            timeout=20,
        )
        return {
            "ok": True,
            "configured": True,
            "batch_id": batch_id,
            "row_counts": row_counts,
            "raw_status": raw_status,
            "message": f"Data tanggal {trading_date} berhasil disimpan.",
        }
    except requests.RequestException as exc:
        return {"ok": False, "configured": True, "message": "Gagal terhubung ke Supabase.", "detail": str(exc)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def save_analysis(ticker: str, context: dict[str, Any], analysis: dict[str, Any], local_path: str = "data/ai_analysis_history.json") -> dict[str, str]:
    record = _jsonable({
        "ticker": ticker,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": context,
        "analysis": analysis,
    })
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(rows, list):
            rows = []
    except Exception:
        rows = []
    rows.append(record)
    path.write_text(json.dumps(rows[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    status = {"local": "saved", "supabase": "not_configured"}

    url = _configured_value("SUPABASE_URL").rstrip("/")
    key = (
        _configured_value("SUPABASE_SERVICE_ROLE_KEY")
        or _configured_value("SUPABASE_SECRET_KEY")
        or _configured_value("SUPABASE_KEY")
    ).strip()
    table = _configured_value("SUPABASE_AI_ANALYSIS_TABLE", "ai_analyses")
    if not url or not key:
        return status
    try:
        response = requests.post(
            f"{url}/rest/v1/{table}",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Prefer": "return=minimal"},
            json=record,
            timeout=15,
        )
        status["supabase"] = "saved" if response.ok else f"error_{response.status_code}"
    except requests.RequestException:
        status["supabase"] = "connection_error"
    return status
