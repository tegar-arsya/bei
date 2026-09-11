"""Save structured analyses locally."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def save_analysis(
    ticker: str,
    context: dict[str, Any],
    analysis: dict[str, Any],
    local_path: str = "data/ai_analysis_history.json",
) -> dict[str, str]:
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
    return {"local": "saved"}