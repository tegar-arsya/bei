"""Guardrails for AI-produced levels and drawing instructions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _in_price_range(value: Any, low: float, high: float) -> bool:
    return _finite(value) and low * 0.5 <= float(value) <= high * 1.5


def validate_analysis(analysis: dict[str, Any], ticker: str, history: pd.DataFrame, quant: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep valid AI output and replace unsafe chart instructions with none."""

    result = dict(analysis or {})
    warnings: list[str] = []
    target = str(ticker).upper().strip()
    if str(result.get("ticker", "")).upper().strip() != target:
        raise ValueError("Ticker pada respons AI tidak sesuai dengan saham yang dipilih.")

    close_values = pd.to_numeric(history.get("close", pd.Series(dtype=float)), errors="coerce") if isinstance(history, pd.DataFrame) else pd.Series(dtype=float)
    low = float(pd.to_numeric(history.get("low", close_values), errors="coerce").min()) if not history.empty else 0.0
    high = float(pd.to_numeric(history.get("high", close_values), errors="coerce").max()) if not history.empty else 0.0
    if not np.isfinite(low) or low <= 0:
        low = float(close_values.min()) if not close_values.empty else 0.0
    if not np.isfinite(high) or high <= 0:
        high = float(close_values.max()) if not close_values.empty else 0.0
    safe_low, safe_high = (min(low, high), max(low, high)) if high > 0 else (0.0, float("inf"))

    clean_drawings = []
    for drawing in result.get("drawings", []) if isinstance(result.get("drawings"), list) else []:
        if not isinstance(drawing, dict) or drawing.get("type") not in {"horizontal_line", "trendline", "rectangle", "label"}:
            warnings.append("Satu drawing AI dihapus karena tipe tidak didukung.")
            continue
        y_start = drawing.get("y_start")
        y_end = drawing.get("y_end")
        if not _in_price_range(y_start, safe_low, safe_high) or not _in_price_range(y_end, safe_low, safe_high):
            warnings.append(f"Drawing '{drawing.get('name', 'AI')}' dihapus karena level di luar rentang OHLC.")
            continue
        item = {
            "type": drawing["type"],
            "name": str(drawing.get("name", "AI"))[:80],
            "purpose": str(drawing.get("purpose", ""))[:240],
            "x_start": str(drawing.get("x_start", "")),
            "x_end": str(drawing.get("x_end", "")),
            "y_start": float(y_start),
            "y_end": float(y_end),
            "evidence": str(drawing.get("evidence", ""))[:240],
        }
        clean_drawings.append(item)
    result["drawings"] = clean_drawings

    candidate_execution = quant.get("execution", {}) if isinstance(quant, dict) else {}
    known_levels = quant.get("levels", {}) if isinstance(quant, dict) else {}
    ai_levels = result.get("levels") if isinstance(result.get("levels"), dict) else {}
    for key in ("support", "resistance"):
        known = [float(value) for value in known_levels.get(key, []) if _in_price_range(value, safe_low, safe_high)]
        if known:
            original = ai_levels.get(key, [])
            if original != known:
                warnings.append(f"{key} memakai level quant engine, bukan angka bebas dari model.")
            ai_levels[key] = known
    for key in ("entry_low", "entry_high", "stop", "tp1", "tp2"):
        value = ai_levels.get(key)
        expected = candidate_execution.get(key)
        tolerance = max(abs(float(expected)) * 0.03, 0.000001) if _finite(expected) else 0
        differs_from_engine = _finite(expected) and (not _finite(value) or abs(float(value) - float(expected)) > tolerance)
        if (not _in_price_range(value, safe_low, safe_high) or differs_from_engine) and key in candidate_execution:
            ai_levels[key] = candidate_execution[key]
            warnings.append(f"Level {key} AI tidak valid; dikembalikan ke kandidat quant engine.")
    result["levels"] = ai_levels
    if warnings:
        result["validation_warnings"] = warnings
    return result, warnings
