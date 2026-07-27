"""Deterministic market calculations used by the AI and Plotly chart.

This module deliberately does not call an LLM. Every level exposed here is
derived from the OHLCV rows supplied by the application.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return default if not np.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def _first(row: pd.Series, names: Iterable[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def build_price_history(scored_days: list[pd.DataFrame], ticker: str, day_labels: list[str] | None = None) -> pd.DataFrame:
    """Collect one ticker from each scored BEI upload day.

    A real ``trading_date`` is preferred. If the source workbook does not
    expose it, the upload label is used so the chart remains honest about the
    available time axis instead of inventing dates.
    """

    labels = day_labels or []
    target = str(ticker).upper().strip()
    rows: list[dict[str, Any]] = []
    for index, day in enumerate(scored_days or []):
        if day is None or day.empty or "kode saham" not in day.columns:
            continue
        matches = day[day["kode saham"].astype(str).str.upper().str.strip() == target]
        if matches.empty:
            continue
        row = matches.iloc[-1]
        raw_date = _first(row, ("trading_date", "tanggal perdagangan terakhir"), default=None)
        date_value = pd.to_datetime(raw_date, errors="coerce") if raw_date is not None else pd.NaT
        if pd.isna(date_value):
            date_value = labels[index] if index < len(labels) else f"D{index + 1}"
        rows.append({
            "date": date_value,
            "open": _number(_first(row, ("open_num", "open price"))),
            "high": _number(_first(row, ("high_num", "tertinggi"))),
            "low": _number(_first(row, ("low_num", "terendah"))),
            "close": _number(_first(row, ("penutupan_num", "penutupan"))),
            "volume": _number(_first(row, ("volume_num", "volume")), 0.0),
            "value": _number(_first(row, ("nilai_num", "nilai")), 0.0),
            "change_pct": _number(_first(row, ("change_pct",)), np.nan),
            "foreign_net": _number(_first(row, ("foreign_net",)), np.nan),
            "day_index": index,
        })
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "value"])
    history = pd.DataFrame(rows)
    # Preserve upload order when dates are labels or duplicate dates.
    history = history.sort_values("day_index", kind="stable").reset_index(drop=True)
    return history


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(history: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = history["close"].shift(1)
    true_range = pd.concat([
        history["high"] - history["low"],
        (history["high"] - previous_close).abs(),
        (history["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _clean_levels(values: Iterable[float], close: float) -> list[float]:
    levels = []
    for value in values:
        number = _number(value)
        if np.isfinite(number) and number > 0 and (not np.isfinite(close) or number < close * 4):
            levels.append(round(number, 6))
    return sorted(set(levels))


def build_quant_context(history: pd.DataFrame, selected: dict[str, Any] | pd.Series | None = None) -> dict[str, Any]:
    """Return indicators, levels, and execution candidates from known data."""

    data = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if data.empty:
        return {
            "observations": 0,
            "status": "insufficient_data",
            "indicators": {},
            "levels": {"support": [], "resistance": [], "fibonacci": []},
            "execution": {},
            "history": [],
        }

    data["ema20"] = _ema(data["close"], 20)
    data["ema50"] = _ema(data["close"], 50)
    data["ema200"] = _ema(data["close"], 200)
    data["rsi14"] = _rsi(data["close"], 14)
    data["atr14"] = _atr(data, 14)
    data["macd"] = _ema(data["close"], 12) - _ema(data["close"], 26)
    data["macd_signal"] = _ema(data["macd"], 9)

    close = _number(data["close"].iloc[-1])
    atr = _number(data["atr14"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        # A one-day upload can show a deterministic range, but it is marked as
        # a proxy so it is never presented as a full ATR(14).
        atr = _number((data["high"] - data["low"]).iloc[-1], close * 0.02)
    atr = max(atr, close * 0.005 if np.isfinite(close) else 0.0)

    lookback = data.tail(min(len(data), 60))
    swing_high = _number(lookback["high"].max())
    swing_low = _number(lookback["low"].min())
    if not np.isfinite(swing_low):
        swing_low = close
    if not np.isfinite(swing_high):
        swing_high = close
    spread = max(swing_high - swing_low, 0.0)
    ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    fibonacci = [
        {"ratio": ratio, "price": round(swing_high - spread * ratio, 6), "label": f"Fib {ratio:.3f}"}
        for ratio in ratios
    ] if spread > 0 else []

    recent = data.tail(min(len(data), 20))
    support_candidates = [recent["low"].min()]
    resistance_candidates = [recent["high"].max()]
    if len(data) >= 3:
        lows = data["low"].rolling(3, center=True).min()
        highs = data["high"].rolling(3, center=True).max()
        support_candidates.extend(data.loc[data["low"].eq(lows), "low"].tail(3).tolist())
        resistance_candidates.extend(data.loc[data["high"].eq(highs), "high"].tail(3).tolist())
    supports = _clean_levels(support_candidates, close)
    resistances = _clean_levels(resistance_candidates, close)
    supports = [level for level in supports if level <= close * 1.02][-4:]
    resistances = [level for level in resistances if level >= close * 0.98][:4]

    execution = {
        "entry_low": round(close * 0.995, 6),
        "entry_high": round(close + atr * 0.3, 6),
        "stop": round(close - atr * 1.5, 6),
        "tp1": round(close + atr * 2.0, 6),
        "tp2": round(close + atr * 3.5, 6),
        "atr_proxy": len(data) < 14,
    }
    latest = data.iloc[-1]
    indicators = {
        "close": close,
        "ema20": _number(latest.get("ema20")),
        "ema50": _number(latest.get("ema50")),
        "ema200": _number(latest.get("ema200")),
        "rsi14": _number(latest.get("rsi14")),
        "atr14": _number(latest.get("atr14")),
        "macd": _number(latest.get("macd")),
        "macd_signal": _number(latest.get("macd_signal")),
        "swing_high": swing_high,
        "swing_low": swing_low,
    }
    serial_history = data.tail(120).copy()
    serial_history["date"] = serial_history.get("date", pd.Series(range(len(serial_history)))).astype(str)
    serial_history = serial_history.replace([np.inf, -np.inf], np.nan).where(pd.notna(serial_history), None)
    return {
        "observations": int(len(data)),
        "status": "ok" if len(data) >= 2 else "insufficient_data",
        "indicators": indicators,
        "levels": {"support": supports, "resistance": resistances, "fibonacci": fibonacci},
        "execution": execution,
        "history": serial_history[[column for column in ["date", "open", "high", "low", "close", "volume", "ema20", "ema50", "ema200"] if column in serial_history]].to_dict("records"),
    }
