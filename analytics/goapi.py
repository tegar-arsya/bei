"""Small, server-side client for the GOAPI IDX market-data API.

The client deliberately keeps the API key out of URLs and out of returned
objects. GOAPI is used for quotes and historical OHLC; TradingView remains the
interactive chart provider.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


DEFAULT_BASE_URL = "https://api.goapi.io"


class GoAPIError(RuntimeError):
    """A safe, user-facing GOAPI error without credentials in the message."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def goapi_config() -> dict[str, str]:
    return {
        "url": (_secret("GOAPI_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
        "key": _secret("GOAPI_API_KEY"),
    }


def goapi_is_configured() -> bool:
    config = goapi_config()
    return bool(config["url"] and config["key"])


def _headers(config: dict[str, str]) -> dict[str, str]:
    # The OpenAPI security scheme uses X-API-KEY. The public introduction page
    # mentions Authorization, but the live API accepts X-API-KEY consistently.
    return {
        "X-API-KEY": config["key"],
        "Accept": "application/json",
        "User-Agent": "BEI-Stock-Screener/1.0",
    }


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    """Handle the documented data array and data.results/result variants."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return _extract_rows(data)
    return _extract_rows(payload.get("results", []))


def _request(path: str, params: dict[str, Any] | None = None) -> Any:
    config = goapi_config()
    if not config["key"]:
        raise GoAPIError("GOAPI_API_KEY belum diisi di Streamlit Secrets.")
    try:
        response = requests.get(
            f"{config['url']}{path}",
            params=params or {},
            headers=_headers(config),
            timeout=25,
        )
    except requests.RequestException as exc:
        raise GoAPIError(f"Gagal terhubung ke GoAPI: {exc}") from exc
    if not response.ok:
        if response.status_code in {401, 403}:
            raise GoAPIError("GOAPI menolak akses. Periksa API key dan hak akses trial.", response.status_code)
        if response.status_code == 429:
            raise GoAPIError("Batas request GOAPI tercapai. Tunggu lalu coba refresh.", response.status_code)
        raise GoAPIError(f"GOAPI HTTP {response.status_code}.", response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise GoAPIError("Respons GOAPI bukan JSON yang valid.") from exc


def _number(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(str(value).replace(",", ""))
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _first(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _ticker(row: dict[str, Any]) -> str:
    raw = _first(row, "ticker", "symbol", "code", "kode")
    if isinstance(raw, dict):
        raw = _first(raw, "ticker", "symbol", "code")
    return str(raw or "").upper().replace("IDX:", "").strip()


def _change_pct(row: dict[str, Any]) -> float:
    lowered = {str(key).lower(): value for key, value in row.items()}
    # GOAPI's documented `change_pct` is already percentage points (for
    # example -0.3984 means -0.3984%). Some ranking endpoints use `percent`
    # as a fractional return (for example -0.0175 means -1.75%).
    direct_value = lowered.get("change_pct", lowered.get("changepercent"))
    if direct_value not in (None, ""):
        return _number(direct_value)
    value = lowered.get("percent", lowered.get("percentage"))
    result = _number(value)
    # Normalize the fractional form used by the ranking endpoints only.
    if np.isfinite(result) and abs(result) <= 1:
        result *= 100
    return result


def normalize_price_rows(rows: list[dict[str, Any]], default_ticker: str = "") -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        ticker = _ticker(row) or str(default_ticker).upper().replace("IDX:", "").strip()
        if not ticker:
            continue
        company = _first(row, "name", "company_name", "description")
        if isinstance(company, dict):
            company = _first(company, "name", "description")
        normalized.append({
            "kode saham": ticker,
            "symbol": f"IDX:{ticker}",
            "nama perusahaan": str(company or ticker).strip(),
            "close": _number(_first(row, "close", "last", "last_price", "price")),
            "open": _number(_first(row, "open")),
            "high": _number(_first(row, "high")),
            "low": _number(_first(row, "low")),
            "volume": _number(_first(row, "volume")),
            "change": _number(_first(row, "change")),
            "change_pct": _change_pct(row),
            "date": _first(row, "date", "trading_date"),
            "source": "GOAPI IDX",
        })
    return normalized


@st.cache_data(ttl=300, show_spinner=False)
def fetch_goapi_prices(symbols: tuple[str, ...]) -> list[dict[str, Any]]:
    """Fetch up to GOAPI's documented 50 symbols per prices request."""
    clean = tuple(dict.fromkeys(str(item).upper().replace("IDX:", "").strip() for item in symbols if item))
    if not clean:
        return []
    if len(clean) > 50:
        clean = clean[:50]
    payload = _request("/stock/idx/prices", {"symbols": ",".join(clean)})
    return normalize_price_rows(_extract_rows(payload))


def fetch_goapi_snapshot(symbol: str) -> dict[str, Any]:
    rows = fetch_goapi_prices((str(symbol).upper().replace("IDX:", "").strip(),))
    return rows[0] if rows else {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_goapi_historical(symbol: str, from_date: str = "", to_date: str = "") -> pd.DataFrame:
    params = {}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    payload = _request(f"/stock/idx/{str(symbol).upper().strip()}/historical", params)
    rows = normalize_price_rows(_extract_rows(payload), default_ticker=symbol)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    for column in ("open", "high", "low", "close", "volume", "change", "change_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date", kind="stable").drop_duplicates("date").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_goapi_indicators(page: int = 1, date: str = "") -> list[dict[str, Any]]:
    params: dict[str, Any] = {"page": str(max(1, int(page)))}
    if date:
        params["date"] = date
    return _extract_rows(_request("/stock/idx/indicators", params))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_goapi_broker_summary(symbol: str, trading_date: str, investor: str = "ALL") -> list[dict[str, Any]]:
    payload = _request(
        f"/stock/idx/{str(symbol).upper().strip()}/broker_summary",
        {"date": trading_date, "investor": investor},
    )
    return _extract_rows(payload)


def clear_goapi_cache() -> None:
    for function in (fetch_goapi_prices, fetch_goapi_historical, fetch_goapi_indicators, fetch_goapi_broker_summary):
        function.clear()
