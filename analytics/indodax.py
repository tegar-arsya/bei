"""Indodax meme coin data source with technical indicators from trade data."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://indodax.com"

MEME_COINS = {
    "doge", "shib", "pepe", "bonk", "wif", "bome", "floki", "trump",
    "pengu", "fartcoin", "popcat", "mog", "brett", "trollsol", "pump",
    "spx", "useless", "aixbt", "anoa", "jellyjelly", "velvet",
}


def fetch_pairs() -> list[dict[str, Any]]:
    r = requests.get(f"{BASE_URL}/api/pairs", timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_ticker_all() -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}/api/ticker_all", timeout=15)
    r.raise_for_status()
    return r.json().get("tickers", {})


def fetch_trades(pair_id: str, limit: int = 500) -> list[dict[str, Any]]:
    r = requests.get(f"{BASE_URL}/api/trades/{pair_id}", timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data[:limit]
    return []


def _trades_to_candles(trades: list[dict[str, Any]], resolution_minutes: int = 60) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    records = []
    for t in trades:
        try:
            records.append({
                "timestamp": int(t["date"]),
                "price": float(t["price"]),
                "volume": float(t["amount"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df["time_bin"] = df["timestamp"] // (resolution_minutes * 60)
    candles = df.groupby("time_bin").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    candles["time"] = candles["time_bin"] * (resolution_minutes * 60)
    return candles.sort_values("time").reset_index(drop=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line, signal_line


def compute_indicators(candles: pd.DataFrame) -> dict[str, Any]:
    if len(candles) < 3:
        return {}
    close = candles["close"]
    ema20 = _ema(close, 20).iloc[-1] if len(close) >= 20 else np.nan
    ema50 = _ema(close, 50).iloc[-1] if len(close) >= 50 else np.nan
    rsi = _rsi(close).iloc[-1]
    macd_line, signal_line = _macd(close)
    macd_val = macd_line.iloc[-1]
    macd_sig = signal_line.iloc[-1]
    return {
        "rsi": round(rsi, 2) if not np.isnan(rsi) else None,
        "ema20": round(ema20, 8) if not np.isnan(ema20) else None,
        "ema50": round(ema50, 8) if not np.isnan(ema50) else None,
        "macd": round(macd_val, 8) if not np.isnan(macd_val) else None,
        "macd_signal": round(macd_sig, 8) if not np.isnan(macd_sig) else None,
    }


def fetch_indodax_coins(min_volume: int = 10_000_000) -> pd.DataFrame:
    pairs = {p["traded_currency"]: p for p in fetch_pairs() if p.get("base_currency") == "idr"}
    tickers = fetch_ticker_all()

    records = []
    for ticker_id, ticker in tickers.items():
        coin = ticker_id.replace("_idr", "")
        if coin not in MEME_COINS:
            continue
        pair_info = pairs.get(coin, {})
        vol_idr = float(ticker.get("vol_idr", 0))
        last = float(ticker.get("last", 0))
        high = float(ticker.get("high", 0))
        low = float(ticker.get("low", 0))
        buy = float(ticker.get("buy", 0))
        sell = float(ticker.get("sell", 0))
        if vol_idr < min_volume or last <= 0:
            continue

        pair_id = pair_info.get("id", f"{coin}idr")
        trades = fetch_trades(pair_id, limit=500)
        candles = _trades_to_candles(trades, resolution_minutes=60)
        indicators = compute_indicators(candles) if not candles.empty else {}

        name = ticker.get("name") or pair_info.get("description", coin.upper())
        spread_pct = ((sell - buy) / last * 100) if last > 0 else 0

        ema20 = indicators.get("ema20")
        ema50 = indicators.get("ema50")
        ema_score = 0
        ema_count = 0
        if ema20 is not None and not np.isnan(ema20):
            ema_score += (1 if last > ema20 else 0)
            ema_count += 1
        if ema50 is not None and not np.isnan(ema50):
            ema_score += (1 if last > ema50 else 0)
            ema_count += 1
        ema_pct = (ema_score / ema_count * 100) if ema_count else 50.0

        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi >= 70:
                rec = "Sell"
                rec_score = 28
            elif rsi >= 60:
                rec = "Neutral"
                rec_score = 50
            elif rsi >= 40:
                rec = "Buy"
                rec_score = 78
            else:
                rec = "Oversold"
                rec_score = 90
        else:
            rec = "Neutral"
            rec_score = 50

        change_pct = ((last - low) / low * 100) if low > 0 else 0

        records.append({
            "symbol": f"INDODAX:{coin.upper()}IDR",
            "ticker": f"{coin.upper()}/IDR",
            "coin": coin.upper(),
            "name": name,
            "close": last,
            "change_pct": change_pct,
            "volume": vol_idr,
            "value_traded": vol_idr,
            "rel_volume": 1.0,
            "rsi": rsi,
            "adx": None,
            "macd": indicators.get("macd"),
            "macd_signal": indicators.get("macd_signal"),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": None,
            "perf_week": None,
            "perf_month": None,
            "recommendation": rec,
            "rec_score": rec_score,
            "ema_score": ema_pct,
            "spread_pct": round(spread_pct, 2),
            "high_24h": high,
            "low_24h": low,
            "buy": buy,
            "sell": sell,
            "min_order": pair_info.get("trade_min_base_currency", 10000),
            "fee_pct": pair_info.get("trade_fee_percent", 0.2),
            "source": "indodax",
        })

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    def _pct_rank(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return pd.Series(50.0, index=clean.index) if clean.notna().sum() <= 1 else clean.rank(pct=True) * 100

    frame["momentum_score"] = _pct_rank(frame["change_pct"]) * 0.5 + _pct_rank(frame["volume"]) * 0.5
    frame["liquidity_score"] = _pct_rank(frame["value_traded"]) * 0.7 + _pct_rank(frame["volume"]) * 0.3
    frame["volume_score"] = np.clip(frame["rel_volume"].fillna(1) * 45, 0, 100)
    frame["trend_score"] = frame["rec_score"] * 0.55 + frame["ema_score"] * 0.45
    frame["score"] = (
        frame["trend_score"] * 0.35
        + frame["momentum_score"] * 0.30
        + frame["liquidity_score"] * 0.20
        + frame["volume_score"] * 0.15
    )
    frame["signal"] = np.select(
        [frame["score"] >= 75, frame["score"] >= 65, frame["score"] >= 55],
        ["Breakout Candidate", "Trending", "Watch"],
        default="Avoid",
    )
    return frame.sort_values("score", ascending=False).reset_index(drop=True)
