import io
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from analytics.ai import build_runtime_context, call_openrouter_structured
from analytics.indodax import fetch_indodax_coins
from analytics.validator import validate_analysis
from storage.persistence import save_analysis

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Meme Coin Screener", layout="wide")

FIELDS = [
    "name", "description", "close", "change", "volume", "Value.Traded",
    "RSI", "RSI[1]", "MACD.macd", "MACD.signal", "BB.upper", "BB.lower",
    "BB.basis", "EMA20", "EMA50", "EMA200", "Stoch.K", "Stoch.D", "ADX",
    "ADX+DI", "ADX-DI", "Recommend.All", "Recommend.MA", "Recommend.Other",
    "Perf.W", "Perf.1M", "relative_volume_10d_calc", "exchange", "type",
]
MEME_HINTS = {
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "BOME", "TURBO",
    "NEIRO", "BRETT", "MOG", "POPCAT", "MEW", "BABYDOGE", "PENGU", "TRUMP",
    "1000SATS", "1000RATS", "CAT", "DOGS", "NOT", "ACT", "PNUT", "FARTCOIN",
}
DATA_DIR = "data"
WATCHLIST_FILE = os.path.join(DATA_DIR, "crypto_watchlist.json")
HISTORY_FILE = os.path.join(DATA_DIR, "crypto_history.json")


def safe_num(value, default=np.nan):
    try:
        number = float(value)
        return default if np.isnan(number) or np.isinf(number) else number
    except (TypeError, ValueError):
        return default


def price_idr(value):
    number = safe_num(value)
    if np.isnan(number):
        return "N/A"
    if number >= 1_000_000:
        return f"Rp {number:,.0f}"
    if number >= 1_000:
        return f"Rp {number:,.0f}"
    if number >= 1:
        return f"Rp {number:,.2f}"
    return f"Rp {number:,.6f}"


def price_usd(value):
    number = safe_num(value)
    if np.isnan(number):
        return "N/A"
    if abs(number) >= 100:
        return f"${number:,.2f}"
    if abs(number) >= 1:
        return f"${number:,.4f}"
    return f"${number:,.10g}"


def compact(value):
    number = safe_num(value)
    if np.isnan(number):
        return "N/A"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= size:
            return f"{number / size:.2f}{suffix}"
    return f"{number:.2f}"


def compact_idr(value):
    number = safe_num(value)
    if np.isnan(number):
        return "N/A"
    for size, suffix in ((1e12, "T"), (1e9, "M"), (1e6, "Jt"), (1e3, "Rb")):
        if abs(number) >= size:
            return f"Rp {number / size:.1f}{suffix}"
    return f"Rp {number:,.0f}"


def pct(value):
    number = safe_num(value)
    return "N/A" if np.isnan(number) else f"{number:+.2f}%"


def recommendation(value):
    number = safe_num(value)
    if np.isnan(number):
        return "N/A"
    if number >= 0.5:
        return "Strong Buy"
    if number >= 0.1:
        return "Buy"
    if number >= -0.1:
        return "Neutral"
    if number >= -0.5:
        return "Sell"
    return "Strong Sell"


def percentile(series):
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(50.0, index=clean.index) if clean.notna().sum() <= 1 else clean.rank(pct=True) * 100


def scan_headers():
    return {
        "Content-Type": "application/json",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/",
        "User-Agent": "Mozilla/5.0",
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto(limit=1000):
    payload = {
        "columns": FIELDS,
        "filter": [
            {"left": "type", "operation": "equal", "right": "crypto"},
            {"left": "exchange", "operation": "equal", "right": "BINANCE"},
        ],
        "range": [0, int(limit)],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "symbols": {"query": {"types": []}, "tickers": []},
    }
    response = requests.post(
        "https://scanner.tradingview.com/crypto/scan",
        json=payload,
        headers=scan_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def normalize(rows):
    records = []
    for item in rows:
        values = item.get("d", [])
        raw = {field: values[index] if index < len(values) else None for index, field in enumerate(FIELDS)}
        symbol = str(item.get("s") or "")
        ticker = str(raw.get("name") or symbol.split(":")[-1]).upper()
        if not ticker.endswith(("USDT", "USDC", "USD")):
            continue
        base = ticker.removesuffix("USDT").removesuffix("USDC").removesuffix("USD")
        if base not in MEME_HINTS:
            continue
        close = safe_num(raw.get("close"))
        ema_values = [safe_num(raw.get(key)) for key in ("EMA20", "EMA50", "EMA200")]
        ema_score = sum(close > value for value in ema_values if not np.isnan(value)) / 3 * 100
        rec = recommendation(raw.get("Recommend.All"))
        rec_score = {"Strong Buy": 100, "Buy": 78, "Neutral": 50, "Sell": 28, "Strong Sell": 8}.get(rec, 50)
        records.append({
            "symbol": symbol or f"BINANCE:{ticker}", "ticker": ticker, "coin": base,
            "name": str(raw.get("description") or base), "close": close,
            "change_pct": safe_num(raw.get("change")), "volume": safe_num(raw.get("volume")),
            "value_traded": safe_num(raw.get("Value.Traded")), "rel_volume": safe_num(raw.get("relative_volume_10d_calc"), 1),
            "rsi": safe_num(raw.get("RSI")), "adx": safe_num(raw.get("ADX")),
            "macd": safe_num(raw.get("MACD.macd")), "macd_signal": safe_num(raw.get("MACD.signal")),
            "ema20": ema_values[0], "ema50": ema_values[1], "ema200": ema_values[2],
            "perf_week": safe_num(raw.get("Perf.W")), "perf_month": safe_num(raw.get("Perf.1M")),
            "recommendation": rec, "rec_score": rec_score, "ema_score": ema_score,
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["momentum_score"] = percentile(frame["change_pct"]) * .45 + percentile(frame["perf_week"]) * .35 + percentile(frame["perf_month"]) * .20
    frame["liquidity_score"] = percentile(frame["value_traded"]) * .7 + percentile(frame["volume"]) * .3
    frame["volume_score"] = np.clip(frame["rel_volume"].fillna(1) * 45, 0, 100)
    frame["trend_score"] = frame["rec_score"] * .55 + frame["ema_score"] * .45
    frame["score"] = frame["trend_score"] * .35 + frame["momentum_score"] * .30 + frame["liquidity_score"] * .20 + frame["volume_score"] * .15
    frame["signal"] = np.select(
        [frame["score"] >= 75, frame["score"] >= 65, frame["score"] >= 55],
        ["Breakout Candidate", "Trending", "Watch"], default="Avoid",
    )
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def chart_url(symbol, interval="D"):
    return "https://s.tradingview.com/widgetembed/?" + urlencode({
        "symbol": symbol, "interval": interval, "theme": "dark", "style": "1",
        "timezone": "Etc/UTC", "locale": "id", "withdateranges": "1", "hide_volume": "0",
    })


def render_ai(row, api_key, model, is_indodax=False):
    ticker = row["ticker"]
    if not api_key:
        st.warning("Isi 9Router API key di sidebar.")
        return
    if st.button("Generate Analisis AI", width="stretch", key=f"ai_{ticker}"):
        snapshot = {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in row.to_dict().items()}
        mode = "indodax_idr" if is_indodax else "auto_tradingview"
        context = build_runtime_context(ticker, row["name"], snapshot, snapshot, {
            "observations": 0, "status": "insufficient_data", "indicators": {},
            "levels": {"support": [], "resistance": [], "fibonacci": []}, "execution": {}, "history": [],
        }, {}, {"mode": mode, "tv_history_available": False})
        with st.spinner("9Router menganalisis meme coin..."):
            outcome = call_openrouter_structured(context, api_key, model)
        if not outcome.get("ok"):
            st.error(outcome.get("error"))
            return
        analysis, warnings = validate_analysis(outcome["data"], ticker, pd.DataFrame(), context["quant_context"])
        save_analysis(ticker, context, analysis)
        st.session_state[f"ai_{ticker}"] = analysis
        if warnings:
            st.warning("; ".join(warnings))
    analysis = st.session_state.get(f"ai_{ticker}")
    if analysis:
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", analysis.get("verdict", "N/A"))
        c2.metric("Confidence", f"{safe_num(analysis.get('confidence'), 0):.0f}/100")
        c3.metric("Data Quality", analysis.get("data_quality", {}).get("status", "N/A"))
        st.info(analysis.get("summary", ""))
        st.write(analysis.get("market_structure", {}))
        st.subheader("Skenario")
        st.dataframe(pd.DataFrame(analysis.get("scenarios", [])), width="stretch", hide_index=True)
        st.warning("Meme coin sangat volatil dan berisiko rug pull. Bukan rekomendasi investasi.")


def main():
    st.title("Meme Coin Crypto Screener")
    with st.sidebar:
        st.header("Sumber Data")
        source = st.radio(
            "Pilih sumber:",
            ["Indodax (IDR)", "TradingView Binance (USDT)"],
            index=0,
        )
        is_indodax = source.startswith("Indodax")

        st.header("Setup")
        static_key = os.getenv("NINEROUTER_API_KEY", "").strip()
        api_key = st.text_input("9Router API Key", value=static_key, type="password")
        model = st.text_input("9Router Model", value=os.getenv("NINEROUTER_MODEL", "combo-1"))

        if is_indodax:
            st.caption("Data dari Indodax API. Harga dalam Rupiah.")
            min_volume = st.number_input("Min volume IDR", 0, 100_000_000_000, 100_000_000, 10_000_000)
        else:
            limit = st.number_input("Ambil pair", 100, 5000, 1000, 100)
            min_volume = st.number_input("Min value traded USD", 0, 1_000_000_000, 100_000, 100_000)

        min_score = st.slider("Min score", 0, 100, 50)
        top_n = st.number_input("Top N", 5, 200, 50, 5)
        search = st.text_input("Cari coin", placeholder="DOGE, SHIB, PEPE")

        if is_indodax:
            if st.button("Refresh Indodax", width="stretch"):
                fetch_indodax_data.clear()
                st.rerun()
        else:
            if st.button("Refresh TradingView", width="stretch"):
                fetch_crypto.clear()
                st.rerun()

    if is_indodax:
        st.caption("Source: Indodax.com | Harga dalam Rupiah (IDR)")
        try:
            coins = fetch_indodax_data(min_volume)
        except Exception as exc:
            st.error(f"Indodax API gagal: {exc}")
            st.stop()
        if coins.empty:
            st.warning("Meme coin Indodax tidak ditemukan.")
            st.stop()
    else:
        st.caption("Source: TradingView Binance | Harga dalam USDT")
        try:
            body = fetch_crypto(limit)
            coins = normalize(body.get("data", []))
        except Exception as exc:
            st.error(f"TradingView crypto scanner gagal: {exc}")
            st.stop()
        if coins.empty:
            st.warning("Meme coin Binance tidak ditemukan.")
            st.stop()

    view = coins[(coins["score"] >= min_score) & (coins["value_traded"].fillna(0) >= min_volume)].copy()
    if search:
        view = view[view["ticker"].str.contains(search.upper(), na=False) | view["name"].str.upper().str.contains(search.upper(), na=False)]
    view = view.head(int(top_n))

    m1, m2, m3 = st.columns(3)
    m1.metric("Meme Coin Terdeteksi", len(coins))
    m2.metric("Ditampilkan", len(view))
    if is_indodax:
        m3.metric("Top Volume", compact_idr(coins["value_traded"].max()))
    else:
        m3.metric("Top Volume", f"${compact(coins['value_traded'].max())}")

    display_cols = ["ticker", "name", "close", "change_pct", "value_traded", "rsi", "recommendation", "score", "signal"]
    if "spread_pct" in view.columns:
        display_cols.append("spread_pct")
    display = view[[c for c in display_cols if c in view.columns]].copy()

    if is_indodax:
        display["close"] = display["close"].map(price_idr)
        display["value_traded"] = display["value_traded"].map(compact_idr)
        display["change_pct"] = display["change_pct"].map(pct)
        if "spread_pct" in display.columns:
            display["spread_pct"] = display["spread_pct"].map(lambda x: f"{x:.2f}%")
            display = display.rename(columns={"spread_pct": "Spread"})
    else:
        display["close"] = display["close"].map(price_usd)
        display["value_traded"] = display["value_traded"].map(lambda x: f"${compact(x)}")
        display["change_pct"] = display["change_pct"].map(pct)

    st.dataframe(display, width="stretch", hide_index=True)

    if view.empty:
        st.info("Tidak ada coin sesuai filter.")
        return

    selected_ticker = st.selectbox("Pilih meme coin", view["ticker"])
    selected = view[view["ticker"] == selected_ticker].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    if is_indodax:
        c1.metric("Harga", price_idr(selected["close"]))
        if "high_24h" in selected and "low_24h" in selected:
            c2.metric("24h Range", f"{price_idr(selected['low_24h'])} - {price_idr(selected['high_24h'])}")
        else:
            c2.metric("24h", pct(selected["change_pct"]))
    else:
        c1.metric("Harga", price_usd(selected["close"]))
        c2.metric("24h", pct(selected["change_pct"]))
    c3.metric("Score", f"{selected['score']:.1f}")
    c4.metric("Sinyal", selected["signal"])

    if is_indodax and "min_order" in selected:
        st.caption(f"Min order: {price_idr(selected['min_order'])} | Fee: {selected.get('fee_pct', 0.2)}% | Spread: {selected.get('spread_pct', 0):.2f}%")

    watchlist = read_json(WATCHLIST_FILE)
    item = {"id": selected["symbol"], "type": "crypto", "ticker": selected_ticker, "name": selected["name"], "added_at": datetime.now(timezone.utc).isoformat()}
    if any(old.get("id") == item["id"] for old in watchlist):
        if st.button("Hapus Watchlist"):
            write_json(WATCHLIST_FILE, [old for old in watchlist if old.get("id") != item["id"]])
            st.rerun()
    elif st.button("Tambah Watchlist"):
        write_json(WATCHLIST_FILE, watchlist + [item])
        history = read_json(HISTORY_FILE)
        history.append({"ts": datetime.now(timezone.utc).isoformat(), "type": "crypto", "ticker": selected_ticker, "price": selected["close"], "score": selected["score"]})
        write_json(HISTORY_FILE, history[-5000:])
        st.rerun()

    chart_tab, ai_tab, watch_tab = st.tabs(["Chart", "AI 9Router", "Watchlist"])
    with chart_tab:
        if is_indodax:
            tradingview_symbol = selected["symbol"].replace("INDODAX:", "INDODAX:")
            components.iframe(chart_url(tradingview_symbol), height=720)
            st.info("Chart Indodax via TradingView embed. Jika tidak muncul, coin mungkin belum terdaftar di TradingView.")
        else:
            components.iframe(chart_url(selected["symbol"]), height=720)
    with ai_tab:
        render_ai(selected, api_key.strip(), model.strip(), is_indodax=is_indodax)
    with watch_tab:
        current = read_json(WATCHLIST_FILE)
        st.dataframe(pd.DataFrame(current), width="stretch", hide_index=True) if current else st.info("Watchlist kosong.")

    st.caption("Output hanya alat bantu analisis, bukan rekomendasi investasi.")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_indodax_data(min_volume):
    return fetch_indodax_coins(min_volume=min_volume)


if __name__ == "__main__":
    main()
