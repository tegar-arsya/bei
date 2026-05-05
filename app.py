import io
import json
import os
import requests
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlencode
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Market Screener", layout="wide")

REQUIRED_COLUMNS = {
    "ringkasan_saham": [
        "kode saham", "nama perusahaan", "sebelumnya", "open price",
        "tanggal perdagangan terakhir", "first trade", "tertinggi", "terendah",
        "penutupan", "selisih", "volume", "nilai", "frekuensi",
        "offer", "offer volume", "bid", "bid volume", "listed shares",
        "tradeble shares", "weight for index", "foreign sell", "foreign buy",
        "non regular volume", "non regular value", "non regular frequency",
    ],
    "ringkasan_broker":      ["kode perusahaan", "nama perusahaan", "volume", "nilai", "frekuensi"],
    "ringkasan_perdagangan": ["id instrument", "id board", "volume", "nilai", "frekuensi"],
    "daftar_saham":          ["kode", "nama perusahaan", "tanggal pencatatan", "saham", "papan pencatatan"],
}
ALTERNATE_COLUMNS = {
    "daftar_saham": ["id instrument", "id board", "volume", "nilai", "frekuensi"],
}
INDEX_REQUIRED_COLUMNS = [
    "kode indeks", "sebelumnya", "tertinggi", "terendah",
    "penutupan", "selisih", "volume", "nilai", "frekuensi",
]
DEFAULT_WEIGHTS = {
    "momentum": 0.25, "liquidity": 0.20, "flow": 0.20,
    "market_activity": 0.15, "volume_trend": 0.10,
    "price_structure": 0.05, "broker": 0.05,
}

TV_FIELDS = [
    "name", "close", "change", "volume",
    "RSI", "RSI[1]",
    "MACD.macd", "MACD.signal",
    "BB.upper", "BB.lower", "BB.basis",
    "EMA20", "EMA50", "EMA200",
    "Stoch.K", "Stoch.D",
    "ADX", "ADX+DI", "ADX-DI",
    "Recommend.All", "Recommend.MA", "Recommend.Other",
    "CCI20",
    "Perf.W", "Perf.1M",
    "relative_volume_10d_calc",
    "Mom", "AO",
]
STOCK_AUTO_FIELDS = [
    "name", "description", "close", "change", "volume", "Value.Traded", "market_cap_basic",
    "RSI", "RSI[1]",
    "MACD.macd", "MACD.signal",
    "BB.upper", "BB.lower", "BB.basis",
    "EMA20", "EMA50", "EMA200",
    "Stoch.K", "Stoch.D",
    "ADX", "ADX+DI", "ADX-DI",
    "Recommend.All", "Recommend.MA", "Recommend.Other",
    "CCI20",
    "Perf.W", "Perf.1M",
    "relative_volume_10d_calc",
    "Mom", "AO",
    "sector", "type",
]
STOCK_SORT_OPTIONS = {
    "Nilai transaksi": "Value.Traded",
    "Volume": "volume",
    "Momentum harian": "change",
    "Performa 1 minggu": "Perf.W",
    "Performa 1 bulan": "Perf.1M",
    "Market cap": "market_cap_basic",
}

APP_PAGES = ["Home", "Saham BEI", "Crypto Market", "Meme Coin Radar", "Watchlist & Alerts"]
DEX_CHAIN_OPTIONS = {
    "Semua chain": "all",
    "Solana": "solana",
    "Base": "base",
    "BSC": "bsc",
    "Ethereum": "ethereum",
    "Arbitrum": "arbitrum",
    "Polygon": "polygon",
}
BINANCE_QUOTES = ["IDR"]
CRYPTO_MARKET_SOURCES = ["Indodax IDR", "CoinGecko IDR"]
COINGECKO_VS_MAP = {
    "USDT": "usd",
    "FDUSD": "usd",
    "USDC": "usd",
    "BTC": "btc",
    "ETH": "eth",
    "BNB": "bnb",
    "IDR": "idr",
}
DATA_DIR = "data"
WATCHLIST_FILE = os.path.join(DATA_DIR, "crypto_watchlist.json")
HISTORY_FILE = os.path.join(DATA_DIR, "crypto_history.json")
ALERT_RULES_FILE = os.path.join(DATA_DIR, "crypto_alert_rules.json")
MAX_HISTORY_ROWS = 5000
DEX_SOURCE_OPTIONS = [
    "Search",
    "Latest Profiles",
    "Latest Boosted",
    "Top Boosted",
    "Watchlist",
]
MEME_COIN_BASES = {
    "ACT", "BABYDOGE", "BOME", "BONK", "BRETT", "CAT", "DEGEN", "DOGE",
    "FLOKI", "MEME", "MOG", "NEIRO", "PEOPLE", "PENGU", "PEPE", "POPCAT",
    "SHIB", "TOSHI", "TURBO", "WIF",
}
CHAIN_SECURITY_IDS = {
    "ethereum": "1",
    "bsc": "56",
    "base": "8453",
    "arbitrum": "42161",
    "polygon": "137",
    "optimism": "10",
    "avalanche": "43114",
}
HONEYPOT_CHAIN_IDS = {"ethereum": "1", "bsc": "56", "base": "8453"}
DEFAULT_ALERT_RULES = {
    "stock_min_score": 72.0,
    "stock_min_change": 3.0,
    "stock_dump_change": -4.0,
    "stock_min_rel_volume": 1.5,
    "dex_min_radar": 70.0,
    "dex_max_risk": 55.0,
    "dex_min_liquidity": 25_000.0,
    "dex_min_h1_change": 12.0,
    "dex_dump_h1_change": -25.0,
    "dex_min_buy_ratio": 0.55,
    "cex_min_score": 72.0,
    "cex_min_24h_change": 6.0,
    "cex_dump_24h_change": -8.0,
}

def tv_rec_label(val):
    try:
        if val is None: return "N/A"
        v = float(val)
        if v != v: return "N/A"
        if v >= 0.5:  return "Strong Buy"
        if v >= 0.1:  return "Buy"
        if v >= -0.1: return "Neutral"
        if v >= -0.5: return "Sell"
        return "Strong Sell"
    except: return "N/A"

def tv_rec_emoji(label):
    return {"Strong Buy":"\U0001f7e2\U0001f7e2","Buy":"\U0001f7e2","Neutral":"\U0001f7e1",
            "Sell":"\U0001f534","Strong Sell":"\U0001f534\U0001f534"}.get(label,"\u26aa")

def normalize_column_name(name):
    return " ".join(str(name).strip().lower().replace("_"," ").split())

def normalize_columns(df):
    return df.rename(columns={c: normalize_column_name(c) for c in df.columns})

def find_col(df, target):
    t = normalize_column_name(target)
    for col in df.columns:
        if normalize_column_name(col) == t:
            return col
    raise KeyError(f"Kolom '{target}' tidak ditemukan")

def to_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    raw = series.astype(str).str.strip().replace({"": np.nan, "-": np.nan, "--": np.nan})
    raw = raw.str.replace("%","",regex=False).str.replace(" ","",regex=False)
    p1 = pd.to_numeric(raw, errors="coerce")
    p2 = pd.to_numeric(raw.str.replace(".","",regex=False).str.replace(",",".",regex=False),errors="coerce")
    p3 = pd.to_numeric(raw.str.replace(",","",regex=False),errors="coerce")
    return p1.fillna(p2).fillna(p3)

def percentile_series(s):
    s = s.copy().replace([np.inf,-np.inf],np.nan)
    if s.notna().sum() <= 1:
        return pd.Series([50.0]*len(s), index=s.index)
    return s.rank(pct=True)*100

def safe_col(df, col): return to_numeric(df[find_col(df,col)])

def safe_num(v, default=0.0):
    try: return default if pd.isna(v) else float(v)
    except: return default

def has_columns(df, cols):
    ex = {normalize_column_name(c) for c in df.columns}
    return {normalize_column_name(c) for c in cols}.issubset(ex)

def validate_columns(df, required_cols):
    ex = [normalize_column_name(c) for c in df.columns]
    missing = [c for c in required_cols if c not in ex]
    return len(missing)==0, missing

def color_signal(val):
    if isinstance(val, float):
        if val>=75: return "background-color:#1a472a;color:#90ee90"
        if val>=60: return "background-color:#1e3a5f;color:#87ceeb"
        if val>=45: return "background-color:#3d2b00;color:#ffd700"
        return "background-color:#3d0000;color:#ff9999"
    return ""


def render_df_with_style_fallback(df, subset_cols):
    try:
        st.dataframe(df.style.applymap(color_signal, subset=subset_cols), width="stretch")
    except Exception:
        st.dataframe(df, width="stretch")

def factor_label(score):
    if pd.isna(score): return "N/A"
    v = float(score)
    if v>=75: return "Sangat Kuat"
    if v>=60: return "Kuat"
    if v>=45: return "Netral"
    return "Lemah"


def get_static_openrouter_key() -> str:
    key = ""
    try:
        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key


def get_openrouter_model() -> str:
    model = ""
    try:
        model = str(st.secrets.get("OPENROUTER_MODEL", "")).strip()
    except Exception:
        model = ""
    if not model:
        model = os.getenv("OPENROUTER_MODEL", "").strip()
    return model if model else "openrouter/auto"

# ── TRADINGVIEW ──
@st.cache_data(ttl=300, show_spinner=False)
def fetch_tv_data(ticker):
    symbol  = f"IDX:{ticker.upper().strip()}"
    url     = "https://scanner.tradingview.com/indonesia/scan"
    payload = {"symbols":{"tickers":[symbol],"query":{"types":[]}},"columns":TV_FIELDS}
    headers = {"Content-Type":"application/json","Origin":"https://www.tradingview.com",
                "Referer":"https://www.tradingview.com/",
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"): return None
        row = data["data"][0]["d"]
        result = {field:row[i] for i,field in enumerate(TV_FIELDS)}
        result["symbol"] = symbol
        return result
    except: return None

def parse_tv_data(tv):
    if not tv: return {}
    close  = safe_num(tv.get("close"),  np.nan)
    ema20  = safe_num(tv.get("EMA20"),  np.nan)
    ema50  = safe_num(tv.get("EMA50"),  np.nan)
    ema200 = safe_num(tv.get("EMA200"), np.nan)
    bb_up  = safe_num(tv.get("BB.upper"),  np.nan)
    bb_low = safe_num(tv.get("BB.lower"),  np.nan)
    bb_mid = safe_num(tv.get("BB.basis"),  np.nan)
    macd_v = safe_num(tv.get("MACD.macd"),   np.nan)
    macd_s = safe_num(tv.get("MACD.signal"), np.nan)
    bb_pos = None
    if not any(pd.isna(x) for x in [close,bb_low,bb_up]) and (bb_up-bb_low)>0:
        bb_pos = (close-bb_low)/(bb_up-bb_low)*100
    above_ema20  = (close>ema20)  if not any(pd.isna([close,ema20]))  else None
    above_ema50  = (close>ema50)  if not any(pd.isna([close,ema50]))  else None
    above_ema200 = (close>ema200) if not any(pd.isna([close,ema200])) else None
    ema_count    = sum([1 if x else 0 for x in [above_ema20,above_ema50,above_ema200]])
    macd_cross   = None
    if not any(pd.isna([macd_v,macd_s])): macd_cross = "bullish" if macd_v>macd_s else "bearish"
    return {
        "close":close,"change_pct":safe_num(tv.get("change"),np.nan),
        "rec_all":tv_rec_label(tv.get("Recommend.All")),
        "rec_ma":tv_rec_label(tv.get("Recommend.MA")),
        "rec_other":tv_rec_label(tv.get("Recommend.Other")),
        "rsi":safe_num(tv.get("RSI"),np.nan),"rsi_prev":safe_num(tv.get("RSI[1]"),np.nan),
        "macd":macd_v,"macd_signal":macd_s,"macd_cross":macd_cross,
        "stoch_k":safe_num(tv.get("Stoch.K"),np.nan),"stoch_d":safe_num(tv.get("Stoch.D"),np.nan),
        "adx":safe_num(tv.get("ADX"),np.nan),
        "adx_plus":safe_num(tv.get("ADX+DI"),np.nan),"adx_minus":safe_num(tv.get("ADX-DI"),np.nan),
        "cci":safe_num(tv.get("CCI20"),np.nan),
        "mom":safe_num(tv.get("Mom"),np.nan),"ao":safe_num(tv.get("AO"),np.nan),
        "ema20":ema20,"ema50":ema50,"ema200":ema200,
        "bb_upper":bb_up,"bb_lower":bb_low,"bb_basis":bb_mid,"bb_position":bb_pos,
        "ema_trend_count":ema_count,
        "above_ema20":above_ema20,"above_ema50":above_ema50,"above_ema200":above_ema200,
        "rel_volume":safe_num(tv.get("relative_volume_10d_calc"),np.nan),
        "perf_week":safe_num(tv.get("Perf.W"),np.nan),"perf_month":safe_num(tv.get("Perf.1M"),np.nan),
    }

# ── OPENROUTER ──
def build_prompt(ticker, company, bei_row, tv, market_regime, md_ctx):
    close   = safe_num(bei_row.get("penutupan_num"), np.nan)
    chg_pct = safe_num(bei_row.get("change_pct"), np.nan)
    f_net   = safe_num(bei_row.get("foreign_net"), 0)
    f_ratio = safe_num(bei_row.get("foreign_net_ratio"), 0)
    bid_p   = safe_num(bei_row.get("bid_offer_pressure"), 0)
    atr_p   = max(safe_num(bei_row.get("atr_pct"), 2.0), 0.5)
    sig     = safe_num(bei_row.get("signal_strength", bei_row.get("final_score",0)), 0)
    fscore  = safe_num(bei_row.get("final_score", 0), 0)
    atr_abs = close * atr_p / 100 if not pd.isna(close) else 0

    fmt = lambda v, d=2: f"{v:.{d}f}" if not pd.isna(v) else "N/A"
    inv = fmt(close-atr_abs*1.5) if not pd.isna(close) else "N/A"
    tp1 = fmt(close+atr_abs*2.0) if not pd.isna(close) else "N/A"
    tp2 = fmt(close+atr_abs*3.5) if not pd.isna(close) else "N/A"

    md_sec = ""
    if md_ctx and md_ctx.get("days_data",1) > 1:
        md_sec = f"""
MULTI-HARI ({md_ctx["days_data"]} hari):
- Tren skor: {md_ctx["trend_slope"]:+.1f} pts/hari
- Konsistensi score >=60: {md_ctx["score_consistency"]:.0f}%
- Signal: {bei_row.get("signal_label","-")}"""

    reg_sec = ""
    if market_regime:
        reg_sec = (
            f"MARKET REGIME: {market_regime['regime_label']} "
            f"(score={market_regime['regime_score']:.1f}, IHSG {market_regime['index_change_pct']:+.2f}%)"
        )

    if tv:
        ema_desc = {3:"BULLISH - di atas semua EMA 20/50/200",2:"CUKUP BULLISH - di atas 2 dari 3 EMA",
                    1:"LEMAH - di atas 1 dari 3 EMA",0:"BEARISH - di bawah semua EMA"}
        rsi_note = "OVERBOUGHT" if tv.get("rsi",50)>70 else "OVERSOLD" if tv.get("rsi",50)<30 else "normal"
        adx_note = "tren kuat" if tv.get("adx",0)>25 else "sideways/lemah"
        tv_sec = f"""
TEKNIKAL REAL-TIME (TradingView):
- Rekomendasi: {tv.get("rec_all","N/A")} (MA:{tv.get("rec_ma","N/A")} | Osc:{tv.get("rec_other","N/A")})
- RSI(14): {fmt(tv.get("rsi"),1)} [{rsi_note}] (sebelumnya {fmt(tv.get("rsi_prev"),1)})
- MACD: crossover {tv.get("macd_cross","N/A")} | MACD={fmt(tv.get("macd"),4)} Signal={fmt(tv.get("macd_signal"),4)}
- Stochastic K/D: {fmt(tv.get("stoch_k"),1)}/{fmt(tv.get("stoch_d"),1)}
- ADX: {fmt(tv.get("adx"),1)} [{adx_note}] | +DI={fmt(tv.get("adx_plus"),1)} vs -DI={fmt(tv.get("adx_minus"),1)}
- CCI(20): {fmt(tv.get("cci"),1)} | Mom={fmt(tv.get("mom"),4)} | AO={fmt(tv.get("ao"),4)}
- EMA: {ema_desc.get(tv.get("ema_trend_count",0),"N/A")}
  EMA20={fmt(tv.get("ema20"))} | EMA50={fmt(tv.get("ema50"))} | EMA200={fmt(tv.get("ema200"))}
- Bollinger: posisi {fmt(tv.get("bb_position"),1) if tv.get("bb_position") is not None else "N/A"}% (0=lower,100=upper)
  Lower={fmt(tv.get("bb_lower"))} | Mid={fmt(tv.get("bb_basis"))} | Upper={fmt(tv.get("bb_upper"))}
- Volume Relatif: {fmt(tv.get("rel_volume"),2)}x vs 10d avg
- Performa: 1W={fmt(tv.get("perf_week"),2)}% | 1M={fmt(tv.get("perf_month"),2)}%"""
    else:
        tv_sec = "TEKNIKAL: Tidak tersedia. Analisis dari data BEI saja."

    return f"""Kamu analis saham BEI spesialis swing trading 1-5 hari. Tulis analisis tajam, kuantitatif, dan anti-kontradiksi dalam bahasa Indonesia.

SAHAM: {ticker} - {company}

DATA BEI:
- Close: {fmt(close)} | Change: {fmt(chg_pct,2) if not pd.isna(chg_pct) else "N/A"}%
- ATR-1d: {atr_p:.2f}% = {atr_abs:.2f} absolut
- Net Foreign: {f_net:,.0f} (rasio: {f_ratio:.3f})
- Bid-Offer Pressure: {bid_p:+.3f} (>0=dominan buyer)
- Signal Strength: {sig:.1f}/100 | Final Score: {fscore:.1f}/100
- Momentum: {safe_num(bei_row.get("momentum_score"),50):.1f} | Likuiditas: {safe_num(bei_row.get("liquidity_score"),50):.1f} | Flow: {safe_num(bei_row.get("flow_score"),50):.1f}
- Market Activity: {safe_num(bei_row.get("market_activity_score"),50):.1f} | Vol Trend: {safe_num(bei_row.get("volume_trend_score"),50):.1f} | Price Structure: {safe_num(bei_row.get("price_structure_score"),50):.1f}
{md_sec}
{reg_sec}
{tv_sec}

LEVEL EKSEKUSI: Stop={inv} | TP1={tp1} | TP2={tp2}

ATURAN WAJIB:
- Jangan halusinasi level harga. Semua level harus konsisten dengan data di atas.
- Jika ada konflik data (contoh: indikator overbought tapi verdict bullish), jelaskan eksplisit sebagai trade-off.
- Jika confidence rendah, katakan tegas "WAIT".
- Gunakan angka aktual dari input, jangan placeholder.

FORMAT OUTPUT (WAJIB IKUTI):

**Verdict:** [1 kalimat tegas]

**Skor Keyakinan (0-100):**
- Total Confidence: [angka]
- Breakdown: Teknikal [x]/40 | Flow [x]/35 | Regime [x]/25
- Decision Tag: [AGRESIF BUY | BUY ON PULLBACK | WAIT | AVOID]

**Validasi Konsistensi Data:**
- Konsisten: [maks 3 poin]
- Konflik: [maks 3 poin]
- Kesimpulan konflik: [konflik masih acceptable atau tidak]

**Konfluensi Sinyal (Ringkas):**
[Apakah BEI vs TradingView searah/berlawanan, 2-4 kalimat padat]

**Kondisi Teknikal:**
[RSI, MACD, ADX, EMA, Bollinger, volume relatif, performa 1W/1M. Cantumkan angka kunci]

**Kondisi Flow:**
[Foreign, bid-offer pressure, market regime, likuiditas]

**Peta Timeframe:**
- Intraday bias: [Bullish/Netral/Bearish + alasan 1 kalimat]
- Swing 1-3 hari: [Bullish/Netral/Bearish + alasan 1 kalimat]
- Swing 1-2 minggu: [Bullish/Netral/Bearish + alasan 1 kalimat]

**Skenario Eksekusi (3 Skenario):**
1. Breakout continuation:
- Trigger:
- Entry zone:
- Stop:
- TP1/TP2:
- Probabilitas: [Low/Medium/High]
2. Pullback setup:
- Trigger:
- Entry zone:
- Stop:
- TP1/TP2:
- Probabilitas: [Low/Medium/High]
3. Failed breakout / invalid setup:
- Trigger invalidasi:
- Aksi:
- Probabilitas: [Low/Medium/High]

**Risk Memo:**
- Max risk per trade: [contoh 0.5%-1% modal, sesuaikan confidence]
- Volatility risk: [rendah/sedang/tinggi + alasan]
- Liquidity trap risk: [ada/tidak]
- Event risk: [ada/tidak]

**Rencana Final:**
- Entry final: [level + syarat]
- Stop final:
- TP final:
- R/R final:
- Holding plan: [berapa hari]
- Kapan review ulang: [trigger harga/indikator]

**Red Flag:**
[Jujur soal kelemahan setup. Jika tidak ada: tulis "Tidak ada red flag signifikan."]

Target 450-650 kata.
"""

@st.cache_data(ttl=600, show_spinner=False)
def call_openrouter(prompt, api_key, model, max_tokens=1200):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.35,
        "max_tokens": max_tokens,
        "top_p": 0.85,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sahambei.streamlit.app",
        "X-Title": "BEI Screener v3",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        body = resp.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content", "")
        if text:
            return text
        return "Model tidak mengembalikan konten teks."
    except requests.exceptions.HTTPError:
        code = resp.status_code
        if code == 400:
            return "Request invalid. Cek model/payload OpenRouter."
        if code == 401:
            return "API key OpenRouter tidak valid atau belum aktif."
        if code == 402:
            return "Kredit OpenRouter tidak cukup (payment required)."
        if code == 429:
            return "Rate limit OpenRouter. Tunggu lalu coba lagi."
        return f"HTTP Error {code}"
    except Exception as e: return f"Gagal: {e}"


# -- CRYPTO DATA --
def fetch_public_json(url, params=None, timeout=15):
    headers = {
        "Accept": "application/json",
        "User-Agent": "Market-Screener/1.0",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def format_compact(value, prefix=""):
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
    except Exception:
        return "N/A"
    sign = "-" if v < 0 else ""
    v_abs = abs(v)
    for size, suffix in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]:
        if v_abs >= size:
            return f"{sign}{prefix}{v_abs / size:.2f}{suffix}"
    return f"{sign}{prefix}{v_abs:.2f}"


def format_age(minutes):
    try:
        m = float(minutes)
        if pd.isna(m) or m < 0:
            return "N/A"
    except Exception:
        return "N/A"
    if m < 60:
        return f"{m:.0f} menit"
    if m < 1440:
        return f"{m / 60:.1f} jam"
    return f"{m / 1440:.1f} hari"


def pct_text(value):
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        return f"{v:+.2f}%"
    except Exception:
        return "N/A"


def ratio_text(value, suffix="x"):
    try:
        v = float(value)
        if pd.isna(v) or np.isinf(v):
            return "N/A"
        return f"{v:.2f}{suffix}"
    except Exception:
        return "N/A"


def price_text(value):
    try:
        v = float(value)
        if pd.isna(v) or np.isinf(v):
            return "N/A"
        if abs(v) >= 100:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:,.4f}"
        return f"{v:,.8g}"
    except Exception:
        return "N/A"


def format_idr(value, compact=True):
    if compact:
        return format_compact(value, "Rp")
    raw = price_text(value)
    return raw if raw == "N/A" else f"Rp{raw}"


def format_market_price(row, key="last_price"):
    quote = str(row.get("quote", "IDR")).upper()
    value = row.get(key)
    if quote == "IDR":
        return format_idr(value, compact=False)
    prefix = "$" if quote in {"USDT", "FDUSD", "USDC", "USD"} else ""
    raw = price_text(value)
    if raw == "N/A":
        return raw
    return f"{prefix}{raw}" if prefix else f"{raw} {quote}"


def format_market_amount(row, key="quote_volume"):
    quote = str(row.get("quote", "IDR")).upper()
    value = row.get(key)
    if quote == "IDR":
        return format_idr(value)
    prefix = "$" if quote in {"USDT", "FDUSD", "USDC", "USD"} else ""
    text = format_compact(value, prefix)
    return text if prefix or text == "N/A" else f"{text} {quote}"


def compact_address(value):
    text = str(value or "").strip()
    if len(text) <= 14:
        return text or "N/A"
    return f"{text[:6]}...{text[-6:]}"


def safe_json(value, default):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if parsed is not None else default
        except Exception:
            return default
    return value if value is not None else default


def extract_dex_links(info):
    info = safe_json(info, {})
    links = []
    if not isinstance(info, dict):
        return links
    for item in info.get("websites") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        label = item.get("label") or "Website"
        links.append({"label": label, "url": item.get("url"), "type": "website"})
    for item in info.get("socials") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        label = item.get("type") or item.get("label") or "Social"
        links.append({"label": str(label).title(), "url": item.get("url"), "type": "social"})
    return links


def link_summary(info):
    links = extract_dex_links(info)
    if not links:
        return "Tidak ada"
    labels = [item["label"] for item in links[:4]]
    extra = len(links) - len(labels)
    return ", ".join(labels) + (f" +{extra}" if extra > 0 else "")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def read_json_file(path, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_file(path, payload):
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_watchlist():
    items = read_json_file(WATCHLIST_FILE, [])
    return items if isinstance(items, list) else []


def save_watchlist(items):
    deduped = {}
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            deduped[item["id"]] = item
    write_json_file(WATCHLIST_FILE, list(deduped.values()))


def load_history():
    rows = read_json_file(HISTORY_FILE, [])
    return rows if isinstance(rows, list) else []


def save_history(rows):
    write_json_file(HISTORY_FILE, rows[-MAX_HISTORY_ROWS:])


def append_history_rows(rows):
    if not rows:
        return
    history = load_history()
    history.extend(rows)
    save_history(history)


def load_alert_rules():
    rules = DEFAULT_ALERT_RULES.copy()
    stored = read_json_file(ALERT_RULES_FILE, {})
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key in rules:
                rules[key] = safe_num(value, rules[key])
    return rules


def save_alert_rules(rules):
    clean = {key: safe_num(rules.get(key), default) for key, default in DEFAULT_ALERT_RULES.items()}
    write_json_file(ALERT_RULES_FILE, clean)


def build_local_backup_payload():
    return {
        "schema": "market_screener_backup_v1",
        "created_at": utc_now_iso(),
        "watchlist": load_watchlist(),
        "history": load_history(),
        "alert_rules": load_alert_rules(),
    }


def restore_local_backup_payload(payload, merge=True):
    if not isinstance(payload, dict) or payload.get("schema") != "market_screener_backup_v1":
        raise ValueError("Format backup tidak dikenali.")
    watchlist = payload.get("watchlist", [])
    history = payload.get("history", [])
    alert_rules = payload.get("alert_rules", {})
    if not isinstance(watchlist, list) or not isinstance(history, list) or not isinstance(alert_rules, dict):
        raise ValueError("Isi backup tidak valid.")

    if merge:
        existing_watchlist = {item.get("id"): item for item in load_watchlist() if isinstance(item, dict)}
        for item in watchlist:
            if isinstance(item, dict) and item.get("id"):
                existing_watchlist[item["id"]] = item
        save_watchlist(list(existing_watchlist.values()))

        existing_history = load_history()
        existing_keys = {
            (row.get("ts"), row.get("id"), str(row.get("price")), str(row.get("score")))
            for row in existing_history if isinstance(row, dict)
        }
        for row in history:
            if not isinstance(row, dict):
                continue
            key = (row.get("ts"), row.get("id"), str(row.get("price")), str(row.get("score")))
            if key not in existing_keys:
                existing_history.append(row)
                existing_keys.add(key)
        save_history(existing_history)

        merged_rules = load_alert_rules()
        for key, value in alert_rules.items():
            if key in DEFAULT_ALERT_RULES:
                merged_rules[key] = safe_num(value, DEFAULT_ALERT_RULES[key])
        save_alert_rules(merged_rules)
    else:
        save_watchlist(watchlist)
        save_history(history)
        merged_rules = DEFAULT_ALERT_RULES.copy()
        for key, value in alert_rules.items():
            if key in DEFAULT_ALERT_RULES:
                merged_rules[key] = safe_num(value, DEFAULT_ALERT_RULES[key])
        save_alert_rules(merged_rules)


def get_secret_source():
    try:
        if str(st.secrets.get("OPENROUTER_API_KEY", "")).strip():
            return "Streamlit secrets"
    except Exception:
        pass
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return ".env / environment"
    return "Belum diset"


def get_deploy_readiness_rows():
    secret_source = get_secret_source()
    try:
        ensure_data_dir()
        probe_path = os.path.join(DATA_DIR, ".write_probe")
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write(utc_now_iso())
        os.remove(probe_path)
        data_status = "OK"
        data_note = "Folder data/ bisa ditulis."
    except Exception as exc:
        data_status = "Error"
        data_note = f"Folder data/ tidak bisa ditulis: {exc}"

    rows = [
        {
            "Area": "Secrets",
            "Status": "OK" if secret_source != "Belum diset" else "Perlu set",
            "Catatan": f"OpenRouter source: {secret_source}",
        },
        {
            "Area": "Config",
            "Status": "OK" if os.path.exists(".streamlit/config.toml") else "Kurang",
            "Catatan": ".streamlit/config.toml tersedia untuk deploy.",
        },
        {
            "Area": "Storage",
            "Status": data_status,
            "Catatan": data_note,
        },
        {
            "Area": "Persistence",
            "Status": "Perhatian",
            "Catatan": "Streamlit Cloud tidak boleh dianggap sebagai database permanen; gunakan Export/Restore backup.",
        },
        {
            "Area": "Dependencies",
            "Status": "OK",
            "Catatan": "requirements.txt dipakai Streamlit saat build.",
        },
    ]
    return pd.DataFrame(rows)


def render_deploy_readiness_panel():
    with st.expander("Deploy Readiness", expanded=False):
        st.dataframe(get_deploy_readiness_rows(), width="stretch")
        st.caption("Untuk Streamlit Cloud, isi OPENROUTER_API_KEY dan OPENROUTER_MODEL di menu Secrets, bukan di file .env.")


def is_watchlisted(item_id):
    return any(item.get("id") == item_id for item in load_watchlist())


def add_watchlist_item(item):
    items = load_watchlist()
    item = item.copy()
    item["added_at"] = item.get("added_at") or utc_now_iso()
    items = [old for old in items if old.get("id") != item.get("id")]
    items.append(item)
    save_watchlist(items)


def remove_watchlist_item(item_id):
    save_watchlist([item for item in load_watchlist() if item.get("id") != item_id])


def watch_item_from_cex_row(row):
    symbol = str(row.get("symbol", "")).upper()
    quote = ""
    for candidate in BINANCE_QUOTES:
        if symbol.endswith(candidate):
            quote = candidate
            break
    base = symbol[:-len(quote)] if quote else str(row.get("base", ""))
    source = str(row.get("source", "Binance")).strip() or "Binance"
    return {
        "id": f"cex:{source.lower().replace(' ', '_')}:{symbol}",
        "type": "cex",
        "source": source,
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "label": symbol,
    }


def watch_item_from_stock_row(row):
    ticker = str(row.get("kode saham", row.get("name", ""))).upper().strip()
    symbol = str(row.get("symbol") or f"IDX:{ticker}").upper().strip()
    company = str(row.get("nama perusahaan", row.get("description", ticker))).strip()
    return {
        "id": f"stock:{symbol}",
        "type": "stock",
        "source": "TradingView IDX",
        "symbol": symbol,
        "ticker": ticker,
        "label": ticker,
        "company": company,
        "sector": row.get("sector", "N/A"),
    }


def watch_item_from_dex_row(row):
    chain = str(row.get("chain", "")).lower()
    pair_address = str(row.get("pair_address", ""))
    return {
        "id": f"dex:{chain}:{pair_address.lower()}",
        "type": "dex",
        "source": "DEX Screener",
        "chain": chain,
        "dex": row.get("dex", ""),
        "pair": row.get("pair", ""),
        "pair_address": pair_address,
        "base_address": row.get("base_address", ""),
        "base_symbol": row.get("base_symbol", ""),
        "quote_symbol": row.get("quote_symbol", ""),
        "url": row.get("url", ""),
        "label": f"{row.get('pair', '')} - {chain}",
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_24h():
    urls = [
        "https://api.binance.com/api/v3/ticker/24hr",
    ]
    last_error = None
    for url in urls:
        try:
            data = fetch_public_json(url, timeout=20)
            if isinstance(data, list):
                return data, url
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Gagal mengambil data Binance public API: {last_error}")


@st.cache_data(ttl=45, show_spinner=False)
def fetch_indodax_summaries():
    data = fetch_public_json("https://indodax.com/api/summaries", timeout=20)
    if isinstance(data, dict) and isinstance(data.get("tickers"), dict):
        return data, "https://indodax.com/api/summaries"
    raise RuntimeError("Indodax tidak mengembalikan data summaries.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indodax_pairs():
    data = fetch_public_json("https://indodax.com/api/pairs", timeout=20)
    if isinstance(data, list):
        return data, "https://indodax.com/api/pairs"
    raise RuntimeError("Indodax tidak mengembalikan data pairs.")


@st.cache_data(ttl=120, show_spinner=False)
def fetch_coingecko_markets(vs_currency="usd"):
    data = fetch_public_json(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": vs_currency,
            "order": "volume_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        },
        timeout=25,
    )
    if isinstance(data, list):
        return data, f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}"
    raise RuntimeError("CoinGecko tidak mengembalikan list market.")


def normalize_binance_tickers(rows, quote="USDT"):
    df = pd.DataFrame(rows)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].str.endswith(quote)].copy()
    if df.empty:
        return df
    df["base"] = df["symbol"].str[:-len(quote)]
    exclude = r"(UP|DOWN|BULL|BEAR)$"
    df = df[~df["base"].str.contains(exclude, regex=True, na=False)].copy()
    numeric_cols = [
        "lastPrice", "priceChangePercent", "volume", "quoteVolume",
        "highPrice", "lowPrice", "openPrice", "weightedAvgPrice", "count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.rename(columns={
        "lastPrice": "last_price",
        "priceChangePercent": "change_24h_pct",
        "quoteVolume": "quote_volume",
        "highPrice": "high_24h",
        "lowPrice": "low_24h",
        "openPrice": "open_24h",
        "weightedAvgPrice": "weighted_avg",
        "count": "trade_count",
    })
    df["liquidity_score"] = percentile_series(np.log1p(df["quote_volume"].fillna(0)))
    df["momentum_score"] = percentile_series(df["change_24h_pct"].fillna(0))
    df["activity_score"] = percentile_series(np.log1p(df["trade_count"].fillna(0)))
    df["crypto_score"] = (
        df["liquidity_score"] * 0.40 +
        df["momentum_score"] * 0.35 +
        df["activity_score"] * 0.25
    )

    def label(row):
        change = safe_num(row.get("change_24h_pct"), 0)
        score = safe_num(row.get("crypto_score"), 0)
        qv = safe_num(row.get("quote_volume"), 0)
        if change >= 12 and score >= 70:
            return "Momentum kuat"
        if change >= 4 and qv >= 5_000_000:
            return "Trending"
        if change <= -8:
            return "Tekanan jual"
        if score >= 70:
            return "Likuid aktif"
        return "Watchlist"

    df["signal"] = df.apply(label, axis=1)
    df["source"] = "Binance"
    df["source_url"] = "https://api.binance.com"
    return df.sort_values("crypto_score", ascending=False).reset_index(drop=True)


def normalize_indodax_markets(summary, pairs):
    tickers = summary.get("tickers", {}) if isinstance(summary, dict) else {}
    prices_24h = summary.get("prices_24h", {}) if isinstance(summary, dict) else {}
    prices_7d = summary.get("prices_7d", {}) if isinstance(summary, dict) else {}
    pair_meta = {
        str(item.get("ticker_id", "")).lower(): item
        for item in pairs
        if isinstance(item, dict) and str(item.get("base_currency", "")).lower() == "idr"
    }
    rows = []
    for ticker_id, ticker in tickers.items():
        if not isinstance(ticker, dict):
            continue
        ticker_id = str(ticker_id).lower()
        if not ticker_id.endswith("_idr"):
            continue
        meta = pair_meta.get(ticker_id, {})
        pair_key = ticker_id.replace("_", "")
        traded_currency = str(meta.get("traded_currency") or ticker_id.replace("_idr", "")).lower()
        base = str(meta.get("traded_currency_unit") or traded_currency).upper()
        last = safe_num(ticker.get("last"), np.nan)
        open_24h = safe_num(prices_24h.get(pair_key), np.nan)
        open_7d = safe_num(prices_7d.get(pair_key), np.nan)
        change_24h = (last - open_24h) / open_24h * 100 if not pd.isna(last) and not pd.isna(open_24h) and open_24h else np.nan
        change_7d = (last - open_7d) / open_7d * 100 if not pd.isna(last) and not pd.isna(open_7d) and open_7d else np.nan
        buy = safe_num(ticker.get("buy"), np.nan)
        sell = safe_num(ticker.get("sell"), np.nan)
        weighted = np.nan
        if not pd.isna(buy) and not pd.isna(sell) and buy > 0 and sell > 0:
            weighted = (buy + sell) / 2
        is_maintenance = bool(safe_num(meta.get("is_maintenance"), 0))
        is_suspended = bool(safe_num(meta.get("is_market_suspended"), 0))
        symbol = str(meta.get("symbol") or f"{base}IDR").upper()
        rows.append({
            "symbol": symbol,
            "tv_symbol": f"INDODAX:{symbol}",
            "ticker_id": ticker_id,
            "base": base,
            "quote": "IDR",
            "asset_name": ticker.get("name") or meta.get("description") or base,
            "last_price": last,
            "change_24h_pct": change_24h,
            "change_7d_pct": change_7d,
            "quote_volume": safe_num(ticker.get("vol_idr"), 0),
            "base_volume": safe_num(ticker.get(f"vol_{traded_currency}"), np.nan),
            "high_24h": safe_num(ticker.get("high"), np.nan),
            "low_24h": safe_num(ticker.get("low"), np.nan),
            "open_24h": open_24h,
            "open_7d": open_7d,
            "weighted_avg": weighted,
            "trade_count": np.nan,
            "buy_price": buy,
            "sell_price": sell,
            "exchange": "Indodax",
            "market_status": "Maintenance" if is_maintenance or is_suspended else "Aktif",
            "trade_min_idr": safe_num(meta.get("trade_min_base_currency"), np.nan),
            "coingecko_id": meta.get("coingecko_id", ""),
            "logo_url": meta.get("url_logo_png") or meta.get("url_logo", ""),
            "is_meme": base in MEME_COIN_BASES,
            "source": "Indodax",
            "source_url": "https://indodax.com/api/summaries",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["liquidity_score"] = percentile_series(np.log1p(df["quote_volume"].fillna(0)))
    df["momentum_score"] = percentile_series(df["change_24h_pct"].fillna(0))
    df["activity_score"] = percentile_series(np.log1p(df["quote_volume"].fillna(0))) * 0.7 + percentile_series(df["change_7d_pct"].fillna(0)) * 0.3
    df["crypto_score"] = (
        df["liquidity_score"] * 0.45 +
        df["momentum_score"] * 0.35 +
        df["activity_score"] * 0.20
    )
    df.loc[df["market_status"] != "Aktif", "crypto_score"] *= 0.70

    def label(row):
        change = safe_num(row.get("change_24h_pct"), 0)
        score = safe_num(row.get("crypto_score"), 0)
        qv = safe_num(row.get("quote_volume"), 0)
        if row.get("market_status") != "Aktif":
            return "Maintenance"
        if change >= 8 and score >= 70:
            return "Momentum IDR kuat"
        if change >= 3 and qv >= 100_000_000:
            return "Trending IDR"
        if change <= -6:
            return "Tekanan jual IDR"
        if score >= 70:
            return "Likuid IDR"
        return "Watchlist IDR"

    df["signal"] = df.apply(label, axis=1)
    return df.sort_values("crypto_score", ascending=False).reset_index(drop=True)


def normalize_coingecko_markets(rows, quote="USDT", vs_currency="usd"):
    df = pd.DataFrame(rows)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    q = quote.upper()
    df["base"] = df["symbol"].astype(str).str.upper()
    df["symbol"] = df["base"] + q
    df["quote"] = q
    df["asset_name"] = df["name"] if "name" in df.columns else df["base"]
    df["exchange"] = "CoinGecko"
    df["market_status"] = "Harga IDR global"
    df["is_meme"] = df["base"].isin(MEME_COIN_BASES)
    numeric_map = {
        "current_price": "last_price",
        "price_change_percentage_24h": "change_24h_pct",
        "total_volume": "quote_volume",
        "high_24h": "high_24h",
        "low_24h": "low_24h",
        "market_cap": "market_cap",
    }
    for src, dst in numeric_map.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    df["open_24h"] = np.nan
    df["change_7d_pct"] = np.nan
    df["weighted_avg"] = np.nan
    df["trade_count"] = np.nan
    df["liquidity_score"] = percentile_series(np.log1p(df["quote_volume"].fillna(0)))
    df["momentum_score"] = percentile_series(df["change_24h_pct"].fillna(0))
    df["activity_score"] = percentile_series(np.log1p(df["market_cap"].fillna(0)))
    df["crypto_score"] = (
        df["liquidity_score"] * 0.45 +
        df["momentum_score"] * 0.35 +
        df["activity_score"] * 0.20
    )

    def label(row):
        change = safe_num(row.get("change_24h_pct"), 0)
        score = safe_num(row.get("crypto_score"), 0)
        qv = safe_num(row.get("quote_volume"), 0)
        trending_volume = 1_000_000_000 if q == "IDR" else 1_000_000
        if change >= 12 and score >= 70:
            return "Momentum kuat"
        if change >= 4 and qv >= trending_volume:
            return "Trending"
        if change <= -8:
            return "Tekanan jual"
        if score >= 70:
            return "Likuid aktif"
        return "Watchlist"

    df["signal"] = df.apply(label, axis=1)
    df["source"] = "CoinGecko"
    df["source_url"] = f"https://api.coingecko.com ({vs_currency.upper()})"
    return df.sort_values("crypto_score", ascending=False).reset_index(drop=True)


def fetch_crypto_market_df(source="Auto", quote="USDT"):
    errors = []
    q = "IDR"
    if source in ["Auto", "Indodax IDR"]:
        try:
            summary, source_url = fetch_indodax_summaries()
            pairs, _ = fetch_indodax_pairs()
            df = normalize_indodax_markets(summary, pairs)
            if not df.empty:
                return df, source_url, "Indodax IDR"
            errors.append("Indodax: tidak ada pair IDR.")
        except Exception as exc:
            errors.append(f"Indodax: {exc}")
            if source == "Indodax IDR":
                raise RuntimeError(errors[-1])
    if source in ["CoinGecko IDR", "CoinGecko"]:
        try:
            vs_currency = COINGECKO_VS_MAP.get(q, "idr")
            rows, source_url = fetch_coingecko_markets(vs_currency=vs_currency)
            df = normalize_coingecko_markets(rows, quote=q, vs_currency=vs_currency)
            if not df.empty:
                df["exchange"] = "CoinGecko"
                df["market_status"] = "Harga IDR global"
                return df, source_url, "CoinGecko IDR"
            errors.append("CoinGecko: tidak ada data market.")
        except Exception as exc:
            errors.append(f"CoinGecko: {exc}")
            if source in ["CoinGecko IDR", "CoinGecko"]:
                raise RuntimeError(errors[-1])
    raise RuntimeError(" | ".join(errors) if errors else "Tidak ada source crypto IDR yang berhasil.")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_indodax_ohlcv(symbol, tf="1D", days=180):
    clean_symbol = str(symbol or "").upper().replace("INDODAX:", "").replace("_", "")
    if not clean_symbol:
        return pd.DataFrame()
    now = int(datetime.now(timezone.utc).timestamp())
    seconds_per_day = 86400
    params = {
        "from": now - int(days * seconds_per_day),
        "to": now,
        "tf": tf,
        "symbol": clean_symbol,
    }
    data = fetch_public_json("https://indodax.com/tradingview/history_v2", params=params, timeout=20)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    rename = {
        "Time": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="s", utc=True).dt.tz_convert("Asia/Jakarta")
    return df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_chart_technicals(df):
    if df.empty or len(df) < 10:
        return {}
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + bb_std * 2
    bb_lower = bb_mid - bb_std * 2
    last_close = safe_num(close.iloc[-1], np.nan)
    bb_pos = np.nan
    if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]) and bb_upper.iloc[-1] > bb_lower.iloc[-1]:
        bb_pos = (last_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
    lookback = min(20, len(df) - 1)
    trend_pct = (last_close / close.iloc[-lookback - 1] - 1) * 100 if lookback > 0 and close.iloc[-lookback - 1] else np.nan
    support = low.tail(min(20, len(df))).min()
    resistance = high.tail(min(20, len(df))).max()
    vol_avg = volume.tail(min(20, len(df))).mean()
    vol_rel = volume.iloc[-1] / vol_avg if vol_avg else np.nan
    recent = df.tail(min(20, len(df))).copy()
    higher_high = bool(len(recent) >= 6 and recent["high"].iloc[-1] > recent["high"].iloc[:-1].max())
    lower_low = bool(len(recent) >= 6 and recent["low"].iloc[-1] < recent["low"].iloc[:-1].min())
    breakout = bool(last_close >= resistance * 0.995) if not pd.isna(resistance) and resistance else False
    breakdown = bool(last_close <= support * 1.005) if not pd.isna(support) and support else False
    candle_range = high.iloc[-1] - low.iloc[-1]
    upper_wick = high.iloc[-1] - max(close.iloc[-1], df["open"].iloc[-1])
    lower_wick = min(close.iloc[-1], df["open"].iloc[-1]) - low.iloc[-1]
    rejection = "upper-wick" if candle_range and upper_wick / candle_range > 0.45 else "lower-wick" if candle_range and lower_wick / candle_range > 0.45 else "none"
    return {
        "last_close": last_close,
        "rsi": safe_num(calculate_rsi(close).iloc[-1], np.nan),
        "ema20": safe_num(ema20.iloc[-1], np.nan),
        "ema50": safe_num(ema50.iloc[-1], np.nan),
        "ema200": safe_num(ema200.iloc[-1], np.nan),
        "above_ema20": bool(last_close > ema20.iloc[-1]) if not pd.isna(ema20.iloc[-1]) else None,
        "above_ema50": bool(last_close > ema50.iloc[-1]) if not pd.isna(ema50.iloc[-1]) else None,
        "above_ema200": bool(last_close > ema200.iloc[-1]) if not pd.isna(ema200.iloc[-1]) else None,
        "macd": safe_num(macd.iloc[-1], np.nan),
        "macd_signal": safe_num(macd_signal.iloc[-1], np.nan),
        "macd_cross": "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish",
        "atr14": safe_num(atr14.iloc[-1], np.nan),
        "atr_pct": safe_num(atr14.iloc[-1] / last_close * 100 if last_close else np.nan, np.nan),
        "bb_upper": safe_num(bb_upper.iloc[-1], np.nan),
        "bb_lower": safe_num(bb_lower.iloc[-1], np.nan),
        "bb_position": safe_num(bb_pos, np.nan),
        "support": safe_num(support, np.nan),
        "resistance": safe_num(resistance, np.nan),
        "volume_rel": safe_num(vol_rel, np.nan),
        "trend_pct": safe_num(trend_pct, np.nan),
        "higher_high": higher_high,
        "lower_low": lower_low,
        "breakout": breakout,
        "breakdown": breakdown,
        "rejection": rejection,
        "candles": len(df),
        "last_time": str(df["time"].iloc[-1]),
    }


def build_chart_context(row):
    symbol = row.get("symbol", "")
    daily = fetch_indodax_ohlcv(symbol, tf="1D", days=220)
    hourly = fetch_indodax_ohlcv(symbol, tf="60", days=14)
    return {
        "daily": calculate_chart_technicals(daily),
        "hourly": calculate_chart_technicals(hourly),
        "daily_df": daily,
        "hourly_df": hourly,
    }


def chart_summary_lines(chart_ctx, row):
    if not chart_ctx:
        return "CHART DATA: Tidak tersedia."
    lines = []
    for label, key in [("Daily", "daily"), ("Hourly", "hourly")]:
        tech = chart_ctx.get(key) or {}
        if not tech:
            lines.append(f"{label}: tidak tersedia.")
            continue
        fake_row = dict(row)
        lines.append(
            f"{label}: close={format_idr(tech.get('last_close'), compact=False)}, "
            f"RSI={safe_num(tech.get('rsi'), np.nan):.1f}, "
            f"MACD={safe_num(tech.get('macd'), np.nan):.4g}/{safe_num(tech.get('macd_signal'), np.nan):.4g} ({tech.get('macd_cross')}), "
            f"EMA20/50/200={format_idr(tech.get('ema20'), compact=False)}/{format_idr(tech.get('ema50'), compact=False)}/{format_idr(tech.get('ema200'), compact=False)}, "
            f"support={format_idr(tech.get('support'), compact=False)}, resistance={format_idr(tech.get('resistance'), compact=False)}, "
            f"BB pos={safe_num(tech.get('bb_position'), np.nan):.1f}%, vol rel={safe_num(tech.get('volume_rel'), np.nan):.2f}x, "
            f"ATR={pct_text(tech.get('atr_pct'))}, trend {tech.get('candles')} candles={pct_text(tech.get('trend_pct'))}, "
            f"breakout={tech.get('breakout')}, breakdown={tech.get('breakdown')}, rejection={tech.get('rejection')}."
        )
    return "\n".join(lines)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_indodax_orderbook(pair_key):
    clean_pair = str(pair_key or "").upper().replace("INDODAX:", "").replace("_", "")
    if not clean_pair:
        return {}
    try:
        data = fetch_public_json(f"https://indodax.com/api/depth/{clean_pair}", timeout=15)
        return data if isinstance(data, dict) and "buy" in data and "sell" in data else {}
    except Exception:
        return {}


def orderbook_from_row(row):
    pair_key = row.get("ticker_id") or row.get("symbol")
    raw = fetch_indodax_orderbook(pair_key)
    if not raw:
        return {}

    def parse_side(items):
        parsed = []
        for item in items or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            price = safe_num(item[0], np.nan)
            amount = safe_num(item[1], np.nan)
            if not pd.isna(price) and not pd.isna(amount) and price > 0 and amount > 0:
                parsed.append((price, amount))
        return parsed

    bids = parse_side(raw.get("buy"))
    asks = parse_side(raw.get("sell"))
    if not bids or not asks:
        return {}
    bids = sorted(bids, key=lambda x: x[0], reverse=True)
    asks = sorted(asks, key=lambda x: x[0])
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else np.nan
    spread_pct = (best_ask / best_bid - 1) * 100 if best_bid else np.nan

    def depth_within(side, pct, direction):
        if pd.isna(mid) or mid <= 0:
            return np.nan
        if direction == "ask":
            levels = [price * amount for price, amount in side if price <= mid * (1 + pct)]
        else:
            levels = [price * amount for price, amount in side if price >= mid * (1 - pct)]
        return sum(levels)

    def buy_slippage(idr_amount):
        remaining = safe_num(idr_amount, 0)
        acquired, spent = 0.0, 0.0
        for price, amount in asks:
            level_value = price * amount
            take_value = min(remaining, level_value)
            acquired += take_value / price
            spent += take_value
            remaining -= take_value
            if remaining <= 0:
                break
        if acquired <= 0 or spent <= 0:
            return np.nan
        avg_price = spent / acquired
        return (avg_price / best_ask - 1) * 100

    def sell_slippage(idr_amount):
        target_base = safe_num(idr_amount, 0) / best_bid if best_bid else 0
        remaining_base = target_base
        proceeds, sold = 0.0, 0.0
        for price, amount in bids:
            take_base = min(remaining_base, amount)
            proceeds += take_base * price
            sold += take_base
            remaining_base -= take_base
            if remaining_base <= 0:
                break
        if sold <= 0 or proceeds <= 0:
            return np.nan
        avg_price = proceeds / sold
        return (1 - avg_price / best_bid) * 100

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_pct": spread_pct,
        "bid_depth_2pct": depth_within(bids, 0.02, "bid"),
        "ask_depth_2pct": depth_within(asks, 0.02, "ask"),
        "buy_slip_1m": buy_slippage(1_000_000),
        "buy_slip_10m": buy_slippage(10_000_000),
        "sell_slip_1m": sell_slippage(1_000_000),
        "sell_slip_10m": sell_slippage(10_000_000),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
    }


def orderbook_summary_lines(orderbook):
    if not orderbook:
        return "ORDERBOOK: Tidak tersedia."
    return (
        f"Best bid/ask: {format_idr(orderbook.get('best_bid'), compact=False)} / {format_idr(orderbook.get('best_ask'), compact=False)}\n"
        f"Spread: {pct_text(orderbook.get('spread_pct'))}\n"
        f"Depth -2%/+2%: bid {format_idr(orderbook.get('bid_depth_2pct'))} / ask {format_idr(orderbook.get('ask_depth_2pct'))}\n"
        f"Est. buy slippage Rp1M/Rp10M: {pct_text(orderbook.get('buy_slip_1m'))} / {pct_text(orderbook.get('buy_slip_10m'))}\n"
        f"Est. sell slippage Rp1M/Rp10M: {pct_text(orderbook.get('sell_slip_1m'))} / {pct_text(orderbook.get('sell_slip_10m'))}"
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_news(query, limit=8):
    clean_query = str(query or "").strip()
    if not clean_query:
        return []
    rss_url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(clean_query)}&hl=id&gl=ID&ceid=ID:id"
    )
    try:
        resp = requests.get(rss_url, headers={"User-Agent": "Market-Screener/1.0"}, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        rows = []
        for item in root.findall(".//item")[:limit]:
            source = item.find("source")
            rows.append({
                "title": (item.findtext("title") or "").strip(),
                "source": (source.text if source is not None else "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
            })
        return rows
    except Exception:
        return []


def score_news_rows(news_rows, row):
    if not news_rows:
        return []
    base = str(row.get("base") or "").lower()
    name_words = [w.lower() for w in str(row.get("asset_name") or "").replace("/", " ").split() if len(w) >= 3]
    scored = []
    now = datetime.now(timezone.utc)
    for item in news_rows:
        title = str(item.get("title", ""))
        title_l = title.lower()
        score = 0
        if base and base.lower() in title_l:
            score += 45
        if any(word in title_l for word in name_words):
            score += 30
        if "indodax" in title_l:
            score += 15
        if any(word in title_l for word in ["listing", "etf", "hack", "sec", "regulasi", "pump", "dump", "airdrop"]):
            score += 10
        age_days = np.nan
        try:
            parsed = parsedate_to_datetime(item.get("published", ""))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max((now - parsed.astimezone(timezone.utc)).days, 0)
            if age_days <= 7:
                score += 15
            elif age_days <= 30:
                score += 8
            elif age_days > 120:
                score -= 20
        except Exception:
            pass
        clean = item.copy()
        clean["relevance"] = int(np.clip(score, 0, 100))
        clean["age_days"] = age_days
        scored.append(clean)
    return sorted(scored, key=lambda x: (safe_num(x.get("relevance"), 0), -safe_num(x.get("age_days"), 9999)), reverse=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coingecko_coin_detail(coin_id):
    clean_id = str(coin_id or "").strip()
    if not clean_id:
        return {}
    try:
        data = fetch_public_json(
            f"https://api.coingecko.com/api/v3/coins/{clean_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "true",
                "developer_data": "false",
                "sparkline": "false",
            },
            timeout=20,
        )
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def summarize_community(row):
    detail = fetch_coingecko_coin_detail(row.get("coingecko_id"))
    community = detail.get("community_data") or {}
    links = detail.get("links") or {}
    twitter = safe_num(community.get("twitter_followers"), np.nan)
    reddit = safe_num(community.get("reddit_subscribers"), np.nan)
    telegram = safe_num(community.get("telegram_channel_user_count"), np.nan)
    score_base = 0
    for value, weight in [(twitter, 45), (reddit, 35), (telegram, 20)]:
        if not pd.isna(value):
            score_base += min(np.log10(max(value, 1)) / 6, 1) * weight
    label = "Tidak tersedia"
    if score_base >= 75:
        label = "Sangat besar"
    elif score_base >= 55:
        label = "Besar"
    elif score_base >= 35:
        label = "Menengah"
    elif score_base > 0:
        label = "Kecil / niche"
    twitter_name = links.get("twitter_screen_name") or ""
    telegram_name = links.get("telegram_channel_identifier") or ""
    subreddit = links.get("subreddit_url") or ""
    homepage = ""
    homepages = links.get("homepage") or []
    if isinstance(homepages, list):
        homepage = next((url for url in homepages if url), "")
    return {
        "label": label,
        "score": float(np.clip(score_base, 0, 100)),
        "twitter_followers": twitter,
        "reddit_subscribers": reddit,
        "telegram_users": telegram,
        "twitter": f"https://x.com/{twitter_name}" if twitter_name else "",
        "telegram": f"https://t.me/{telegram_name}" if telegram_name else "",
        "subreddit": subreddit,
        "homepage": homepage,
        "coingecko_id": row.get("coingecko_id", ""),
    }


def news_query_for_row(row):
    name = str(row.get("asset_name") or row.get("base") or "").strip()
    base = str(row.get("base") or "").strip()
    terms = [term for term in [name, base, "crypto", "Indodax"] if term]
    return " ".join(terms)


def news_summary_lines(news_rows):
    if not news_rows:
        return "NEWS: Tidak ada berita terbaru yang berhasil diambil."
    lines = []
    for idx, item in enumerate(news_rows[:6], 1):
        lines.append(f"{idx}. [{item.get('relevance', 'N/A')}/100] {item.get('title')} | {item.get('source')} | {item.get('published')}")
    return "\n".join(lines)


def community_summary_lines(community):
    if not community:
        return "COMMUNITY: Tidak tersedia."
    return (
        f"Community label: {community.get('label')} ({safe_num(community.get('score'), 0):.1f}/100)\n"
        f"Twitter/X followers: {format_compact(community.get('twitter_followers'))}\n"
        f"Reddit subscribers: {format_compact(community.get('reddit_subscribers'))}\n"
        f"Telegram users: {format_compact(community.get('telegram_users'))}\n"
        f"Links: X={community.get('twitter') or 'N/A'} | Telegram={community.get('telegram') or 'N/A'} | Reddit={community.get('subreddit') or 'N/A'}"
    )


def chart_technical_table(chart_ctx, row):
    rows = []
    for label, key in [("Daily", "daily"), ("Hourly", "hourly")]:
        tech = chart_ctx.get(key) if chart_ctx else {}
        if not tech:
            continue
        rows.append({
            "Timeframe": label,
            "Close": format_idr(tech.get("last_close"), compact=False),
            "RSI": f"{safe_num(tech.get('rsi'), np.nan):.1f}",
            "MACD": f"{safe_num(tech.get('macd'), np.nan):.4g}",
            "MACD Signal": f"{safe_num(tech.get('macd_signal'), np.nan):.4g}",
            "Cross": tech.get("macd_cross"),
            "EMA20": format_idr(tech.get("ema20"), compact=False),
            "EMA50": format_idr(tech.get("ema50"), compact=False),
            "EMA200": format_idr(tech.get("ema200"), compact=False),
            "Support": format_idr(tech.get("support"), compact=False),
            "Resistance": format_idr(tech.get("resistance"), compact=False),
            "BB Pos": f"{safe_num(tech.get('bb_position'), np.nan):.1f}%",
            "Vol Rel": f"{safe_num(tech.get('volume_rel'), np.nan):.2f}x",
            "ATR": pct_text(tech.get("atr_pct")),
            "Breakout": str(tech.get("breakout")),
            "Breakdown": str(tech.get("breakdown")),
            "Rejection": tech.get("rejection"),
            "Trend": pct_text(tech.get("trend_pct")),
        })
    return pd.DataFrame(rows)


def build_candlestick_chart(chart_df):
    if chart_df is None or chart_df.empty:
        return None
    cols = ["time", "open", "high", "low", "close", "volume"]
    if any(col not in chart_df.columns for col in cols):
        return None
    df = chart_df.tail(120)[cols].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "open", "high", "low", "close"])
    if len(df) < 2:
        return None
    width, height = 1120, 380
    margin_left, margin_right, margin_top, margin_bottom = 82, 22, 22, 44
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bg = (15, 23, 42)
    panel = (17, 24, 39)
    grid = (39, 53, 73)
    text = (148, 163, 184)
    up = (34, 197, 94)
    down = (239, 68, 68)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=10, fill=panel, outline=grid)

    min_price = float(df["low"].min())
    max_price = float(df["high"].max())
    price_range = max_price - min_price
    pad = price_range * 0.06 if price_range > 0 else max(abs(max_price) * 0.02, 1.0)
    min_price -= pad
    max_price += pad
    price_range = max(max_price - min_price, 1e-9)

    def y_pos(price):
        return margin_top + (max_price - float(price)) / price_range * plot_h

    for tick in np.linspace(min_price, max_price, 5):
        y = y_pos(tick)
        draw.line([margin_left, y, width - margin_right, y], fill=grid, width=1)
        draw.text((8, y - 6), format_idr(tick), fill=text, font=font)

    n = len(df)
    step = plot_w / n
    candle_w = max(3, min(11, int(step * 0.58)))
    for i, row in enumerate(df.itertuples(index=False)):
        x = margin_left + (i + 0.5) * step
        color = up if row.close >= row.open else down
        y_high = y_pos(row.high)
        y_low = y_pos(row.low)
        y_open = y_pos(row.open)
        y_close = y_pos(row.close)
        draw.line([x, y_high, x, y_low], fill=color, width=1)
        top = min(y_open, y_close)
        bottom = max(y_open, y_close)
        if bottom - top < 2:
            bottom = top + 2
        draw.rectangle([x - candle_w / 2, top, x + candle_w / 2, bottom], fill=color)

    label_idx = np.linspace(0, n - 1, min(5, n)).astype(int)
    for idx in label_idx:
        row = df.iloc[int(idx)]
        x = margin_left + (idx + 0.5) * step
        label = row["time"].strftime("%m-%d %H:%M")
        draw.text((x - 28, height - margin_bottom + 14), label, fill=text, font=font)
    return img


def tradingview_advanced_chart_url(symbol: str, interval: str = "60") -> str:
    params = {
        "frameElementId": f"tradingview_{str(symbol).replace(':', '_')}",
        "symbol": symbol,
        "interval": interval,
        "hidesidetoolbar": "0",
        "symboledit": "1",
        "saveimage": "1",
        "toolbarbg": "f1f3f6",
        "studies": "[]",
        "theme": "dark",
        "style": "1",
        "timezone": "Asia/Jakarta",
        "withdateranges": "1",
        "hideideas": "1",
        "locale": "id",
    }
    return "https://s.tradingview.com/widgetembed/?" + urlencode(params)


def render_tradingview_advanced_chart(symbol: str, interval: str = "60", height: int = 580):
    st.markdown("**TradingView Chart**")
    st.iframe(tradingview_advanced_chart_url(symbol, interval=interval), height=height, width="stretch")


def render_chart_analysis_panel(row, chart_ctx):
    table = chart_technical_table(chart_ctx, row)
    if table.empty:
        st.warning("Data OHLCV Indodax untuk analisis chart belum tersedia.")
        return
    st.markdown("**Teknikal Chart yang Dibaca AI**")
    st.dataframe(table, width="stretch")
    hourly = chart_ctx.get("hourly_df", pd.DataFrame())
    daily = chart_ctx.get("daily_df", pd.DataFrame())
    chart_df = hourly.tail(120) if isinstance(hourly, pd.DataFrame) and not hourly.empty else daily.tail(120)
    if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
        chart = build_candlestick_chart(chart_df)
        if chart is not None:
            st.image(chart, width="stretch")
        else:
            st.info("Chart internal belum cukup data valid untuk dirender.")
    st.caption("AI membaca data OHLCV/indikator dari endpoint chart Indodax. Iframe TradingView tetap untuk inspeksi visual manual.")


def render_news_community_panel(row, news_rows, community):
    st.markdown("**News Terbaru**")
    if news_rows:
        news_df = pd.DataFrame(news_rows)
        show_cols = [c for c in ["relevance", "age_days", "title", "source", "published"] if c in news_df.columns]
        st.dataframe(news_df[show_cols], width="stretch")
        for item in news_rows[:5]:
            if item.get("link"):
                st.link_button(item.get("title")[:80], item.get("link"), width="stretch")
    else:
        st.info("Belum ada berita yang berhasil diambil untuk coin ini.")
    st.markdown("**Community Check**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label", community.get("label", "N/A"))
    c2.metric("X/Twitter", format_compact(community.get("twitter_followers")))
    c3.metric("Reddit", format_compact(community.get("reddit_subscribers")))
    c4.metric("Telegram", format_compact(community.get("telegram_users")))
    for label, url in [
        ("Website", community.get("homepage")),
        ("X/Twitter", community.get("twitter")),
        ("Telegram", community.get("telegram")),
        ("Reddit", community.get("subreddit")),
    ]:
        if url:
            st.link_button(label, url, width="stretch")
    if not community.get("coingecko_id"):
        st.caption("Community data butuh coingecko_id dari metadata Indodax. Jika kosong, AI hanya memakai berita dan market data.")


def render_orderbook_panel(orderbook):
    st.markdown("**Orderbook & Slippage**")
    if not orderbook:
        st.info("Orderbook Indodax belum tersedia untuk pair ini.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Bid", format_idr(orderbook.get("best_bid"), compact=False))
    c2.metric("Best Ask", format_idr(orderbook.get("best_ask"), compact=False))
    c3.metric("Spread", pct_text(orderbook.get("spread_pct")))
    c4.metric("Levels", f"{safe_num(orderbook.get('bid_levels'), 0):.0f}/{safe_num(orderbook.get('ask_levels'), 0):.0f}")
    depth_df = pd.DataFrame([
        {"Metric": "Bid depth -2%", "Value": format_idr(orderbook.get("bid_depth_2pct"))},
        {"Metric": "Ask depth +2%", "Value": format_idr(orderbook.get("ask_depth_2pct"))},
        {"Metric": "Buy slippage Rp1M", "Value": pct_text(orderbook.get("buy_slip_1m"))},
        {"Metric": "Buy slippage Rp10M", "Value": pct_text(orderbook.get("buy_slip_10m"))},
        {"Metric": "Sell slippage Rp1M", "Value": pct_text(orderbook.get("sell_slip_1m"))},
        {"Metric": "Sell slippage Rp10M", "Value": pct_text(orderbook.get("sell_slip_10m"))},
    ])
    st.dataframe(depth_df, width="stretch")
    st.caption("Slippage adalah estimasi dari orderbook public saat data diambil; kondisi bisa berubah cepat.")


def tv_scan_headers():
    return {
        "Content-Type": "application/json",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tv_stock_screener(limit=250, sort_by="Value.Traded", sort_order="desc"):
    payload = {
        "columns": STOCK_AUTO_FIELDS,
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "markets": ["indonesia"],
        "options": {"lang": "id"},
        "range": [0, int(limit)],
        "sort": {"sortBy": sort_by, "sortOrder": sort_order},
        "symbols": {"query": {"types": []}, "tickers": []},
    }
    resp = requests.post(
        "https://scanner.tradingview.com/indonesia/scan",
        json=payload,
        headers=tv_scan_headers(),
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()
    return body.get("data", []), int(body.get("totalCount", 0) or 0)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tv_symbol_snapshot(symbol):
    payload = {
        "columns": STOCK_AUTO_FIELDS,
        "symbols": {"tickers": [symbol], "query": {"types": []}},
    }
    resp = requests.post(
        "https://scanner.tradingview.com/indonesia/scan",
        json=payload,
        headers=tv_scan_headers(),
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return {}
    row = data[0].get("d", [])
    result = {field: row[i] if i < len(row) else None for i, field in enumerate(STOCK_AUTO_FIELDS)}
    result["symbol"] = data[0].get("s", symbol)
    return result


def rec_score(label):
    return {
        "Strong Buy": 100.0,
        "Buy": 78.0,
        "Neutral": 52.0,
        "Sell": 28.0,
        "Strong Sell": 8.0,
    }.get(str(label), 50.0)


def rsi_setup_score(value):
    rsi = safe_num(value, np.nan)
    if pd.isna(rsi):
        return 50.0
    if 45 <= rsi <= 65:
        return 90.0
    if 35 <= rsi < 45 or 65 < rsi <= 72:
        return 68.0
    if 30 <= rsi < 35 or 72 < rsi <= 80:
        return 48.0
    return 25.0


def bb_setup_score(value):
    pos = safe_num(value, np.nan)
    if pd.isna(pos):
        return 50.0
    if 45 <= pos <= 85:
        return 86.0
    if 25 <= pos < 45:
        return 64.0
    if 85 < pos <= 100:
        return 55.0
    return 35.0


def normalize_stock_auto_rows(rows):
    records = []
    for item in rows:
        raw = item.get("d", [])
        tv = {field: raw[i] if i < len(raw) else None for i, field in enumerate(STOCK_AUTO_FIELDS)}
        symbol = str(item.get("s") or tv.get("symbol") or "")
        ticker = str(tv.get("name") or symbol.split(":")[-1]).upper().strip()
        if not ticker or ticker == "NAN":
            continue
        parsed = parse_tv_data(tv)
        records.append({
            "kode saham": ticker,
            "symbol": symbol or f"IDX:{ticker}",
            "nama perusahaan": str(tv.get("description") or ticker).strip(),
            "sector": str(tv.get("sector") or "N/A").strip(),
            "close": safe_num(tv.get("close"), np.nan),
            "change_pct": safe_num(tv.get("change"), np.nan),
            "volume": safe_num(tv.get("volume"), np.nan),
            "value_traded": safe_num(tv.get("Value.Traded"), np.nan),
            "market_cap": safe_num(tv.get("market_cap_basic"), np.nan),
            **parsed,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    for col in ["change_pct", "perf_week", "perf_month", "volume", "value_traded", "market_cap", "rel_volume", "adx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rec_score"] = df["rec_all"].apply(rec_score)
    df["ema_score"] = df["ema_trend_count"].fillna(0).clip(0, 3) / 3 * 100
    df["adx_score"] = np.clip((df["adx"].fillna(15) - 12) * 4.5, 0, 100)
    df["macd_score"] = np.where(df["macd_cross"].eq("bullish"), 78, np.where(df["macd_cross"].eq("bearish"), 34, 50))
    df["trend_score"] = (
        df["rec_score"] * 0.35 +
        df["ema_score"] * 0.25 +
        df["adx_score"] * 0.20 +
        df["macd_score"] * 0.20
    )
    df["momentum_score"] = (
        percentile_series(df["change_pct"].fillna(0)) * 0.40 +
        percentile_series(df["perf_week"].fillna(0)) * 0.35 +
        percentile_series(df["perf_month"].fillna(0)) * 0.25
    )
    df["liquidity_score"] = (
        percentile_series(df["value_traded"].fillna(0)) * 0.70 +
        percentile_series(df["volume"].fillna(0)) * 0.30
    )
    df["volume_score"] = np.clip(df["rel_volume"].fillna(1.0) * 48, 0, 100)
    df["setup_score"] = df["rsi"].apply(rsi_setup_score) * 0.55 + df["bb_position"].apply(bb_setup_score) * 0.45
    df["auto_score"] = (
        df["trend_score"] * 0.30 +
        df["momentum_score"] * 0.25 +
        df["liquidity_score"] * 0.20 +
        df["volume_score"] * 0.15 +
        df["setup_score"] * 0.10
    )

    def signal(row):
        score = safe_num(row.get("auto_score"), 0)
        chg = safe_num(row.get("change_pct"), 0)
        rel = safe_num(row.get("rel_volume"), 1)
        rec = rec_score(row.get("rec_all"))
        week = safe_num(row.get("perf_week"), 0)
        rsi = safe_num(row.get("rsi"), 50)
        if score >= 78 and chg > 0 and rel >= 1.15 and rec >= 70:
            return "Breakout Candidate"
        if score >= 68 and week > 0 and rec >= 52:
            return "Trending Up"
        if score >= 62 and chg <= 0 and 38 <= rsi <= 58:
            return "Pullback Watch"
        if score >= 55:
            return "Watch Only"
        return "Weak / Avoid"

    df["signal_label"] = df.apply(signal, axis=1)
    return df.sort_values("auto_score", ascending=False).reset_index(drop=True)


def fetch_auto_market_regime():
    raw = fetch_tv_symbol_snapshot("IDX:COMPOSITE")
    if not raw:
        return {}
    parsed = parse_tv_data(raw)
    change = safe_num(parsed.get("change_pct"), 0)
    rec = rec_score(parsed.get("rec_all"))
    adx = safe_num(parsed.get("adx"), 15)
    perf_w = safe_num(parsed.get("perf_week"), 0)
    score = float(np.clip(50 + change * 7 + (rec - 50) * 0.35 + min(adx, 35) * 0.35 + perf_w * 1.2, 0, 100))
    label = "Risk-On" if score >= 62 else "Netral" if score >= 45 else "Risk-Off"
    return {
        "index_code": "COMPOSITE",
        "index_change_pct": change,
        "regime_score": score,
        "regime_label": label,
        "rec_all": parsed.get("rec_all", "N/A"),
        "rsi": parsed.get("rsi", np.nan),
        "adx": parsed.get("adx", np.nan),
        "perf_week": parsed.get("perf_week", np.nan),
        "perf_month": parsed.get("perf_month", np.nan),
    }


def build_stock_auto_prompt(row, market_regime=None):
    close = safe_num(row.get("close"), np.nan)
    change = safe_num(row.get("change_pct"), np.nan)
    value_traded = safe_num(row.get("value_traded"), np.nan)
    rel_volume = safe_num(row.get("rel_volume"), np.nan)
    score = safe_num(row.get("auto_score"), 0)
    atr_proxy = max(abs(safe_num(row.get("change_pct"), 1.8)), 1.2)
    stop = close * (1 - atr_proxy * 1.35 / 100) if not pd.isna(close) else np.nan
    tp1 = close * (1 + atr_proxy * 1.8 / 100) if not pd.isna(close) else np.nan
    tp2 = close * (1 + atr_proxy * 3.0 / 100) if not pd.isna(close) else np.nan
    reg_sec = "Market regime otomatis tidak tersedia."
    if market_regime:
        reg_sec = (
            f"Market regime: {market_regime.get('regime_label')} "
            f"(score={safe_num(market_regime.get('regime_score'), 0):.1f}, "
            f"COMPOSITE={pct_text(market_regime.get('index_change_pct'))}, "
            f"rec={market_regime.get('rec_all', 'N/A')})"
        )
    return f"""Kamu analis saham BEI untuk swing trading. Data sumber adalah TradingView scanner otomatis, bukan file upload BEI.

SAHAM: {row.get('kode saham')} - {row.get('nama perusahaan')}
SEKTOR: {row.get('sector')}
SUMBER DATA: TradingView Indonesia scanner.

DATA HARGA & LIKUIDITAS:
- Close: {format_idr(close, compact=False)}
- Change harian: {pct_text(change)}
- Volume: {format_compact(row.get('volume'))}
- Value traded: {format_idr(value_traded)}
- Relative volume 10D: {ratio_text(rel_volume)}
- Market cap: {format_idr(row.get('market_cap'))}

DATA TEKNIKAL:
- Rekomendasi TV: {row.get('rec_all')} | MA: {row.get('rec_ma')} | Oscillator: {row.get('rec_other')}
- RSI: {safe_num(row.get('rsi'), np.nan):.1f} | RSI prev: {safe_num(row.get('rsi_prev'), np.nan):.1f}
- MACD cross: {row.get('macd_cross')} | MACD={safe_num(row.get('macd'), np.nan):.4g} | Signal={safe_num(row.get('macd_signal'), np.nan):.4g}
- ADX: {safe_num(row.get('adx'), np.nan):.1f} | +DI={safe_num(row.get('adx_plus'), np.nan):.1f} | -DI={safe_num(row.get('adx_minus'), np.nan):.1f}
- EMA20={format_idr(row.get('ema20'), compact=False)} | EMA50={format_idr(row.get('ema50'), compact=False)} | EMA200={format_idr(row.get('ema200'), compact=False)}
- Bollinger position: {safe_num(row.get('bb_position'), np.nan):.1f}% | Lower={format_idr(row.get('bb_lower'), compact=False)} | Mid={format_idr(row.get('bb_basis'), compact=False)} | Upper={format_idr(row.get('bb_upper'), compact=False)}
- Perf 1W: {pct_text(row.get('perf_week'))} | Perf 1M: {pct_text(row.get('perf_month'))}

AUTO SCORE:
- Total auto score: {score:.1f}/100
- Trend: {safe_num(row.get('trend_score'), 0):.1f}
- Momentum: {safe_num(row.get('momentum_score'), 0):.1f}
- Liquidity: {safe_num(row.get('liquidity_score'), 0):.1f}
- Volume: {safe_num(row.get('volume_score'), 0):.1f}
- Setup: {safe_num(row.get('setup_score'), 0):.1f}
- Signal label: {row.get('signal_label')}

REGIME:
{reg_sec}

LEVEL AWAL BERBASIS VOLATILITY PROXY:
- Stop awal: {format_idr(stop, compact=False)}
- TP1: {format_idr(tp1, compact=False)}
- TP2: {format_idr(tp2, compact=False)}

BATASAN WAJIB:
- Jangan mengklaim foreign flow, broker summary, atau data orderbook karena tidak tersedia di mode otomatis.
- Kalau setup butuh konfirmasi, beri tag WAIT atau BUY ON CONFIRMATION.
- Semua level harga harus konsisten dengan angka input.
- Jelaskan trade-off jika indikator bertentangan.

FORMAT OUTPUT:
**Verdict:** [1 kalimat tegas]

**Decision Tag:** [AGRESIF BUY | BUY ON CONFIRMATION | BUY ON PULLBACK | WAIT | AVOID]

**Confidence:** [0-100] + alasan singkat

**Analisis Teknikal:**
[RSI, MACD, ADX, EMA, Bollinger, relative volume, performa 1W/1M]

**Analisis Likuiditas:**
[value traded, volume, risiko liquidity trap]

**Skenario 1-5 Hari:**
1. Continuation:
- Trigger:
- Entry:
- Stop:
- TP:
2. Pullback:
- Trigger:
- Entry:
- Stop:
- TP:
3. Invalidasi:
- Trigger:
- Aksi:

**Yang Tidak Terlihat di Auto Mode:**
[sebutkan foreign flow/broker tidak tersedia dan dampaknya]

**Rencana Final:**
[entry, stop, TP, kapan review ulang]
"""


def render_stock_auto_page(openrouter_key, llm_model):
    st.title("BEI Auto Scanner")
    st.caption("Saham Indonesia otomatis dari TradingView scanner. Tidak perlu upload file; upload BEI tetap tersedia sebagai mode advanced untuk foreign/broker flow.")
    with st.sidebar:
        st.header("Filter Auto Saham")
        sort_label = st.selectbox("Urutkan", list(STOCK_SORT_OPTIONS.keys()), index=0, key="stock_auto_sort")
        sort_order_label = st.selectbox("Arah", ["Desc", "Asc"], index=0, key="stock_auto_sort_order")
        fetch_limit = st.number_input("Ambil data", min_value=50, max_value=800, value=250, step=50, key="stock_auto_fetch_limit")
        min_value = st.number_input("Min value traded", min_value=0, value=5_000_000_000, step=1_000_000_000, key="stock_auto_min_value")
        min_score = st.slider("Min auto score", 0, 100, 55, key="stock_auto_min_score")
        top_n = st.number_input("Top N", min_value=5, max_value=200, value=30, step=5, key="stock_auto_top_n")
        search = st.text_input("Cari saham", placeholder="BBCA, BBRI, TLKM", key="stock_auto_search")

    sort_by = STOCK_SORT_OPTIONS.get(sort_label, "Value.Traded")
    sort_order = "asc" if sort_order_label == "Asc" else "desc"
    try:
        with st.spinner("Mengambil saham otomatis dari TradingView..."):
            rows, total_count = fetch_tv_stock_screener(limit=fetch_limit, sort_by=sort_by, sort_order=sort_order)
            df = normalize_stock_auto_rows(rows)
            market_regime = fetch_auto_market_regime()
    except Exception as exc:
        st.error(f"Auto scanner gagal mengambil data TradingView: {exc}")
        st.stop()

    if df.empty:
        st.warning("TradingView tidak mengembalikan data saham.")
        st.stop()

    with st.sidebar:
        sectors = sorted([s for s in df["sector"].dropna().astype(str).unique() if s and s != "N/A"])
        sector_f = st.selectbox("Sektor", ["Semua"] + sectors, key="stock_auto_sector")
        signal_f = st.selectbox("Sinyal", ["Semua", "Breakout Candidate", "Trending Up", "Pullback Watch", "Watch Only", "Weak / Avoid"], key="stock_auto_signal")

    view = df[df["value_traded"].fillna(0) >= float(min_value)].copy()
    view = view[view["auto_score"].fillna(0) >= float(min_score)]
    if sector_f != "Semua":
        view = view[view["sector"] == sector_f]
    if signal_f != "Semua":
        view = view[view["signal_label"] == signal_f]
    if search:
        q = search.upper().strip()
        view = view[
            view["kode saham"].astype(str).str.contains(q, na=False) |
            view["nama perusahaan"].astype(str).str.upper().str.contains(q, na=False)
        ]
    view = view.head(int(top_n))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Universe", f"{total_count} saham")
    r2.metric("Ditampilkan", len(view))
    r3.metric("Top Value", format_idr(df["value_traded"].max()))
    if market_regime:
        r4.metric("Regime", market_regime.get("regime_label", "N/A"), pct_text(market_regime.get("index_change_pct")))
    else:
        r4.metric("Regime", "N/A")
    st.caption("Data otomatis membaca harga, volume, indikator, dan rekomendasi TradingView. Net foreign dan broker summary tidak tersedia tanpa file BEI.")

    st.subheader("Kandidat Saham Auto")
    table_cols = [
        "kode saham", "nama perusahaan", "sector", "close", "change_pct", "value_traded",
        "rel_volume", "rec_all", "rsi", "adx", "perf_week", "perf_month", "auto_score", "signal_label",
    ]
    disp = view[[c for c in table_cols if c in view.columns]].copy()
    if not disp.empty:
        disp["close"] = disp["close"].apply(lambda x: format_idr(x, compact=False))
        disp["value_traded"] = disp["value_traded"].apply(format_idr)
        for col in ["change_pct", "perf_week", "perf_month"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(pct_text)
        if "rel_volume" in disp.columns:
            disp["rel_volume"] = disp["rel_volume"].apply(ratio_text)
        render_df_with_style_fallback(disp, ["auto_score"])
    else:
        st.info("Tidak ada saham sesuai filter.")
    st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), "bei_auto_tradingview.csv", "text/csv")

    if view.empty:
        st.warning("Tidak ada kandidat setelah filter.")
        st.stop()

    st.markdown("---")
    selected_code = st.selectbox("Pilih saham untuk analisis", view["kode saham"].tolist(), key="stock_auto_selected")
    sel = view[view["kode saham"] == selected_code].iloc[0]
    st.subheader(f"{sel['kode saham']}  {sel['nama perusahaan']}  {sel['signal_label']}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Auto Score", f"{safe_num(sel.get('auto_score'), 0):.1f}")
    m2.metric("Close", format_idr(sel.get("close"), compact=False))
    m3.metric("Change", pct_text(sel.get("change_pct")))
    m4.metric("Value", format_idr(sel.get("value_traded")))
    m5.metric("TV Rec", str(sel.get("rec_all", "N/A")))

    stock_item = watch_item_from_stock_row(sel)
    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        if is_watchlisted(stock_item["id"]):
            if st.button("Hapus Watchlist", width="stretch", key=f"stock_rm_{stock_item['id']}"):
                remove_watchlist_item(stock_item["id"])
                st.rerun()
        elif st.button("Tambah Watchlist", width="stretch", key=f"stock_add_{stock_item['id']}"):
            add_watchlist_item(stock_item)
            st.success("Saham masuk watchlist.")
    with a2:
        if st.button("Simpan Snapshot", width="stretch", key=f"stock_snap_{stock_item['id']}"):
            append_history_rows([history_row_from_stock(sel, stock_item["id"])])
            st.success("Snapshot saham tersimpan ke history.")
    with a3:
        st.caption("Watchlist saham tersimpan lokal. Refresh dari Watchlist & Alerts untuk update score dan alert.")

    section = st.radio(
        "Detail Saham Auto",
        ["Score", "Teknikal", "Chart", "News", "History", "AI Analisis"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"stock_auto_detail_{selected_code}",
    )
    if section == "Score":
        bd = pd.DataFrame({
            "Faktor": ["Trend", "Momentum", "Likuiditas", "Volume Relatif", "Setup"],
            "Skor": [
                safe_num(sel.get("trend_score"), np.nan),
                safe_num(sel.get("momentum_score"), np.nan),
                safe_num(sel.get("liquidity_score"), np.nan),
                safe_num(sel.get("volume_score"), np.nan),
                safe_num(sel.get("setup_score"), np.nan),
            ],
        })
        bd["Status"] = bd["Skor"].apply(factor_label)
        render_df_with_style_fallback(bd, ["Skor"])
        st.info("Mode auto tidak membaca net foreign dan broker summary. Untuk flow detail, gunakan mode Upload BEI Advanced.")
    elif section == "Teknikal":
        ta1, ta2, ta3 = st.columns(3)
        ta1.metric("Rekomendasi", str(sel.get("rec_all", "N/A")))
        ta2.metric("MA Signal", str(sel.get("rec_ma", "N/A")))
        ta3.metric("Oscillator", str(sel.get("rec_other", "N/A")))
        tb1, tb2, tb3, tb4 = st.columns(4)
        tb1.metric("RSI(14)", f"{safe_num(sel.get('rsi'), 0):.1f}")
        tb2.metric("ADX", f"{safe_num(sel.get('adx'), 0):.1f}")
        tb3.metric("Stoch K/D", f"{safe_num(sel.get('stoch_k'), 0):.1f}/{safe_num(sel.get('stoch_d'), 0):.1f}")
        tb4.metric("Vol Rel", ratio_text(sel.get("rel_volume")))
        tc1, tc2, tc3 = st.columns(3)
        ef = lambda x: "di atas" if x else "di bawah" if x is False else "-"
        tc1.metric("EMA20", format_idr(sel.get("ema20"), compact=False), delta=ef(sel.get("above_ema20")))
        tc2.metric("EMA50", format_idr(sel.get("ema50"), compact=False), delta=ef(sel.get("above_ema50")))
        tc3.metric("EMA200", format_idr(sel.get("ema200"), compact=False), delta=ef(sel.get("above_ema200")))
        if sel.get("bb_position") is not None and not pd.isna(sel.get("bb_position")):
            st.progress(min(max(int(safe_num(sel.get("bb_position"), 0)), 0), 100), text=f"Bollinger Position: {safe_num(sel.get('bb_position'), 0):.1f}%")
    elif section == "Chart":
        render_tradingview_advanced_chart(f"IDX:{selected_code}", interval="D", height=580)
    elif section == "News":
        st.iframe(f"https://id.tradingview.com/symbols/IDX-{selected_code}/news/", height=720, width="stretch")
    elif section == "History":
        history = pd.DataFrame(load_history())
        item_history = history[history["id"] == stock_item["id"]].copy() if not history.empty and "id" in history.columns else pd.DataFrame()
        if item_history.empty:
            st.info("Belum ada history untuk saham ini. Klik Simpan Snapshot atau refresh dari Watchlist & Alerts.")
        else:
            item_history["ts"] = pd.to_datetime(item_history["ts"], errors="coerce")
            metric_cols = [c for c in ["score", "change", "rel_volume"] if c in item_history.columns]
            if metric_cols:
                st.line_chart(item_history.sort_values("ts").set_index("ts")[metric_cols])
            st.dataframe(item_history.sort_values("ts", ascending=False), width="stretch")
    elif section == "AI Analisis":
        if not openrouter_key:
            st.warning("Masukkan OpenRouter API Key di sidebar untuk AI.")
        elif st.button("Generate Analisis AI", width="stretch", key=f"stock_auto_ai_{selected_code}"):
            with st.spinner("OpenRouter menganalisis saham otomatis..."):
                st.markdown(call_openrouter(
                    build_stock_auto_prompt(sel, market_regime=market_regime),
                    openrouter_key,
                    llm_model,
                    max_tokens=2200,
                ))
            st.caption("Output AI memakai data TradingView otomatis. Bukan rekomendasi investasi.")


@st.cache_data(ttl=45, show_spinner=False)
def fetch_dex_search(query):
    data = fetch_public_json(
        "https://api.dexscreener.com/latest/dex/search",
        params={"q": query},
        timeout=20,
    )
    return data.get("pairs", []) if isinstance(data, dict) else []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_dex_pairs(chain_id, pair_id):
    data = fetch_public_json(
        f"https://api.dexscreener.com/latest/dex/pairs/{chain_id}/{pair_id}",
        timeout=20,
    )
    return data.get("pairs", []) if isinstance(data, dict) else []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_dex_latest_profiles():
    data = fetch_public_json("https://api.dexscreener.com/token-profiles/latest/v1", timeout=20)
    return data if isinstance(data, list) else []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_dex_latest_boosted():
    data = fetch_public_json("https://api.dexscreener.com/token-boosts/latest/v1", timeout=20)
    return data if isinstance(data, list) else []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_dex_top_boosted():
    data = fetch_public_json("https://api.dexscreener.com/token-boosts/top/v1", timeout=20)
    return data if isinstance(data, list) else []


@st.cache_data(ttl=90, show_spinner=False)
def fetch_dex_tokens(chain_id, token_addresses):
    cleaned = [str(addr).strip() for addr in token_addresses if str(addr).strip()]
    if not cleaned:
        return []
    data = fetch_public_json(
        f"https://api.dexscreener.com/tokens/v1/{chain_id}/{','.join(cleaned)}",
        timeout=25,
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("pairs", [])
    return []


def fetch_pairs_from_token_profiles(profiles, limit=60):
    grouped = {}
    for item in profiles[:limit]:
        if not isinstance(item, dict):
            continue
        chain = str(item.get("chainId", "")).lower().strip()
        token = str(item.get("tokenAddress", "")).strip()
        if chain and token:
            grouped.setdefault(chain, []).append(token)
    pairs = []
    for chain, addresses in grouped.items():
        seen = []
        for address in addresses:
            if address not in seen:
                seen.append(address)
        for start in range(0, len(seen), 30):
            try:
                pairs.extend(fetch_dex_tokens(chain, seen[start:start + 30]))
            except Exception:
                continue
    return pairs


def fetch_meme_source_pairs(source, query, watchlist_limit=120):
    if source == "Search":
        return fetch_dex_search(query)
    if source == "Latest Profiles":
        return fetch_pairs_from_token_profiles(fetch_dex_latest_profiles(), limit=watchlist_limit)
    if source == "Latest Boosted":
        return fetch_pairs_from_token_profiles(fetch_dex_latest_boosted(), limit=watchlist_limit)
    if source == "Top Boosted":
        return fetch_pairs_from_token_profiles(fetch_dex_top_boosted(), limit=watchlist_limit)
    pairs = []
    for item in load_watchlist():
        if item.get("type") != "dex":
            continue
        try:
            pairs.extend(fetch_dex_pairs(item.get("chain", ""), item.get("pair_address", "")))
        except Exception:
            continue
    return pairs


def nested_num(obj, key, subkey=None, default=np.nan):
    try:
        value = obj.get(key, {}) if isinstance(obj, dict) else {}
        if subkey is not None:
            value = value.get(subkey, {}) if isinstance(value, dict) else default
        return safe_num(value, default)
    except Exception:
        return default


def normalize_dex_pairs(pairs):
    rows = []
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    for item in pairs:
        if not isinstance(item, dict):
            continue
        base = item.get("baseToken") or {}
        quote = item.get("quoteToken") or {}
        txns = item.get("txns") or {}
        volume = item.get("volume") or {}
        price_change = item.get("priceChange") or {}
        liquidity = item.get("liquidity") or {}
        info = item.get("info") or {}
        boosts = item.get("boosts") or {}
        created_at = safe_num(item.get("pairCreatedAt"), np.nan)
        age_minutes = (now_ms - created_at) / 60000 if not pd.isna(created_at) else np.nan
        buys_m5 = nested_num(txns, "m5", "buys", 0)
        sells_m5 = nested_num(txns, "m5", "sells", 0)
        buys_h1 = nested_num(txns, "h1", "buys", 0)
        sells_h1 = nested_num(txns, "h1", "sells", 0)
        buys_h6 = nested_num(txns, "h6", "buys", 0)
        sells_h6 = nested_num(txns, "h6", "sells", 0)
        buys_h24 = nested_num(txns, "h24", "buys", 0)
        sells_h24 = nested_num(txns, "h24", "sells", 0)
        total_m5 = buys_m5 + sells_m5
        total_h1 = buys_h1 + sells_h1
        total_h6 = buys_h6 + sells_h6
        total_h24 = buys_h24 + sells_h24
        buy_ratio_m5 = buys_m5 / total_m5 if total_m5 > 0 else np.nan
        buy_ratio_h1 = buys_h1 / total_h1 if total_h1 > 0 else np.nan
        buy_ratio_h6 = buys_h6 / total_h6 if total_h6 > 0 else np.nan
        buy_ratio_h24 = buys_h24 / total_h24 if total_h24 > 0 else np.nan
        rows.append({
            "chain": str(item.get("chainId", "")).lower(),
            "dex": item.get("dexId", ""),
            "pair_address": item.get("pairAddress", ""),
            "url": item.get("url", ""),
            "base_symbol": base.get("symbol", ""),
            "base_name": base.get("name", ""),
            "base_address": base.get("address", ""),
            "quote_name": quote.get("name", ""),
            "quote_symbol": quote.get("symbol", ""),
            "quote_address": quote.get("address", ""),
            "price_usd": safe_num(item.get("priceUsd"), np.nan),
            "price_native": safe_num(item.get("priceNative"), np.nan),
            "liquidity_usd": safe_num(liquidity.get("usd"), np.nan),
            "liquidity_base": safe_num(liquidity.get("base"), np.nan),
            "liquidity_quote": safe_num(liquidity.get("quote"), np.nan),
            "fdv": safe_num(item.get("fdv"), np.nan),
            "market_cap": safe_num(item.get("marketCap"), np.nan),
            "volume_m5": safe_num(volume.get("m5"), 0),
            "volume_h1": safe_num(volume.get("h1"), 0),
            "volume_h6": safe_num(volume.get("h6"), 0),
            "volume_h24": safe_num(volume.get("h24"), 0),
            "change_m5_pct": safe_num(price_change.get("m5"), np.nan),
            "change_h1_pct": safe_num(price_change.get("h1"), np.nan),
            "change_h6_pct": safe_num(price_change.get("h6"), np.nan),
            "change_h24_pct": safe_num(price_change.get("h24"), np.nan),
            "buys_m5": buys_m5,
            "sells_m5": sells_m5,
            "buys_h1": buys_h1,
            "sells_h1": sells_h1,
            "buys_h6": buys_h6,
            "sells_h6": sells_h6,
            "buys_h24": buys_h24,
            "sells_h24": sells_h24,
            "txns_m5": total_m5,
            "txns_h1": total_h1,
            "txns_h6": total_h6,
            "txns_h24": total_h24,
            "buy_ratio_m5": buy_ratio_m5,
            "buy_ratio_h1": buy_ratio_h1,
            "buy_ratio_h6": buy_ratio_h6,
            "buy_ratio_h24": buy_ratio_h24,
            "age_minutes": age_minutes,
            "pair_created_at": created_at,
            "labels": ", ".join(item.get("labels") or []),
            "boosts_active": safe_num(boosts.get("active"), 0),
            "info_json": json.dumps(info, ensure_ascii=False),
            "info_image_url": info.get("imageUrl", "") if isinstance(info, dict) else "",
            "info_header_url": info.get("header", "") if isinstance(info, dict) else "",
            "info_open_graph": info.get("openGraph", "") if isinstance(info, dict) else "",
            "link_summary": link_summary(info),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pair"] = (
        df["base_symbol"].fillna("").astype(str) + "/" +
        df["quote_symbol"].fillna("").astype(str)
    )
    df["liquidity_score"] = percentile_series(np.log1p(df["liquidity_usd"].fillna(0)))
    df["volume_score"] = percentile_series(np.log1p(df["volume_h1"].fillna(0) + df["volume_h24"].fillna(0) * 0.2))
    df["momentum_score"] = np.clip(
        50 +
        df["change_m5_pct"].fillna(0) * 2.5 +
        df["change_h1_pct"].fillna(0) * 1.3 +
        df["change_h24_pct"].fillna(0) * 0.2,
        0, 100,
    )
    df["buy_score"] = np.clip(df["buy_ratio_h1"].fillna(0.5) * 100, 0, 100)
    df["age_score"] = np.select(
        [
            df["age_minutes"].isna(),
            df["age_minutes"] < 10,
            df["age_minutes"] <= 1440,
            df["age_minutes"] <= 10080,
        ],
        [35, 45, 80, 60],
        default=40,
    )
    df["radar_score"] = (
        df["momentum_score"] * 0.30 +
        df["volume_score"] * 0.25 +
        df["liquidity_score"] * 0.20 +
        df["buy_score"] * 0.15 +
        df["age_score"] * 0.10
    )
    fdv_liq = np.where(df["liquidity_usd"].fillna(0) > 0, df["fdv"].fillna(0) / df["liquidity_usd"].fillna(1), np.nan)
    mcap_liq = np.where(df["liquidity_usd"].fillna(0) > 0, df["market_cap"].fillna(0) / df["liquidity_usd"].fillna(1), np.nan)
    df["fdv_liq_ratio"] = fdv_liq
    df["mcap_liq_ratio"] = mcap_liq
    df["volume_liq_m5"] = np.where(df["liquidity_usd"].fillna(0) > 0, df["volume_m5"].fillna(0) / df["liquidity_usd"].fillna(1), np.nan)
    df["volume_liq_h1"] = np.where(df["liquidity_usd"].fillna(0) > 0, df["volume_h1"].fillna(0) / df["liquidity_usd"].fillna(1), np.nan)
    df["volume_liq_h24"] = np.where(df["liquidity_usd"].fillna(0) > 0, df["volume_h24"].fillna(0) / df["liquidity_usd"].fillna(1), np.nan)
    df["risk_score"] = 0
    df.loc[df["liquidity_usd"].fillna(0) < 5_000, "risk_score"] += 35
    df.loc[df["liquidity_usd"].fillna(0).between(5_000, 25_000, inclusive="left"), "risk_score"] += 20
    df.loc[df["age_minutes"].fillna(0) < 30, "risk_score"] += 20
    df.loc[df["buy_ratio_h1"].fillna(0.5) < 0.35, "risk_score"] += 15
    df.loc[df["change_h1_pct"].fillna(0) < -25, "risk_score"] += 15
    df.loc[pd.Series(fdv_liq, index=df.index).fillna(0) > 150, "risk_score"] += 10
    df.loc[df["volume_liq_h1"].fillna(0) > 5, "risk_score"] += 10
    df["risk_score"] = df["risk_score"].clip(0, 100)

    def risk_label(score):
        s = safe_num(score, 0)
        if s >= 75:
            return "Ekstrem"
        if s >= 55:
            return "Tinggi"
        if s >= 35:
            return "Sedang"
        return "Rendah"

    df["risk_label"] = df["risk_score"].apply(risk_label)
    return df.sort_values("radar_score", ascending=False).reset_index(drop=True)


def get_pair_flags(row):
    flags = []
    liq = safe_num(row.get("liquidity_usd"), 0)
    age = safe_num(row.get("age_minutes"), np.nan)
    buy_ratio = safe_num(row.get("buy_ratio_h1"), np.nan)
    fdv_liq = safe_num(row.get("fdv_liq_ratio"), np.nan)
    vol_liq_h1 = safe_num(row.get("volume_liq_h1"), np.nan)
    if liq < 5_000:
        flags.append("Likuiditas sangat tipis")
    elif liq < 25_000:
        flags.append("Likuiditas masih tipis")
    if not pd.isna(age) and age < 30:
        flags.append("Pair sangat baru")
    if not pd.isna(buy_ratio) and buy_ratio < 0.35:
        flags.append("Sell pressure dominan")
    if safe_num(row.get("change_h1_pct"), 0) < -25:
        flags.append("Dump tajam 1 jam")
    if not pd.isna(fdv_liq) and fdv_liq > 150:
        flags.append("FDV terlalu besar dibanding likuiditas")
    if not pd.isna(vol_liq_h1) and vol_liq_h1 > 5:
        flags.append("Volume 1h sangat besar dibanding likuiditas")
    if not flags:
        flags.append("Belum ada red flag kuantitatif besar dari data DEX")
    flags.append("Belum termasuk audit kontrak/honeypot")
    return flags


def get_pair_risk_breakdown(row):
    items = []
    checks = [
        ("Likuiditas", safe_num(row.get("liquidity_usd"), 0), "USD"),
        ("Umur pair", safe_num(row.get("age_minutes"), np.nan), "minutes"),
        ("Buy ratio 1h", safe_num(row.get("buy_ratio_h1"), np.nan), "ratio"),
        ("Change 1h", safe_num(row.get("change_h1_pct"), np.nan), "pct"),
        ("FDV/Liquidity", safe_num(row.get("fdv_liq_ratio"), np.nan), "x"),
        ("Volume/Liquidity 1h", safe_num(row.get("volume_liq_h1"), np.nan), "x"),
    ]
    for label, value, kind in checks:
        status = "OK"
        note = ""
        if label == "Likuiditas":
            status = "Bahaya" if value < 5_000 else "Waspada" if value < 25_000 else "OK"
            note = format_compact(value, "$")
        elif label == "Umur pair":
            status = "Bahaya" if not pd.isna(value) and value < 30 else "Waspada" if not pd.isna(value) and value < 240 else "OK"
            note = format_age(value)
        elif label == "Buy ratio 1h":
            status = "Waspada" if not pd.isna(value) and value < 0.35 else "OK"
            note = f"{value * 100:.1f}%" if not pd.isna(value) else "N/A"
        elif label == "Change 1h":
            status = "Bahaya" if not pd.isna(value) and value < -25 else "Waspada" if not pd.isna(value) and abs(value) > 80 else "OK"
            note = pct_text(value)
        elif label == "FDV/Liquidity":
            status = "Waspada" if not pd.isna(value) and value > 150 else "OK"
            note = ratio_text(value)
        elif label == "Volume/Liquidity 1h":
            status = "Waspada" if not pd.isna(value) and value > 5 else "OK"
            note = ratio_text(value)
        items.append({"Faktor": label, "Nilai": note, "Status": status})
    return pd.DataFrame(items)


def range_position_pct(last, low, high):
    last = safe_num(last, np.nan)
    low = safe_num(low, np.nan)
    high = safe_num(high, np.nan)
    if pd.isna(last) or pd.isna(low) or pd.isna(high) or high <= low:
        return np.nan
    return float(np.clip((last - low) / (high - low) * 100, 0, 100))


def pct_distance(value, reference):
    value = safe_num(value, np.nan)
    reference = safe_num(reference, np.nan)
    if pd.isna(value) or pd.isna(reference) or reference == 0:
        return np.nan
    return (value / reference - 1) * 100


def crypto_volume_label(row):
    volume = safe_num(row.get("quote_volume"), 0)
    quote = str(row.get("quote", "IDR")).upper()
    usd_proxy = volume / 16_000 if quote == "IDR" else volume
    if usd_proxy >= 1_000_000_000:
        return "Sangat tebal"
    if usd_proxy >= 100_000_000:
        return "Tebal"
    if usd_proxy >= 10_000_000:
        return "Cukup aktif"
    if usd_proxy >= 1_000_000:
        return "Tipis-menengah"
    return "Tipis"


def build_crypto_forward_analysis(row):
    last = safe_num(row.get("last_price"), np.nan)
    open_24h = safe_num(row.get("open_24h"), np.nan)
    high = safe_num(row.get("high_24h"), np.nan)
    low = safe_num(row.get("low_24h"), np.nan)
    weighted = safe_num(row.get("weighted_avg"), np.nan)
    change = safe_num(row.get("change_24h_pct"), 0)
    score = safe_num(row.get("crypto_score"), 0)
    liquidity = safe_num(row.get("liquidity_score"), 50)
    momentum = safe_num(row.get("momentum_score"), 50)
    activity = safe_num(row.get("activity_score"), 50)
    pos = range_position_pct(last, low, high)
    range_pct = pct_distance(high, low)
    from_high = pct_distance(last, high)
    from_low = pct_distance(last, low)

    forward_score = score * 0.45 + momentum * 0.25 + liquidity * 0.20 + activity * 0.10
    if change < -2:
        forward_score -= min(abs(change) * 1.2, 25)
    elif change > 18:
        forward_score -= min((change - 18) * 0.6, 10)
    if not pd.isna(pos):
        if pos >= 65 and change > 0:
            forward_score += 5
        elif pos <= 25 and change < 0:
            forward_score -= 7
    forward_score = float(np.clip(forward_score, 0, 100))

    if change <= -8:
        verdict = "WAIT / DEFENSIVE"
        state = "Tekanan jual 24h masih dominan."
        short_bias = "Rebound baru lebih sehat kalau harga bisa reclaim area open/average dan volume beli muncul."
    elif change >= 12 and score >= 70:
        verdict = "MOMENTUM - JANGAN CHASE"
        state = "Momentum kuat, tetapi sudah ekspansif dalam 24 jam."
        short_bias = "Peluang lanjut ada, namun entry paling aman menunggu retest atau candle baru yang tidak ditolak."
    elif change >= 4 and score >= 60:
        verdict = "WATCH MOMENTUM"
        state = "Trend pendek positif dengan volume cukup mendukung."
        short_bias = "Bias masih naik selama harga bertahan di separuh atas range 24h."
    elif score >= 70:
        verdict = "WATCH BREAKOUT"
        state = "Likuiditas dan aktivitas bagus, arah harga belum sekuat skornya."
        short_bias = "Tunggu breakout high 24h atau pullback ringan yang cepat dibeli."
    elif score < 45:
        verdict = "LOW PRIORITY"
        state = "Data belum menunjukkan kombinasi momentum dan aktivitas yang kuat."
        short_bias = "Lebih baik tunggu volume dan arah harga lebih jelas."
    else:
        verdict = "NETRAL / WAIT"
        state = "Setup belum buruk, tapi konfirmasi lanjutan masih kurang."
        short_bias = "Gunakan area range 24h sebagai peta tunggu, bukan alasan mengejar candle."

    support_candidates = [weighted, open_24h, low]
    support = next((x for x in support_candidates if not pd.isna(x) and x > 0), np.nan)
    support_label = format_market_price({**dict(row), "support": support}, "support")
    high_label = format_market_price(row, "high_24h")
    low_label = format_market_price(row, "low_24h")

    metrics = pd.DataFrame([
        {"Metric": "Forward Score", "Value": f"{forward_score:.1f}/100", "Bacaan": "Kualitas setup ke depan dari momentum, volume, aktivitas, dan posisi range."},
        {"Metric": "Range 24h", "Value": pct_text(range_pct), "Bacaan": "Makin lebar, makin besar risiko whipsaw dan entry terlambat."},
        {"Metric": "Posisi di range 24h", "Value": f"{pos:.1f}%" if not pd.isna(pos) else "N/A", "Bacaan": "Di atas 60% berarti buyer masih menjaga area atas; di bawah 40% mulai lemah."},
        {"Metric": "Jarak dari high", "Value": pct_text(from_high), "Bacaan": "Dekat high cocok untuk breakout watch, tapi rawan rejection."},
        {"Metric": "Pantulan dari low", "Value": pct_text(from_low), "Bacaan": "Pantulan besar tanpa volume lanjutan sering berubah jadi pullback."},
        {"Metric": "Kualitas volume", "Value": crypto_volume_label(row), "Bacaan": f"Volume 24h {format_market_amount(row)}."},
    ])

    outlook = pd.DataFrame([
        {"Horizon": "1-4 jam", "Bias": short_bias, "Yang Dipantau": f"Reaksi harga terhadap support {support_label} dan high 24h {high_label}."},
        {"Horizon": "24 jam", "Bias": state, "Yang Dipantau": "Apakah change 24h bertahan positif sambil volume tetap masuk, bukan hanya spike pendek."},
        {"Horizon": "3-7 hari", "Bias": "Masuk watchlist kalau score tetap tinggi di beberapa snapshot.", "Yang Dipantau": "Simpan snapshot harian; cari score stabil, volume stabil, dan pullback yang tidak merusak struktur."},
    ])

    scenarios = pd.DataFrame([
        {
            "Skenario": "Bullish lanjut",
            "Trigger": f"Break/retest high 24h {high_label}, posisi range tetap >60%, volume tidak mengering.",
            "Respons": "Boleh cari entry kecil bertahap; hindari all-in saat candle sudah terlalu jauh.",
        },
        {
            "Skenario": "Base / tunggu",
            "Trigger": f"Harga bergerak di antara support {support_label} dan high {high_label}, score tetap bagus.",
            "Respons": "Tunggu arah baru. Pullback ringan lebih sehat daripada mengejar pucuk range.",
        },
        {
            "Skenario": "Bearish / batal",
            "Trigger": f"Gagal bertahan di support {support_label} atau kembali mendekati low 24h {low_label}.",
            "Respons": "Setup batal untuk momentum. Review ulang setelah volume beli muncul lagi.",
        },
    ])

    checklist = [
        f"Breakout valid jika harga tidak langsung ditolak dari high 24h {high_label}.",
        f"Invalidasi awal ada di area {support_label}; invalidasi keras dekat low 24h {low_label}.",
        "Untuk entry receh, gunakan posisi kecil dulu dan tambah hanya kalau skenario bullish benar-benar aktif.",
        "Review ulang setelah snapshot berikutnya, terutama jika score turun atau volume 24h mulai melemah.",
    ]

    return {
        "verdict": verdict,
        "state": state,
        "forward_score": forward_score,
        "metrics": metrics,
        "outlook": outlook,
        "scenarios": scenarios,
        "checklist": checklist,
    }


def meme_flow_label(row):
    buy_m5 = safe_num(row.get("buy_ratio_m5"), np.nan)
    buy_h1 = safe_num(row.get("buy_ratio_h1"), np.nan)
    chg_m5 = safe_num(row.get("change_m5_pct"), 0)
    chg_h1 = safe_num(row.get("change_h1_pct"), 0)
    if not pd.isna(buy_m5) and not pd.isna(buy_h1) and buy_m5 >= 0.58 and buy_h1 >= 0.55 and chg_m5 >= 0:
        return "Buyer masih dominan di window pendek."
    if not pd.isna(buy_h1) and buy_h1 < 0.40:
        return "Sell pressure dominan; rawan distribusi."
    if chg_m5 < 0 and chg_h1 > 0:
        return "Momentum 1h masih hijau, tapi 5m mulai cooling off."
    if chg_h1 < -15:
        return "Dump 1h sedang aktif."
    return "Flow campuran; tunggu konfirmasi window berikutnya."


def build_meme_forward_analysis(row):
    radar = safe_num(row.get("radar_score"), 0)
    risk = safe_num(row.get("risk_score"), 100)
    liquidity = safe_num(row.get("liquidity_usd"), 0)
    liquidity_score = safe_num(row.get("liquidity_score"), 50)
    volume_score = safe_num(row.get("volume_score"), 50)
    buy_score = safe_num(row.get("buy_score"), 50)
    age_score = safe_num(row.get("age_score"), 50)
    age = safe_num(row.get("age_minutes"), np.nan)
    chg_m5 = safe_num(row.get("change_m5_pct"), 0)
    chg_h1 = safe_num(row.get("change_h1_pct"), 0)
    chg_h6 = safe_num(row.get("change_h6_pct"), 0)
    chg_h24 = safe_num(row.get("change_h24_pct"), 0)
    buy_m5 = safe_num(row.get("buy_ratio_m5"), np.nan)
    buy_h1 = safe_num(row.get("buy_ratio_h1"), np.nan)
    vol_liq_h1 = safe_num(row.get("volume_liq_h1"), np.nan)
    fdv_liq = safe_num(row.get("fdv_liq_ratio"), np.nan)

    quality_score = (
        radar * 0.55 +
        liquidity_score * 0.15 +
        volume_score * 0.10 +
        buy_score * 0.10 +
        age_score * 0.10 -
        risk * 0.45
    )
    if liquidity < 5_000:
        quality_score -= 20
    elif liquidity < 25_000:
        quality_score -= 8
    if not pd.isna(fdv_liq) and fdv_liq > 150:
        quality_score -= 10
    if not pd.isna(vol_liq_h1) and vol_liq_h1 > 5:
        quality_score -= 8
    quality_score = float(np.clip(quality_score, 0, 100))

    if risk >= 75 or liquidity < 5_000:
        verdict = "AVOID"
        state = "Risk terlalu dominan dibanding peluang."
    elif risk >= 55 or liquidity < 25_000:
        verdict = "WATCH ONLY"
        state = "Masih spekulatif, slippage dan exit risk perlu dianggap besar."
    elif chg_h1 >= 20 and not pd.isna(buy_h1) and buy_h1 >= 0.55 and radar >= 65:
        verdict = "SPECULATIVE MOMENTUM"
        state = "Momentum pendek kuat, tetap risk-first karena ini pair DEX."
    elif chg_h1 <= -20 or (not pd.isna(buy_h1) and buy_h1 < 0.35):
        verdict = "WAIT / DISTRIBUSI"
        state = "Tekanan jual lebih penting daripada radar score."
    elif radar >= 65 and risk < 55:
        verdict = "WATCH"
        state = "Ada aktivitas yang layak dipantau, tapi butuh refresh data."
    else:
        verdict = "LOW PRIORITY"
        state = "Belum ada keunggulan data yang jelas."

    if chg_m5 >= 0 and chg_h1 >= 0 and not pd.isna(buy_m5) and buy_m5 >= 0.55:
        short_bias = "Buyer masih memegang window 5m-1h; pantau apakah volume lanjut, bukan hanya spike."
    elif chg_m5 < 0 and chg_h1 > 0:
        short_bias = "Momentum mulai mendingin; tunggu retest atau buyer masuk lagi."
    elif chg_h1 < 0:
        short_bias = "Window pendek negatif; entry baru perlu konfirmasi reversal."
    else:
        short_bias = "Window pendek belum tegas; jangan ambil keputusan dari satu metrik saja."

    if risk >= 55:
        day_bias = "24 jam ke depan harus fokus bertahan hidup: risk, liquidity, dan sell pressure lebih penting dari pump."
    elif chg_h6 >= 0 and chg_h24 >= 0 and radar >= 65:
        day_bias = "Jika flow tetap hijau setelah beberapa refresh, pair bisa masuk watchlist spekulatif."
    else:
        day_bias = "Butuh bukti volume dan buy ratio tetap sehat sebelum dianggap punya lanjutan."

    if pd.isna(age) or age < 240:
        multi_day_bias = "Masih terlalu muda; analisis 2-3 hari sangat bergantung pada holder, lock liquidity, tax, dan security check."
    elif risk < 55 and liquidity >= 25_000:
        multi_day_bias = "Bisa dipantau beberapa hari kalau liquidity tidak turun dan social/holder makin kuat."
    else:
        multi_day_bias = "Jangan diperlakukan seperti posisi panjang sampai data kontrak dan liquidity lebih jelas."

    metrics = pd.DataFrame([
        {"Metric": "Verdict", "Value": verdict, "Bacaan": state},
        {"Metric": "Forward Quality", "Value": f"{quality_score:.1f}/100", "Bacaan": "Skor gabungan radar dikurangi risk dan kelemahan liquidity."},
        {"Metric": "Flow", "Value": meme_flow_label(row), "Bacaan": f"Buy ratio 5m/1h: {safe_num(buy_m5, 0.5) * 100:.1f}% / {safe_num(buy_h1, 0.5) * 100:.1f}%."},
        {"Metric": "Liquidity Risk", "Value": format_compact(liquidity, "$"), "Bacaan": "Di bawah $25K rawan slippage dan exit susah."},
        {"Metric": "FDV/Liquidity", "Value": ratio_text(fdv_liq), "Bacaan": "Rasio tinggi berarti valuasi mudah digerakkan oleh likuiditas kecil."},
        {"Metric": "Volume/Liquidity 1h", "Value": ratio_text(vol_liq_h1), "Bacaan": "Terlalu tinggi bisa berarti hype kuat atau wash/trap yang perlu dicurigai."},
    ])

    outlook = pd.DataFrame([
        {"Horizon": "15m-1h", "Bias": short_bias, "Yang Dipantau": "Buy ratio 5m/1h, change 5m, dan txns baru saat refresh."},
        {"Horizon": "24 jam", "Bias": day_bias, "Yang Dipantau": "Liquidity tidak turun tajam, risk score tidak memburuk, volume tetap organik."},
        {"Horizon": "2-3 hari", "Bias": multi_day_bias, "Yang Dipantau": "Holder, tax/honeypot, liquidity lock, creator wallet, dan konsistensi social."},
    ])

    scenarios = pd.DataFrame([
        {
            "Skenario": "Lanjut pump",
            "Trigger": "Buy ratio 5m/1h tetap >55%, change 5m tidak cepat merah, volume naik saat refresh.",
            "Respons": "Hanya spekulatif kecil; scale out lebih cepat saat candle memanjang.",
        },
        {
            "Skenario": "Cooling off",
            "Trigger": "Change 5m merah, buy ratio turun <50%, volume melemah setelah spike.",
            "Respons": "Wait. Jangan average down; tunggu data baru lebih bersih.",
        },
        {
            "Skenario": "Dump / rug risk",
            "Trigger": "Liquidity turun, sell pressure dominan, security flag muncul, atau risk score naik.",
            "Respons": "Avoid atau keluar sesuai rencana risiko. Jangan tunggu narasi social.",
        },
    ])

    checklist = [
        "Run Security Check sebelum AI atau sebelum entry manual.",
        "Cek holder, top wallet, tax buy/sell, honeypot, liquidity lock, dan creator wallet di explorer.",
        "Refresh snapshot 15-60 menit kemudian; meme coin yang sehat biasanya tidak hanya bagus di satu tarikan data.",
        "Untuk modal kecil, risiko utama bukan salah arah saja, tapi tidak bisa keluar karena liquidity tipis.",
    ]

    return {
        "verdict": verdict,
        "state": state,
        "quality_score": quality_score,
        "metrics": metrics,
        "outlook": outlook,
        "scenarios": scenarios,
        "checklist": checklist,
    }


def risk_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_tax(value):
    v = safe_num(value, np.nan)
    if pd.isna(v):
        return np.nan
    return v * 100 if 0 <= v <= 1 else v


@st.cache_data(ttl=600, show_spinner=False)
def fetch_goplus_security(chain, token_address):
    chain_id = CHAIN_SECURITY_IDS.get(str(chain).lower())
    if not chain_id or not token_address:
        return {"available": False, "error": "GoPlus belum didukung untuk chain ini."}
    try:
        data = fetch_public_json(
            f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}",
            params={"contract_addresses": token_address},
            timeout=20,
        )
        result = data.get("result", {}) if isinstance(data, dict) else {}
        token_data = result.get(str(token_address).lower()) or result.get(str(token_address)) or {}
        if not token_data:
            return {"available": False, "error": "GoPlus tidak mengembalikan data token."}
        return {"available": True, "chain_id": chain_id, "data": token_data}
    except Exception as exc:
        return {"available": False, "error": f"GoPlus error: {exc}"}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_honeypot_security(chain, token_address, pair_address=""):
    chain_id = HONEYPOT_CHAIN_IDS.get(str(chain).lower())
    if not chain_id or not token_address:
        return {"available": False, "error": "Honeypot.is hanya mendukung Ethereum, BSC, dan Base."}
    params = {"address": token_address, "chainID": chain_id}
    if pair_address:
        params["pair"] = pair_address
    try:
        data = fetch_public_json("https://api.honeypot.is/v2/IsHoneypot", params=params, timeout=25)
        return {"available": True, "chain_id": chain_id, "data": data}
    except Exception as exc:
        return {"available": False, "error": f"Honeypot.is error: {exc}"}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_rugcheck_security(token_address):
    if not token_address:
        return {"available": False, "error": "Token address kosong."}
    for suffix in ["report/summary", "report"]:
        try:
            data = fetch_public_json(f"https://api.rugcheck.xyz/v1/tokens/{token_address}/{suffix}", timeout=25)
            return {"available": True, "data": data, "endpoint": suffix}
        except Exception as exc:
            last_error = exc
    return {"available": False, "error": f"RugCheck error: {last_error}"}


def summarize_goplus(report):
    if not report.get("available"):
        return [], [], report.get("error", "GoPlus tidak tersedia.")
    data = report.get("data", {})
    flags = []
    rows = []
    severe_fields = {
        "is_honeypot": "GoPlus: indikasi honeypot",
        "cannot_sell_all": "GoPlus: token tidak bisa dijual penuh",
        "is_blacklisted": "GoPlus: blacklist aktif",
        "is_mintable": "GoPlus: mintable",
        "hidden_owner": "GoPlus: hidden owner",
        "owner_change_balance": "GoPlus: owner bisa ubah balance",
        "selfdestruct": "GoPlus: selfdestruct risk",
        "transfer_pausable": "GoPlus: transfer bisa dipause",
    }
    warn_fields = {
        "external_call": "GoPlus: external call",
        "trading_cooldown": "GoPlus: trading cooldown",
        "personal_slippage_modifiable": "GoPlus: slippage personal bisa diubah",
        "slippage_modifiable": "GoPlus: slippage bisa diubah",
        "is_proxy": "GoPlus: proxy contract",
        "is_whitelisted": "GoPlus: whitelist mechanism",
    }
    for field, label in severe_fields.items():
        if risk_bool(data.get(field)):
            flags.append(("High", label))
    for field, label in warn_fields.items():
        if risk_bool(data.get(field)):
            flags.append(("Medium", label))
    if str(data.get("is_open_source", "")).strip() == "0":
        flags.append(("High", "GoPlus: contract tidak open source"))
    buy_tax = safe_tax(data.get("buy_tax"))
    sell_tax = safe_tax(data.get("sell_tax"))
    for label, tax in [("Buy tax", buy_tax), ("Sell tax", sell_tax)]:
        if not pd.isna(tax):
            rows.append({"Metric": f"GoPlus {label}", "Value": f"{tax:.2f}%"})
            if tax >= 25:
                flags.append(("High", f"GoPlus: {label.lower()} sangat tinggi ({tax:.1f}%)"))
            elif tax >= 10:
                flags.append(("Medium", f"GoPlus: {label.lower()} tinggi ({tax:.1f}%)"))
    for key in ["holder_count", "lp_holder_count", "creator_address", "owner_address"]:
        if data.get(key) not in [None, ""]:
            rows.append({"Metric": f"GoPlus {key}", "Value": str(data.get(key))})
    return flags, rows, "GoPlus aktif"


def summarize_honeypot(report):
    if not report.get("available"):
        return [], [], report.get("error", "Honeypot.is tidak tersedia.")
    data = report.get("data", {})
    flags = []
    rows = []
    hp = data.get("honeypotResult") or {}
    sim = data.get("simulationResult") or {}
    contract_code = data.get("contractCode") or {}
    if risk_bool(hp.get("isHoneypot")):
        flags.append(("High", f"Honeypot.is: honeypot terdeteksi ({hp.get('honeypotReason', '-')})"))
    for label, key in [("Buy tax", "buyTax"), ("Sell tax", "sellTax"), ("Transfer tax", "transferTax")]:
        tax = safe_tax(sim.get(key))
        if not pd.isna(tax):
            rows.append({"Metric": f"Honeypot {label}", "Value": f"{tax:.2f}%"})
            if tax >= 25:
                flags.append(("High", f"Honeypot.is: {label.lower()} sangat tinggi ({tax:.1f}%)"))
            elif tax >= 10:
                flags.append(("Medium", f"Honeypot.is: {label.lower()} tinggi ({tax:.1f}%)"))
    if contract_code:
        rows.append({"Metric": "Honeypot openSource", "Value": str(contract_code.get("openSource", "N/A"))})
        rows.append({"Metric": "Honeypot isProxy", "Value": str(contract_code.get("isProxy", "N/A"))})
        if contract_code.get("openSource") is False:
            flags.append(("High", "Honeypot.is: contract tidak open source"))
    for item in data.get("flags") or []:
        text = item.get("description") if isinstance(item, dict) else str(item)
        flags.append(("Medium", f"Honeypot.is flag: {text}"))
    return flags, rows, "Honeypot.is aktif"


def summarize_rugcheck(report):
    if not report.get("available"):
        return [], [], report.get("error", "RugCheck tidak tersedia.")
    data = report.get("data", {})
    flags = []
    rows = []
    score = safe_num(data.get("score"), np.nan)
    if not pd.isna(score):
        rows.append({"Metric": "RugCheck score", "Value": f"{score:.0f}"})
        if score >= 10_000:
            flags.append(("High", f"RugCheck: score risiko tinggi ({score:.0f})"))
        elif score >= 1_000:
            flags.append(("Medium", f"RugCheck: score perlu perhatian ({score:.0f})"))
    for key in ["riskLevel", "verification", "tokenType"]:
        if data.get(key) not in [None, ""]:
            rows.append({"Metric": f"RugCheck {key}", "Value": str(data.get(key))})
    for risk in data.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        level = str(risk.get("level", "Medium")).title()
        name = risk.get("name") or risk.get("description") or "RugCheck risk"
        flags.append(("High" if level == "Danger" else level, f"RugCheck: {name}"))
    return flags, rows, "RugCheck aktif"


def run_security_checks(row):
    chain = str(row.get("chain", "")).lower()
    token_address = str(row.get("base_address", "")).strip()
    pair_address = str(row.get("pair_address", "")).strip()
    reports = {}
    if chain == "solana":
        reports["rugcheck"] = fetch_rugcheck_security(token_address)
    else:
        reports["goplus"] = fetch_goplus_security(chain, token_address)
        reports["honeypot"] = fetch_honeypot_security(chain, token_address, pair_address)
    flags, rows, statuses = [], [], []
    for key, summarizer in [
        ("goplus", summarize_goplus),
        ("honeypot", summarize_honeypot),
        ("rugcheck", summarize_rugcheck),
    ]:
        if key not in reports:
            continue
        f, r, status = summarizer(reports[key])
        flags.extend(f)
        rows.extend(r)
        statuses.append(status)
    high = sum(1 for level, _ in flags if str(level).lower() in {"high", "danger", "critical"})
    medium = sum(1 for level, _ in flags if str(level).lower() == "medium")
    security_score = int(np.clip(100 - high * 25 - medium * 10, 0, 100))
    if security_score >= 80:
        label = "Relatif aman"
    elif security_score >= 60:
        label = "Perlu cek manual"
    elif security_score >= 35:
        label = "Berisiko"
    else:
        label = "Bahaya"
    return {
        "score": security_score,
        "label": label,
        "flags": flags,
        "rows": rows,
        "statuses": statuses,
        "reports": reports,
    }


def history_row_from_cex(row, item_id=None):
    symbol = str(row.get("symbol", "")).upper()
    source = str(row.get("source", "indodax")).lower()
    return {
        "ts": utc_now_iso(),
        "id": item_id or f"cex:{source}:{symbol}",
        "type": "cex",
        "label": symbol,
        "price": safe_num(row.get("last_price"), np.nan),
        "score": safe_num(row.get("crypto_score"), np.nan),
        "risk": np.nan,
        "volume": safe_num(row.get("quote_volume"), np.nan),
        "change": safe_num(row.get("change_24h_pct"), np.nan),
    }


def history_row_from_dex(row, item_id=None):
    chain = str(row.get("chain", "")).lower()
    pair_address = str(row.get("pair_address", "")).lower()
    return {
        "ts": utc_now_iso(),
        "id": item_id or f"dex:{chain}:{pair_address}",
        "type": "dex",
        "label": row.get("pair", ""),
        "price": safe_num(row.get("price_usd"), np.nan),
        "score": safe_num(row.get("radar_score"), np.nan),
        "risk": safe_num(row.get("risk_score"), np.nan),
        "volume": safe_num(row.get("volume_h1"), np.nan),
        "change": safe_num(row.get("change_h1_pct"), np.nan),
    }


def history_row_from_stock(row, item_id=None):
    ticker = str(row.get("kode saham", row.get("name", ""))).upper()
    symbol = str(row.get("symbol") or f"IDX:{ticker}").upper()
    return {
        "ts": utc_now_iso(),
        "id": item_id or f"stock:{symbol}",
        "type": "stock",
        "label": ticker,
        "price": safe_num(row.get("close"), np.nan),
        "score": safe_num(row.get("auto_score"), np.nan),
        "risk": np.nan,
        "volume": safe_num(row.get("volume"), np.nan),
        "value_traded": safe_num(row.get("value_traded"), np.nan),
        "change": safe_num(row.get("change_pct"), np.nan),
        "rel_volume": safe_num(row.get("rel_volume"), np.nan),
        "status": row.get("signal_label", ""),
        "rec": row.get("rec_all", ""),
    }


def evaluate_alerts_for_stock(row, rules):
    alerts = []
    label = str(row.get("kode saham", row.get("name", ""))).upper()
    score = safe_num(row.get("auto_score"), 0)
    change = safe_num(row.get("change_pct"), 0)
    rel_volume = safe_num(row.get("rel_volume"), 0)
    signal = str(row.get("signal_label", ""))
    if score >= rules["stock_min_score"]:
        alerts.append({"Level": "Info", "Asset": label, "Alert": f"Auto score >= {rules['stock_min_score']:.0f}", "Value": f"{score:.1f}"})
    if change >= rules["stock_min_change"]:
        alerts.append({"Level": "Momentum", "Asset": label, "Alert": f"Change >= {rules['stock_min_change']:.1f}%", "Value": pct_text(change)})
    if change <= rules["stock_dump_change"]:
        alerts.append({"Level": "Risk", "Asset": label, "Alert": f"Drop <= {rules['stock_dump_change']:.1f}%", "Value": pct_text(change)})
    if rel_volume >= rules["stock_min_rel_volume"]:
        alerts.append({"Level": "Volume", "Asset": label, "Alert": f"Rel volume >= {rules['stock_min_rel_volume']:.1f}x", "Value": ratio_text(rel_volume)})
    if signal == "Breakout Candidate":
        alerts.append({"Level": "Momentum", "Asset": label, "Alert": "Breakout Candidate", "Value": f"{score:.1f}"})
    return alerts


def evaluate_alerts_for_cex(row, rules):
    alerts = []
    label = str(row.get("symbol", ""))
    score = safe_num(row.get("crypto_score"), 0)
    change = safe_num(row.get("change_24h_pct"), 0)
    if score >= rules["cex_min_score"]:
        alerts.append({"Level": "Info", "Asset": label, "Alert": f"Score >= {rules['cex_min_score']:.0f}", "Value": f"{score:.1f}"})
    if change >= rules["cex_min_24h_change"]:
        alerts.append({"Level": "Momentum", "Asset": label, "Alert": f"24h change >= {rules['cex_min_24h_change']:.1f}%", "Value": pct_text(change)})
    if change <= rules["cex_dump_24h_change"]:
        alerts.append({"Level": "Risk", "Asset": label, "Alert": f"Dump 24h <= {rules['cex_dump_24h_change']:.1f}%", "Value": pct_text(change)})
    return alerts


def evaluate_alerts_for_dex(row, rules):
    alerts = []
    label = str(row.get("pair", ""))
    radar = safe_num(row.get("radar_score"), 0)
    risk = safe_num(row.get("risk_score"), 0)
    liquidity = safe_num(row.get("liquidity_usd"), 0)
    change = safe_num(row.get("change_h1_pct"), 0)
    buy_ratio = safe_num(row.get("buy_ratio_h1"), 0)
    if radar >= rules["dex_min_radar"]:
        alerts.append({"Level": "Momentum", "Asset": label, "Alert": f"Radar >= {rules['dex_min_radar']:.0f}", "Value": f"{radar:.1f}"})
    if risk >= rules["dex_max_risk"]:
        alerts.append({"Level": "Risk", "Asset": label, "Alert": f"Risk >= {rules['dex_max_risk']:.0f}", "Value": f"{risk:.1f}"})
    if liquidity < rules["dex_min_liquidity"]:
        alerts.append({"Level": "Risk", "Asset": label, "Alert": f"Liquidity < {format_compact(rules['dex_min_liquidity'], '$')}", "Value": format_compact(liquidity, "$")})
    if change >= rules["dex_min_h1_change"]:
        alerts.append({"Level": "Momentum", "Asset": label, "Alert": f"1h change >= {rules['dex_min_h1_change']:.1f}%", "Value": pct_text(change)})
    if change <= rules["dex_dump_h1_change"]:
        alerts.append({"Level": "Risk", "Asset": label, "Alert": f"Dump 1h <= {rules['dex_dump_h1_change']:.1f}%", "Value": pct_text(change)})
    if buy_ratio >= rules["dex_min_buy_ratio"]:
        alerts.append({"Level": "Flow", "Asset": label, "Alert": f"Buy ratio >= {rules['dex_min_buy_ratio'] * 100:.0f}%", "Value": f"{buy_ratio * 100:.1f}%"})
    return alerts


def refresh_watchlist_snapshot(run_security=False):
    items = load_watchlist()
    now_rows, history_rows, alerts = [], [], []
    rules = load_alert_rules()
    stock_items = [item for item in items if item.get("type") == "stock"]
    cex_items = [
        item for item in items
        if item.get("type") == "cex" and str(item.get("symbol", "")).upper().endswith("IDR")
    ]
    dex_items = []

    if stock_items:
        try:
            rows, _ = fetch_tv_stock_screener(limit=900, sort_by="Value.Traded", sort_order="desc")
            stock_df = normalize_stock_auto_rows(rows)
            for item in stock_items:
                symbol = str(item.get("symbol") or f"IDX:{item.get('ticker', '')}").upper()
                ticker = symbol.split(":")[-1]
                row_df = stock_df[stock_df["kode saham"] == ticker] if not stock_df.empty else pd.DataFrame()
                if row_df.empty:
                    now_rows.append({"id": item.get("id"), "type": "stock", "label": item.get("label"), "status": "Tidak ditemukan"})
                    continue
                row = row_df.iloc[0]
                history_rows.append(history_row_from_stock(row, item.get("id")))
                alerts.extend(evaluate_alerts_for_stock(row, rules))
                now_rows.append({
                    "id": item.get("id"),
                    "type": "stock",
                    "label": row.get("kode saham"),
                    "source": "TradingView IDX",
                    "price": safe_num(row.get("close"), np.nan),
                    "score": safe_num(row.get("auto_score"), np.nan),
                    "risk": np.nan,
                    "volume": safe_num(row.get("volume"), np.nan),
                    "value_traded": safe_num(row.get("value_traded"), np.nan),
                    "change": safe_num(row.get("change_pct"), np.nan),
                    "rel_volume": safe_num(row.get("rel_volume"), np.nan),
                    "status": row.get("signal_label", ""),
                })
        except Exception as exc:
            now_rows.append({"id": "stock:error", "type": "stock", "label": "TradingView IDX", "status": f"Error: {exc}"})

    if cex_items:
        try:
            normalized_parts = []
            quotes = sorted(set(str(item.get("quote") or "USDT").upper() for item in cex_items))
            for quote in quotes:
                try:
                    part, _, _ = fetch_crypto_market_df(source="Auto", quote=quote)
                    if not part.empty:
                        normalized_parts.append(part)
                except Exception:
                    continue
            cex_df = pd.concat(normalized_parts, ignore_index=True) if normalized_parts else pd.DataFrame()
            for item in cex_items:
                symbol = str(item.get("symbol", "")).upper()
                row_df = cex_df[cex_df["symbol"] == symbol] if not cex_df.empty else pd.DataFrame()
                if row_df.empty:
                    now_rows.append({"id": item.get("id"), "type": "cex", "label": item.get("label"), "status": "Tidak ditemukan"})
                    continue
                row = row_df.iloc[0]
                history_rows.append(history_row_from_cex(row, item.get("id")))
                alerts.extend(evaluate_alerts_for_cex(row, rules))
                now_rows.append({
                    "id": item.get("id"),
                    "type": "cex",
                    "label": symbol,
                    "price": safe_num(row.get("last_price"), np.nan),
                    "score": safe_num(row.get("crypto_score"), np.nan),
                    "risk": np.nan,
                    "volume": safe_num(row.get("quote_volume"), np.nan),
                    "change": safe_num(row.get("change_24h_pct"), np.nan),
                    "status": row.get("signal", ""),
                })
        except Exception as exc:
            now_rows.append({"id": "cex:error", "type": "cex", "label": "Indodax IDR", "status": f"Error: {exc}"})

    for item in dex_items:
        try:
            pairs = fetch_dex_pairs(item.get("chain", ""), item.get("pair_address", ""))
            df = normalize_dex_pairs(pairs)
            if df.empty:
                now_rows.append({"id": item.get("id"), "type": "dex", "label": item.get("label"), "status": "Tidak ditemukan"})
                continue
            row = df.iloc[0]
            security_label = ""
            security_score = np.nan
            if run_security:
                security = run_security_checks(row)
                security_label = security["label"]
                security_score = security["score"]
            history_rows.append(history_row_from_dex(row, item.get("id")))
            alerts.extend(evaluate_alerts_for_dex(row, rules))
            now_rows.append({
                "id": item.get("id"),
                "type": "dex",
                "label": row.get("pair"),
                "chain": row.get("chain"),
                "price": safe_num(row.get("price_usd"), np.nan),
                "score": safe_num(row.get("radar_score"), np.nan),
                "risk": safe_num(row.get("risk_score"), np.nan),
                "security": security_score,
                "security_label": security_label,
                "volume": safe_num(row.get("volume_h1"), np.nan),
                "change": safe_num(row.get("change_h1_pct"), np.nan),
                "status": row.get("risk_label", ""),
            })
        except Exception as exc:
            now_rows.append({"id": item.get("id"), "type": "dex", "label": item.get("label"), "status": f"Error: {exc}"})

    append_history_rows(history_rows)
    return pd.DataFrame(now_rows), pd.DataFrame(alerts), len(history_rows)


def build_crypto_market_prompt(row, chart_ctx=None, news_rows=None, community=None, orderbook=None):
    forward = build_crypto_forward_analysis(row)
    return f"""Kamu analis crypto market IDR Indonesia untuk trader retail modal kecil. Tulis analisis detail dalam bahasa Indonesia, bukan rekomendasi finansial.

DATA MARKET:
- Symbol: {row.get("symbol")} ({row.get("base")})
- Exchange/source: {row.get("exchange", row.get("source"))}
- Status market: {row.get("market_status", "N/A")}
- Last price: {format_market_price(row)}
- Change 24h: {pct_text(row.get("change_24h_pct"))}
- Change 7d: {pct_text(row.get("change_7d_pct"))}
- High/Low 24h: {format_market_price(row, "high_24h")} / {format_market_price(row, "low_24h")}
- Volume IDR 24h: {format_market_amount(row)}
- Trade count 24h: {format_compact(row.get("trade_count"))}
- Score: {safe_num(row.get("crypto_score"), 0):.1f}/100
- Signal: {row.get("signal")}
- Forward verdict: {forward["verdict"]}
- Forward score: {forward["forward_score"]:.1f}/100
- Kondisi sekarang: {forward["state"]}

CHART REAL / OHLCV INDODAX:
{chart_summary_lines(chart_ctx, row)}

BERITA TERBARU:
{news_summary_lines(news_rows or [])}

KOMUNITAS:
{community_summary_lines(community or {})}

ORDERBOOK / LIQUIDITY:
{orderbook_summary_lines(orderbook or {})}

FORMAT:
**Verdict:** [MOMENTUM | WATCH | WAIT | AVOID] + 1 kalimat alasan.
**Kondisi Sekarang:** 4-6 poin, jelaskan momentum, volume, posisi range, dan apakah rawan chase.
**Chart Reading:** baca struktur daily dan hourly dari OHLCV, RSI, MACD, EMA, Bollinger, support/resistance, volume relatif.
**News Impact:** ringkas sentimen berita, apakah mendukung, netral, atau jadi risiko.
**Community Check:** apakah komunitas besar/aktif atau lemah; jelaskan dampaknya ke risiko likuiditas dan hype.
**Orderbook Risk:** baca spread, depth, dan estimasi slippage; beri warning jika tipis.
**Analisis Ke Depan:**
- 1-4 jam:
- 24 jam:
- 3-7 hari:
**Scenario Map:**
- Bullish lanjut: trigger, cara respons, risiko.
- Base / tunggu: trigger, cara respons.
- Bearish / batal: trigger invalidasi, cara respons.
**Level Praktis:** area breakout, support/invalidasi, dan take profit bertahap. Jika level tidak cukup kuat dari data, bilang "butuh chart manual".
**Risk Plan Modal Kecil:** ukuran posisi, stop, kapan tidak entry, dan kapan review ulang.
**Kesimpulan Simpel:** format "Kalau X, maka Y. Kalau Z, wait."

Jaga bahasa tegas, detail, dan tidak membingungkan. Jangan mengaku melihat gambar chart; kamu membaca data OHLCV/indikator chart yang diberikan. Jangan menjanjikan cuan. Target 850-1200 kata."""


def build_meme_idr_prompt(row, chart_ctx=None, news_rows=None, community=None, orderbook=None):
    forward = build_crypto_forward_analysis(row)
    return f"""Kamu analis meme coin yang punya pair IDR di exchange Indonesia. Tulis analisis risk-first dalam bahasa Indonesia.

DATA MARKET IDR:
- Symbol: {row.get("symbol")} ({row.get("asset_name", row.get("base"))})
- Exchange/source: {row.get("exchange", row.get("source"))}
- Status market: {row.get("market_status", "N/A")}
- Last price: {format_market_price(row)}
- Change 24h / 7d: {pct_text(row.get("change_24h_pct"))} / {pct_text(row.get("change_7d_pct"))}
- High/Low 24h: {format_market_price(row, "high_24h")} / {format_market_price(row, "low_24h")}
- Volume IDR 24h: {format_market_amount(row)}
- Buy/Sell: {format_market_price(row, "buy_price")} / {format_market_price(row, "sell_price")}
- Score: {safe_num(row.get("crypto_score"), 0):.1f}/100
- Signal: {row.get("signal")}
- Forward verdict: {forward["verdict"]}
- Forward score: {forward["forward_score"]:.1f}/100

CHART REAL / OHLCV INDODAX:
{chart_summary_lines(chart_ctx, row)}

BERITA TERBARU:
{news_summary_lines(news_rows or [])}

KOMUNITAS:
{community_summary_lines(community or {})}

ORDERBOOK / LIQUIDITY:
{orderbook_summary_lines(orderbook or {})}

FORMAT:
**Verdict:** [AVOID | WATCH | SPECULATIVE ONLY] + alasan singkat.
**Kondisi Sekarang:** momentum 24h/7d, volume IDR, spread buy/sell, status market.
**Chart Reading:** baca struktur daily dan hourly dari OHLCV, RSI, MACD, EMA, Bollinger, support/resistance, volume relatif.
**News Impact:** ringkas sentimen berita, hype, risiko rumor, dan apakah berita cukup kuat untuk mendukung momentum.
**Community Check:** nilai apakah komunitas besar/aktif atau lemah; jelaskan dampaknya ke meme coin.
**Orderbook Risk:** baca spread, depth, dan estimasi slippage; beri warning jika exit risk besar.
**Analisis Ke Depan:**
- 1-4 jam:
- 24 jam:
- 3-7 hari:
**Scenario Map:**
- Lanjut pump:
- Cooling off:
- Dump:
**Risk Plan Modal Kecil:** ukuran posisi, invalidasi, take profit bertahap, dan aturan jangan average down.
**Kesimpulan Simpel:** masuk hanya kalau apa, hindari kalau apa.

Ingat: meme coin tetap sangat spekulatif walaupun sudah punya pair IDR. Jangan mengaku melihat gambar chart; kamu membaca data OHLCV/indikator chart yang diberikan. Jangan menjanjikan cuan. Target 850-1200 kata."""


def build_meme_prompt(row, security=None):
    flags = "; ".join(get_pair_flags(row))
    forward = build_meme_forward_analysis(row)
    security_sec = "SECURITY API: Belum dijalankan."
    if security:
        sec_flags = "; ".join([f"{level}: {text}" for level, text in security.get("flags", [])]) or "Tidak ada flag besar."
        security_sec = (
            f"SECURITY API:\n"
            f"- Security score: {security.get('score')}/100 ({security.get('label')})\n"
            f"- Provider: {', '.join(security.get('statuses', [])) or 'N/A'}\n"
            f"- Flags: {sec_flags}"
        )
    return f"""Kamu analis meme coin on-chain. Fokus ke risk-first screening untuk trader modal kecil. Tulis analisis detail bahasa Indonesia.

DATA PAIR:
- Pair: {row.get("pair")} di {row.get("chain")} / {row.get("dex")}
- Price USD: {row.get("price_usd")}
- Liquidity: {format_compact(row.get("liquidity_usd"), "$")}
- FDV: {format_compact(row.get("fdv"), "$")}
- Market cap: {format_compact(row.get("market_cap"), "$")}
- Age: {format_age(row.get("age_minutes"))}
- Volume 5m/1h/24h: {format_compact(row.get("volume_m5"), "$")} / {format_compact(row.get("volume_h1"), "$")} / {format_compact(row.get("volume_h24"), "$")}
- Change 5m/1h/24h: {pct_text(row.get("change_m5_pct"))} / {pct_text(row.get("change_h1_pct"))} / {pct_text(row.get("change_h24_pct"))}
- Txns 5m/1h/24h: {safe_num(row.get("txns_m5"), 0):.0f} / {safe_num(row.get("txns_h1"), 0):.0f} / {safe_num(row.get("txns_h24"), 0):.0f}
- Buys-Sells 1h: {safe_num(row.get("buys_h1"), 0):.0f} vs {safe_num(row.get("sells_h1"), 0):.0f}
- Buy ratio 5m/1h/24h: {safe_num(row.get("buy_ratio_m5"), 0.5) * 100:.1f}% / {safe_num(row.get("buy_ratio_h1"), 0.5) * 100:.1f}% / {safe_num(row.get("buy_ratio_h24"), 0.5) * 100:.1f}%
- FDV/Liquidity: {ratio_text(row.get("fdv_liq_ratio"))}
- Volume/Liquidity 1h: {ratio_text(row.get("volume_liq_h1"))}
- Links tersedia: {row.get("link_summary", "N/A")}
- Base contract: {row.get("base_address")}
- Radar score: {safe_num(row.get("radar_score"), 0):.1f}/100
- Risk score: {safe_num(row.get("risk_score"), 0):.1f}/100 ({row.get("risk_label")})
- Flags: {flags}
- Forward verdict: {forward["verdict"]}
- Forward quality: {forward["quality_score"]:.1f}/100
- Kondisi sekarang: {forward["state"]}

{security_sec}

FORMAT:
**Verdict:** [AVOID | WATCH | SPECULATIVE ONLY]
**Kondisi Sekarang:** 4-6 poin, pisahkan momentum, flow buyer/seller, liquidity, age, FDV/liquidity.
**Analisis Ke Depan:**
- 15m-1h:
- 24 jam:
- 2-3 hari:
**Scenario Map:**
- Lanjut pump: trigger valid dan respons.
- Cooling off: trigger wait.
- Dump/rug risk: trigger bahaya dan respons.
**Red Flag:** maksimal 6 poin, urutkan dari paling bahaya.
**Plan Receh Risk-First:** ukuran posisi, invalidasi, take profit bertahap, dan aturan jangan average down.
**Data Wajib Cek Manual:** contract, holder/top wallet, tax/honeypot, liquidity lock, creator wallet, social.
**Kesimpulan Simpel:** format "Masuk hanya kalau X. Hindari kalau Y."

Jangan menjanjikan cuan. Kalau data risk tinggi, bilang tegas. Target 750-1000 kata."""


def render_openrouter_controls(prefix):
    st.markdown("---")
    st.header("AI")
    static_key = get_static_openrouter_key()
    model = get_openrouter_model()
    st.caption(f"Model aktif: {model}")
    if static_key:
        st.success("API key statis terdeteksi.")
        with st.expander("Override API Key"):
            override = st.text_input("Override Key", type="password", placeholder="sk-or-v1-...", key=f"{prefix}_override")
        return (override.strip() if override else static_key), model
    key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...", key=f"{prefix}_key")
    return key.strip(), model


def set_page(page):
    st.session_state["page"] = page
    st.rerun()


def render_home():
    st.markdown("""
    <style>
    .home-card {
        min-height: 190px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 22px;
        background: rgba(15, 23, 42, 0.42);
    }
    .home-card h3 { margin-top: 0; margin-bottom: 10px; }
    .home-card p { color: rgba(226, 232, 240, 0.82); line-height: 1.55; }
    </style>
    """, unsafe_allow_html=True)
    st.title("Market Screener")
    st.caption("Pilih mode analisis. Saham BEI tetap utama, crypto/meme coin dibatasi ke market IDR Indonesia.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="home-card">
          <h3>Saham BEI</h3>
          <p>Auto scanner saham IDX dari TradingView. Upload BEI tersedia sebagai advanced mode untuk foreign/broker flow.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buka Saham BEI", width="stretch"):
            set_page("Saham BEI")
    with c2:
        st.markdown("""
        <div class="home-card">
          <h3>Crypto Market</h3>
          <p>Pair crypto IDR dari Indodax. Cocok untuk BTC, ETH, SOL, dan altcoin yang tersedia di market Rupiah.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buka Crypto Market", width="stretch"):
            set_page("Crypto Market")
    with c3:
        st.markdown("""
        <div class="home-card">
          <h3>Meme Coin Radar</h3>
          <p>Meme coin yang punya pair IDR di exchange Indonesia, bukan pair DEX global.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buka Meme Coin Radar", width="stretch"):
            set_page("Meme Coin Radar")
    with c4:
        st.markdown("""
        <div class="home-card">
          <h3>Watchlist & Alerts</h3>
          <p>Simpan kandidat saham/crypto IDR, refresh snapshot, lihat alert aktif, dan pantau history score lokal.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buka Watchlist", width="stretch"):
            set_page("Watchlist & Alerts")
    st.markdown("---")
    st.info("Catatan: semua output adalah alat bantu analisis, bukan rekomendasi investasi.")
    render_deploy_readiness_panel()


def render_crypto_market_page():
    st.title("Crypto Market IDR")
    st.caption("Data utama dari Indodax public API: pair crypto yang tersedia di market Rupiah Indonesia.")
    with st.sidebar:
        st.header("Filter Crypto IDR")
        source = st.selectbox("Source", CRYPTO_MARKET_SOURCES, index=0)
        quote = "IDR"
        st.caption("Quote dikunci ke IDR supaya tidak mencampur market global non-Rupiah.")
        if source == "CoinGecko IDR":
            st.info("CoinGecko IDR hanya konversi harga Rupiah global. Untuk listing Indonesia, gunakan Indodax IDR.")
        min_volume = st.number_input("Min volume 24h IDR", min_value=0, value=10_000_000, step=10_000_000)
        sort_by = st.selectbox("Urutkan", ["crypto_score", "quote_volume", "change_24h_pct", "change_7d_pct"], index=0)
        top_n = st.number_input("Top N", min_value=5, max_value=300, value=50, step=5)
        search = st.text_input("Cari symbol", placeholder="BTC, ETH, SOL")
        openrouter_key, llm_model = render_openrouter_controls("crypto_market")

    try:
        with st.spinner("Mengambil data crypto IDR..."):
            df, source_url, active_source = fetch_crypto_market_df(source=source, quote=quote)
    except Exception as exc:
        st.error(f"Data crypto gagal dimuat: {exc}")
        st.stop()

    if df.empty:
        st.warning("Tidak ada data untuk filter ini.")
        st.stop()

    view = df[df["quote_volume"].fillna(0) >= float(min_volume)].copy()
    if search:
        q = search.upper().strip()
        view = view[view["symbol"].str.contains(q, na=False) | view["base"].str.contains(q, na=False)]
    view = view.sort_values(sort_by, ascending=False).head(int(top_n))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pairs", f"{len(view)}")
    s2.metric("Source", active_source)
    s3.metric("Top Volume", format_idr(view["quote_volume"].max()))
    s4.metric("Avg Change", pct_text(view["change_24h_pct"].mean()))
    st.caption(f"Endpoint aktif: {source_url}. Semua angka harga/volume memakai IDR.")

    display_cols = [
        "symbol", "asset_name", "last_price", "change_24h_pct", "change_7d_pct",
        "quote_volume", "market_status", "crypto_score", "signal",
    ]
    st.subheader("Market List")
    disp = view[display_cols].rename(columns={
        "last_price": "price",
        "change_24h_pct": "chg_24h%",
        "change_7d_pct": "chg_7d%",
        "quote_volume": "volume_24h",
        "market_status": "status",
        "crypto_score": "score",
    })
    disp["price"] = view.apply(lambda row: format_market_price(row), axis=1).values
    disp["volume_24h"] = view.apply(lambda row: format_market_amount(row), axis=1).values
    render_df_with_style_fallback(disp, ["score"])
    st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), "crypto_market.csv", "text/csv")

    if view.empty:
        st.warning("Tidak ada coin setelah filter.")
        st.stop()

    st.markdown("---")
    selected_symbol = st.selectbox("Pilih coin", view["symbol"].tolist())
    sel = view[view["symbol"] == selected_symbol].iloc[0]
    st.subheader(f"{sel['symbol']} - {sel['signal']}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", format_market_price(sel))
    m2.metric("24h", pct_text(sel.get("change_24h_pct")))
    m3.metric("Volume", format_market_amount(sel))
    m4.metric("7d", pct_text(sel.get("change_7d_pct")))
    m5.metric("Score", f"{safe_num(sel.get('crypto_score'), 0):.1f}")

    cta1, cta2, cta3 = st.columns([1, 1, 2])
    cex_item = watch_item_from_cex_row(sel)
    with cta1:
        if is_watchlisted(cex_item["id"]):
            if st.button("Hapus Watchlist", width="stretch", key=f"rm_{cex_item['id']}"):
                remove_watchlist_item(cex_item["id"])
                st.rerun()
        elif st.button("Tambah Watchlist", width="stretch", key=f"add_{cex_item['id']}"):
            add_watchlist_item(cex_item)
            st.success("Masuk watchlist.")
    with cta2:
        if st.button("Simpan Snapshot", width="stretch", key=f"snap_{cex_item['id']}"):
            append_history_rows([history_row_from_cex(sel, cex_item["id"])])
            st.success("Snapshot tersimpan ke history.")
    with cta3:
        st.caption("Watchlist dan history disimpan lokal di folder data/.")

    detail_options = ["Ringkasan", "Outlook", "Chart", "Liquidity", "News & Community", "AI"]
    section = st.radio(
        "Detail Crypto",
        detail_options,
        horizontal=True,
        label_visibility="collapsed",
        key=f"crypto_detail_{selected_symbol}",
    )
    if section == "Ringkasan":
        st.dataframe(pd.DataFrame({
            "Metric": ["Open 24h", "High 24h", "Low 24h", "Weighted Avg", "Liquidity Score", "Momentum Score", "Activity Score"],
            "Value": [
                format_market_price(sel, "open_24h"),
                format_market_price(sel, "high_24h"),
                format_market_price(sel, "low_24h"),
                format_market_price(sel, "weighted_avg"),
                f"{safe_num(sel.get('liquidity_score'), 0):.1f}",
                f"{safe_num(sel.get('momentum_score'), 0):.1f}",
                f"{safe_num(sel.get('activity_score'), 0):.1f}",
            ],
        }), width="stretch")
    elif section == "Outlook":
        crypto_outlook = build_crypto_forward_analysis(sel)
        o1, o2, o3 = st.columns(3)
        o1.metric("Forward Score", f"{crypto_outlook['forward_score']:.1f}/100")
        o2.metric("Verdict", crypto_outlook["verdict"])
        o3.metric("Signal", str(sel.get("signal", "N/A")))
        st.markdown("**Bacaan Detail**")
        st.dataframe(crypto_outlook["metrics"], width="stretch")
        st.markdown("**Analisis Ke Depan**")
        st.dataframe(crypto_outlook["outlook"], width="stretch")
        st.markdown("**Scenario Map**")
        st.dataframe(crypto_outlook["scenarios"], width="stretch")
        st.markdown("**Checklist Praktis**")
        st.markdown("\n".join(f"- {item}" for item in crypto_outlook["checklist"]))
    elif section == "Chart":
        chart_ctx = build_chart_context(sel) if sel.get("source") == "Indodax" else {}
        tv_sym = f"INDODAX:{selected_symbol}" if sel.get("source") == "Indodax" else f"CRYPTO:{selected_symbol}"
        render_tradingview_advanced_chart(tv_sym, interval="60", height=580)
        render_chart_analysis_panel(sel, chart_ctx)
    elif section == "Liquidity":
        orderbook = orderbook_from_row(sel) if sel.get("source") == "Indodax" else {}
        render_orderbook_panel(orderbook)
    elif section == "News & Community":
        news_rows = score_news_rows(fetch_crypto_news(news_query_for_row(sel)), sel)
        community = summarize_community(sel)
        render_news_community_panel(sel, news_rows, community)
    elif section == "AI":
        if not openrouter_key:
            st.warning("Masukkan OpenRouter API Key di sidebar untuk AI.")
        elif st.button("Generate Analisis AI", width="stretch"):
            with st.spinner("OpenRouter menganalisis crypto..."):
                chart_ctx = build_chart_context(sel) if sel.get("source") == "Indodax" else {}
                news_rows = score_news_rows(fetch_crypto_news(news_query_for_row(sel)), sel)
                community = summarize_community(sel)
                orderbook = orderbook_from_row(sel) if sel.get("source") == "Indodax" else {}
                st.markdown(call_openrouter(
                    build_crypto_market_prompt(sel, chart_ctx=chart_ctx, news_rows=news_rows, community=community, orderbook=orderbook),
                    openrouter_key,
                    llm_model,
                    max_tokens=2400,
                ))
            st.caption("Output AI adalah alat bantu analisis, bukan rekomendasi investasi.")


def render_meme_coin_page():
    st.title("Meme Coin IDR")
    st.caption("Meme coin yang punya pair IDR di Indodax. Ini market Rupiah Indonesia, bukan pair DEX global.")
    with st.sidebar:
        st.header("Filter Meme Coin IDR")
        min_volume = st.number_input("Min volume 24h IDR", min_value=0, value=1_000_000, step=1_000_000, key="meme_idr_min_volume")
        min_score = st.slider("Min score", 0, 100, 0, key="meme_idr_min_score")
        top_n = st.number_input("Top N", min_value=5, max_value=100, value=30, step=5, key="meme_idr_top_n")
        search = st.text_input("Cari meme coin", placeholder="DOGE, SHIB, PEPE, BONK", key="meme_idr_search")
        openrouter_key, llm_model = render_openrouter_controls("meme_coin_idr")

    try:
        with st.spinner("Mengambil meme coin IDR dari Indodax..."):
            df, source_url, active_source = fetch_crypto_market_df(source="Indodax IDR", quote="IDR")
    except Exception as exc:
        st.error(f"Data meme coin IDR gagal dimuat: {exc}")
        st.stop()

    view = df[df["is_meme"].fillna(False)].copy()
    view = view[view["quote_volume"].fillna(0) >= float(min_volume)]
    view = view[view["crypto_score"].fillna(0) >= float(min_score)]
    if search:
        q = search.upper().strip()
        view = view[
            view["symbol"].str.contains(q, na=False) |
            view["base"].str.contains(q, na=False) |
            view["asset_name"].astype(str).str.upper().str.contains(q, na=False)
        ]
    view = view.sort_values("crypto_score", ascending=False).head(int(top_n))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Meme Pairs", f"{len(view)}")
    m2.metric("Source", active_source)
    m3.metric("Top Volume", format_idr(view["quote_volume"].max() if not view.empty else 0))
    m4.metric("Avg 24h", pct_text(view["change_24h_pct"].mean() if not view.empty else np.nan))
    st.caption(f"Endpoint aktif: {source_url}. Daftar ini berbasis pair IDR yang tersedia di Indodax.")

    if view.empty:
        st.warning("Tidak ada meme coin IDR setelah filter. Turunkan min volume/score atau kosongkan pencarian.")
        st.stop()

    st.subheader("Meme Coin IDR List")
    table = view[[
        "symbol", "asset_name", "last_price", "change_24h_pct", "change_7d_pct",
        "quote_volume", "market_status", "crypto_score", "signal",
    ]].rename(columns={
        "last_price": "price",
        "change_24h_pct": "chg_24h%",
        "change_7d_pct": "chg_7d%",
        "quote_volume": "volume_24h",
        "market_status": "status",
        "crypto_score": "score",
    })
    table["price"] = view.apply(lambda row: format_market_price(row), axis=1).values
    table["volume_24h"] = view.apply(lambda row: format_market_amount(row), axis=1).values
    render_df_with_style_fallback(table, ["score"])
    st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), "meme_coin_idr.csv", "text/csv")

    st.markdown("---")
    selected_symbol = st.selectbox("Pilih meme coin", view["symbol"].tolist(), key="meme_idr_selected")
    sel = view[view["symbol"] == selected_symbol].iloc[0]
    st.subheader(f"{sel['symbol']} - {sel['signal']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", format_market_price(sel))
    c2.metric("24h", pct_text(sel.get("change_24h_pct")))
    c3.metric("7d", pct_text(sel.get("change_7d_pct")))
    c4.metric("Volume", format_market_amount(sel))
    c5.metric("Score", f"{safe_num(sel.get('crypto_score'), 0):.1f}")

    cex_item = watch_item_from_cex_row(sel)
    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        if is_watchlisted(cex_item["id"]):
            if st.button("Hapus Watchlist", width="stretch", key=f"meme_rm_{cex_item['id']}"):
                remove_watchlist_item(cex_item["id"])
                st.rerun()
        elif st.button("Tambah Watchlist", width="stretch", key=f"meme_add_{cex_item['id']}"):
            add_watchlist_item(cex_item)
            st.success("Masuk watchlist.")
    with a2:
        if st.button("Simpan Snapshot", width="stretch", key=f"meme_snap_{cex_item['id']}"):
            append_history_rows([history_row_from_cex(sel, cex_item["id"])])
            st.success("Snapshot tersimpan ke history.")
    with a3:
        st.caption("Meme coin tetap spekulatif walaupun tersedia di pair IDR. Pakai ukuran posisi kecil.")

    detail_options = ["Ringkasan", "Outlook", "Chart", "Liquidity", "News & Community", "AI"]
    section = st.radio(
        "Detail Meme Coin",
        detail_options,
        horizontal=True,
        label_visibility="collapsed",
        key=f"meme_detail_{selected_symbol}",
    )
    if section == "Ringkasan":
        if sel.get("logo_url"):
            st.image(sel.get("logo_url"), width=72)
        st.dataframe(pd.DataFrame({
            "Metric": ["Exchange", "Status", "Buy", "Sell", "High 24h", "Low 24h", "Min Trade IDR", "Base Volume"],
            "Value": [
                sel.get("exchange"),
                sel.get("market_status"),
                format_market_price(sel, "buy_price"),
                format_market_price(sel, "sell_price"),
                format_market_price(sel, "high_24h"),
                format_market_price(sel, "low_24h"),
                format_idr(sel.get("trade_min_idr")),
                format_compact(sel.get("base_volume")),
            ],
        }), width="stretch")
    elif section == "Outlook":
        meme_outlook = build_crypto_forward_analysis(sel)
        o1, o2, o3 = st.columns(3)
        o1.metric("Forward Score", f"{meme_outlook['forward_score']:.1f}/100")
        o2.metric("Verdict", meme_outlook["verdict"])
        o3.metric("Signal", str(sel.get("signal", "N/A")))
        st.markdown("**Bacaan Detail**")
        st.dataframe(meme_outlook["metrics"], width="stretch")
        st.markdown("**Analisis Ke Depan**")
        st.dataframe(meme_outlook["outlook"], width="stretch")
        st.markdown("**Scenario Map**")
        st.dataframe(meme_outlook["scenarios"], width="stretch")
        st.markdown("**Checklist Praktis**")
        st.markdown("\n".join(f"- {item}" for item in meme_outlook["checklist"]))
    elif section == "Chart":
        chart_ctx = build_chart_context(sel)
        render_tradingview_advanced_chart(f"INDODAX:{selected_symbol}", interval="60", height=580)
        render_chart_analysis_panel(sel, chart_ctx)
    elif section == "Liquidity":
        orderbook = orderbook_from_row(sel)
        render_orderbook_panel(orderbook)
    elif section == "News & Community":
        news_rows = score_news_rows(fetch_crypto_news(news_query_for_row(sel)), sel)
        community = summarize_community(sel)
        render_news_community_panel(sel, news_rows, community)
    elif section == "AI":
        if not openrouter_key:
            st.warning("Masukkan OpenRouter API Key di sidebar untuk AI.")
        elif st.button("Generate Analisis AI", width="stretch", key=f"meme_ai_{selected_symbol}"):
            with st.spinner("OpenRouter menganalisis meme coin IDR..."):
                chart_ctx = build_chart_context(sel)
                news_rows = score_news_rows(fetch_crypto_news(news_query_for_row(sel)), sel)
                community = summarize_community(sel)
                orderbook = orderbook_from_row(sel)
                st.markdown(call_openrouter(
                    build_meme_idr_prompt(sel, chart_ctx=chart_ctx, news_rows=news_rows, community=community, orderbook=orderbook),
                    openrouter_key,
                    llm_model,
                    max_tokens=2400,
                ))
            st.caption("Output AI adalah alat bantu screening, bukan rekomendasi investasi.")
    return

    st.title("Meme Coin Radar")
    st.caption("Data gratis dari DEX Screener. Cocok untuk screening awal pair DEX, bukan validasi final kontrak.")
    with st.sidebar:
        st.header("Filter Meme Coin")
        source = st.selectbox("Source", DEX_SOURCE_OPTIONS, index=0)
        query = st.text_input("Query/token", value="meme", placeholder="meme, pepe, dog, address")
        chain_label = st.selectbox("Chain", list(DEX_CHAIN_OPTIONS.keys()), index=0)
        min_liquidity = st.number_input("Min liquidity USD", min_value=0, value=5_000, step=5_000)
        max_age_hours = st.number_input("Max age jam", min_value=1, max_value=24 * 365, value=24 * 14, step=24)
        max_risk = st.slider("Max risk score", 0, 100, 80)
        top_n = st.number_input("Top N", min_value=5, max_value=200, value=50, step=5, key="meme_top_n")
        openrouter_key, llm_model = render_openrouter_controls("meme_coin")

    if source == "Search" and not query.strip():
        st.warning("Isi query dulu, misalnya meme, pepe, dog, atau alamat token.")
        st.stop()

    try:
        with st.spinner("Mengambil data DEX Screener..."):
            pairs = fetch_meme_source_pairs(source, query.strip())
        df = normalize_dex_pairs(pairs)
    except Exception as exc:
        st.error(f"Data DEX gagal dimuat: {exc}")
        st.stop()

    if df.empty:
        st.warning("Tidak ada pair dari DEX Screener untuk query ini.")
        st.stop()

    chain = DEX_CHAIN_OPTIONS[chain_label]
    view = df.copy()
    if chain != "all":
        view = view[view["chain"] == chain]
    view = view[view["liquidity_usd"].fillna(0) >= float(min_liquidity)]
    view = view[view["risk_score"].fillna(100) <= int(max_risk)]
    view = view[(view["age_minutes"].isna()) | (view["age_minutes"] <= float(max_age_hours) * 60)]
    view = view.sort_values("radar_score", ascending=False).head(int(top_n))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Pairs", f"{len(view)}")
    r2.metric("Top Liquidity", format_compact(view["liquidity_usd"].max() if not view.empty else 0, "$"))
    r3.metric("Top Radar", f"{safe_num(view['radar_score'].max() if not view.empty else 0, 0):.1f}")
    r4.metric("Avg Risk", f"{safe_num(view['risk_score'].mean() if not view.empty else 0, 0):.1f}")
    st.caption(f"Source aktif: {source}. Latest/boosted adalah proxy untuk discovery token baru/trending dari DEX Screener.")

    if view.empty:
        st.warning("Tidak ada pair setelah filter. Turunkan min liquidity, max risk, atau ganti query.")
        st.stop()

    st.subheader("Radar List")
    table = view[[
        "pair", "chain", "dex", "price_usd", "liquidity_usd", "volume_h1",
        "txns_h1", "change_m5_pct", "change_h1_pct", "change_h24_pct",
        "buy_ratio_h1", "fdv_liq_ratio", "volume_liq_h1",
        "age_minutes", "radar_score", "risk_score", "risk_label", "link_summary",
    ]].copy()
    table["age"] = table["age_minutes"].apply(format_age)
    table["buy_ratio_h1"] = table["buy_ratio_h1"].apply(lambda x: f"{safe_num(x, 0) * 100:.1f}%")
    table["fdv_liq_ratio"] = table["fdv_liq_ratio"].apply(ratio_text)
    table["volume_liq_h1"] = table["volume_liq_h1"].apply(ratio_text)
    table = table.drop(columns=["age_minutes"]).rename(columns={
        "price_usd": "price",
        "liquidity_usd": "liquidity",
        "volume_h1": "vol_1h",
        "txns_h1": "tx_1h",
        "change_m5_pct": "chg_5m%",
        "change_h1_pct": "chg_1h%",
        "change_h24_pct": "chg_24h%",
        "fdv_liq_ratio": "fdv/liq",
        "volume_liq_h1": "vol/liq_1h",
        "radar_score": "radar",
        "risk_score": "risk",
        "risk_label": "risk_label",
        "link_summary": "links",
    })
    render_df_with_style_fallback(table, ["radar"])
    st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), "meme_coin_radar.csv", "text/csv")

    st.markdown("---")
    view = view.copy()
    view["select_label"] = (
        view["pair"].astype(str) + " - " +
        view["chain"].astype(str) + " - " +
        view["dex"].astype(str) + " - " +
        view["pair_address"].astype(str).str[:8]
    )
    selected_label = st.selectbox("Pilih pair", view["select_label"].tolist())
    sel = view[view["select_label"] == selected_label].iloc[0]
    st.subheader(f"{sel['pair']} di {sel['chain']} / {sel['dex']}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"{safe_num(sel.get('price_usd'), 0):,.8g}")
    m2.metric("Liquidity", format_compact(sel.get("liquidity_usd"), "$"))
    m3.metric("Vol 1h", format_compact(sel.get("volume_h1"), "$"))
    m4.metric("Radar", f"{safe_num(sel.get('radar_score'), 0):.1f}")
    m5.metric("Risk", f"{safe_num(sel.get('risk_score'), 0):.0f} - {sel.get('risk_label')}")

    dex_item = watch_item_from_dex_row(sel)
    cta1, cta2, cta3, cta4 = st.columns([1, 1, 1, 2])
    with cta1:
        if is_watchlisted(dex_item["id"]):
            if st.button("Hapus Watchlist", width="stretch", key=f"rm_{dex_item['id']}"):
                remove_watchlist_item(dex_item["id"])
                st.rerun()
        elif st.button("Tambah Watchlist", width="stretch", key=f"add_{dex_item['id']}"):
            add_watchlist_item(dex_item)
            st.success("Masuk watchlist.")
    with cta2:
        if st.button("Simpan Snapshot", width="stretch", key=f"snap_{dex_item['id']}"):
            append_history_rows([history_row_from_dex(sel, dex_item["id"])])
            st.success("Snapshot tersimpan ke history.")
    with cta3:
        if sel.get("url"):
            st.link_button("DEX Screener", sel.get("url"), width="stretch")
    with cta4:
        st.caption("Security check dipanggil untuk pair terpilih supaya rate limit API gratis tetap aman.")

    meme_outlook = build_meme_forward_analysis(sel)
    tab_overview, tab_outlook, tab_flow, tab_liquidity, tab_links, tab_addresses, tab_risk, tab_security, tab_history, tab_ai = st.tabs([
        "Overview", "Outlook", "Flow", "Liquidity", "Links", "Addresses", "Risk", "Security", "History", "AI"
    ])
    with tab_overview:
        info_col, metric_col = st.columns([1, 2])
        with info_col:
            image_url = str(sel.get("info_image_url", "") or sel.get("info_open_graph", "") or "").strip()
            if image_url:
                st.image(image_url, width="stretch")
            st.markdown(f"**{sel.get('base_name') or sel.get('base_symbol')}**")
            st.caption(f"{sel.get('pair')} di {sel.get('dex')} / {sel.get('chain')}")
            if sel.get("labels"):
                st.info(f"Label: {sel.get('labels')}")
        with metric_col:
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Age", format_age(sel.get("age_minutes")))
            o2.metric("FDV", format_compact(sel.get("fdv"), "$"))
            o3.metric("Market Cap", format_compact(sel.get("market_cap"), "$"))
            o4.metric("Boosts", f"{safe_num(sel.get('boosts_active'), 0):.0f}")
            o5, o6, o7, o8 = st.columns(4)
            o5.metric("FDV/Liq", ratio_text(sel.get("fdv_liq_ratio")))
            o6.metric("Vol/Liq 1h", ratio_text(sel.get("volume_liq_h1")))
            o7.metric("Tx 1h", format_compact(sel.get("txns_h1")))
            o8.metric("Buy Ratio 1h", f"{safe_num(sel.get('buy_ratio_h1'), 0.5) * 100:.1f}%")
        st.markdown("**Score Breakdown**")
        score_df = pd.DataFrame({
            "Faktor": ["Radar", "Risk", "Momentum", "Volume", "Liquidity", "Buy Pressure", "Age"],
            "Skor": [
                safe_num(sel.get("radar_score"), np.nan),
                safe_num(sel.get("risk_score"), np.nan),
                safe_num(sel.get("momentum_score"), np.nan),
                safe_num(sel.get("volume_score"), np.nan),
                safe_num(sel.get("liquidity_score"), np.nan),
                safe_num(sel.get("buy_score"), np.nan),
                safe_num(sel.get("age_score"), np.nan),
            ],
        })
        render_df_with_style_fallback(score_df, ["Skor"])
    with tab_outlook:
        o1, o2, o3 = st.columns(3)
        o1.metric("Forward Quality", f"{meme_outlook['quality_score']:.1f}/100")
        o2.metric("Verdict", meme_outlook["verdict"])
        o3.metric("Risk", f"{safe_num(sel.get('risk_score'), 0):.0f} - {sel.get('risk_label')}")
        st.markdown("**Bacaan Detail**")
        st.dataframe(meme_outlook["metrics"], width="stretch")
        st.markdown("**Analisis Ke Depan**")
        st.dataframe(meme_outlook["outlook"], width="stretch")
        st.markdown("**Scenario Map**")
        st.dataframe(meme_outlook["scenarios"], width="stretch")
        st.markdown("**Checklist Praktis**")
        st.markdown("\n".join(f"- {item}" for item in meme_outlook["checklist"]))
    with tab_flow:
        flow_rows = []
        for window, vol_key, chg_key, buys_key, sells_key, tx_key, ratio_key in [
            ("5m", "volume_m5", "change_m5_pct", "buys_m5", "sells_m5", "txns_m5", "buy_ratio_m5"),
            ("1h", "volume_h1", "change_h1_pct", "buys_h1", "sells_h1", "txns_h1", "buy_ratio_h1"),
            ("6h", "volume_h6", "change_h6_pct", "buys_h6", "sells_h6", "txns_h6", "buy_ratio_h6"),
            ("24h", "volume_h24", "change_h24_pct", "buys_h24", "sells_h24", "txns_h24", "buy_ratio_h24"),
        ]:
            buys = safe_num(sel.get(buys_key), 0)
            sells = safe_num(sel.get(sells_key), 0)
            ratio = safe_num(sel.get(ratio_key), np.nan)
            flow_rows.append({
                "Window": window,
                "Volume": format_compact(sel.get(vol_key), "$"),
                "Change": pct_text(sel.get(chg_key)),
                "Buys": f"{buys:.0f}",
                "Sells": f"{sells:.0f}",
                "Txns": f"{safe_num(sel.get(tx_key), 0):.0f}",
                "Buy Ratio": f"{ratio * 100:.1f}%" if not pd.isna(ratio) else "N/A",
                "Net Buy": f"{buys - sells:+.0f}",
            })
        st.dataframe(pd.DataFrame(flow_rows), width="stretch")
        f1, f2, f3 = st.columns(3)
        f1.metric("M5 Buy Ratio", f"{safe_num(sel.get('buy_ratio_m5'), 0.5) * 100:.1f}%")
        f2.metric("H1 Buy Ratio", f"{safe_num(sel.get('buy_ratio_h1'), 0.5) * 100:.1f}%")
        f3.metric("H24 Buy Ratio", f"{safe_num(sel.get('buy_ratio_h24'), 0.5) * 100:.1f}%")
    with tab_liquidity:
        liquidity_rows = pd.DataFrame({
            "Metric": [
                "Liquidity USD", "Liquidity Base", "Liquidity Quote", "FDV", "Market Cap",
                "FDV/Liquidity", "MarketCap/Liquidity", "Volume/Liquidity 5m",
                "Volume/Liquidity 1h", "Volume/Liquidity 24h",
            ],
            "Value": [
                format_compact(sel.get("liquidity_usd"), "$"),
                format_compact(sel.get("liquidity_base")),
                format_compact(sel.get("liquidity_quote")),
                format_compact(sel.get("fdv"), "$"),
                format_compact(sel.get("market_cap"), "$"),
                ratio_text(sel.get("fdv_liq_ratio")),
                ratio_text(sel.get("mcap_liq_ratio")),
                ratio_text(sel.get("volume_liq_m5")),
                ratio_text(sel.get("volume_liq_h1")),
                ratio_text(sel.get("volume_liq_h24")),
            ],
        })
        st.dataframe(liquidity_rows, width="stretch")
        if safe_num(sel.get("liquidity_usd"), 0) < 25_000:
            st.warning("Likuiditas masih tipis. Slippage dan exit risk bisa besar.")
        if safe_num(sel.get("fdv_liq_ratio"), 0) > 150:
            st.warning("FDV jauh lebih besar dari likuiditas. Candle mudah dimanipulasi.")
    with tab_links:
        links = extract_dex_links(sel.get("info_json"))
        if links:
            for idx, item in enumerate(links):
                st.link_button(item["label"], item["url"], width="stretch")
        else:
            st.info("DEX Screener belum menyediakan website/social untuk pair ini.")
        if sel.get("url"):
            st.link_button("DEX Screener", sel.get("url"), width="stretch")
    with tab_addresses:
        addr_rows = pd.DataFrame({
            "Jenis": ["Base Token", "Quote Token", "Pair/Pool"],
            "Symbol": [sel.get("base_symbol"), sel.get("quote_symbol"), sel.get("pair")],
            "Address": [sel.get("base_address"), sel.get("quote_address"), sel.get("pair_address")],
            "Short": [
                compact_address(sel.get("base_address")),
                compact_address(sel.get("quote_address")),
                compact_address(sel.get("pair_address")),
            ],
        })
        st.dataframe(addr_rows, width="stretch")
        st.caption("Gunakan address lengkap ini untuk cek explorer, holder, liquidity lock, dan security scanner eksternal.")
    with tab_risk:
        st.dataframe(get_pair_risk_breakdown(sel), width="stretch")
        st.markdown("**Flags**")
        for flag in get_pair_flags(sel):
            st.warning(flag)
        st.caption("Risk score ini hanya dari data market DEX. Untuk meme coin, tetap cek contract, holder, tax, honeypot, dan liquidity lock secara manual.")
    with tab_security:
        if st.button("Run Security Check", width="stretch", key=f"sec_{dex_item['id']}"):
            with st.spinner("Menjalankan security API gratis..."):
                security = run_security_checks(sel)
            s1, s2, s3 = st.columns(3)
            s1.metric("Security Score", f"{security['score']}/100")
            s2.metric("Label", security["label"])
            s3.metric("Provider", ", ".join(security["statuses"]) if security["statuses"] else "N/A")
            if security["flags"]:
                for level, text in security["flags"]:
                    if str(level).lower() in {"high", "danger", "critical"}:
                        st.error(text)
                    else:
                        st.warning(text)
            else:
                st.success("Tidak ada red flag besar dari provider security yang tersedia.")
            if security["rows"]:
                st.dataframe(pd.DataFrame(security["rows"]), width="stretch")
            with st.expander("Raw provider status"):
                st.json({k: {"available": v.get("available"), "error": v.get("error"), "endpoint": v.get("endpoint")} for k, v in security["reports"].items()})
        else:
            st.info("Klik tombol untuk cek GoPlus/Honeypot.is di EVM atau RugCheck di Solana.")
    with tab_history:
        history = pd.DataFrame(load_history())
        item_history = history[history["id"] == dex_item["id"]].copy() if not history.empty and "id" in history.columns else pd.DataFrame()
        if item_history.empty:
            st.info("Belum ada history untuk pair ini. Klik Simpan Snapshot atau refresh dari Watchlist & Alerts.")
        else:
            item_history["ts"] = pd.to_datetime(item_history["ts"], errors="coerce")
            st.line_chart(item_history.set_index("ts")[["score", "risk", "change"]])
            st.dataframe(item_history.sort_values("ts", ascending=False), width="stretch")
    with tab_ai:
        if not openrouter_key:
            st.warning("Masukkan OpenRouter API Key di sidebar untuk AI.")
        else:
            include_security = st.checkbox("Sertakan security check", value=True)
        if openrouter_key and st.button("Generate AI Risk Summary", width="stretch"):
            security = None
            if include_security:
                with st.spinner("Menjalankan security check sebelum AI..."):
                    security = run_security_checks(sel)
            with st.spinner("OpenRouter menganalisis meme coin..."):
                st.markdown(call_openrouter(build_meme_prompt(sel, security=security), openrouter_key, llm_model, max_tokens=2000))
            st.caption("Output AI adalah alat bantu screening, bukan rekomendasi investasi.")


def render_watchlist_page():
    st.title("Watchlist & Alerts")
    st.caption("Watchlist saham IDX dan crypto IDR, alert sederhana, dan history score disimpan lokal di folder data/.")
    items = [
        item for item in load_watchlist()
        if item.get("type") == "stock" or (
            item.get("type") == "cex" and str(item.get("symbol", "")).upper().endswith("IDR")
        )
    ]

    with st.sidebar:
        st.header("Alert Rules")
        rules = load_alert_rules()
        with st.form("alert_rules_form"):
            st.markdown("**Saham IDX**")
            rules["stock_min_score"] = st.number_input("Saham min score", 0.0, 100.0, float(rules["stock_min_score"]), 1.0)
            rules["stock_min_change"] = st.number_input("Saham naik harian >=", -100.0, 1000.0, float(rules["stock_min_change"]), 1.0)
            rules["stock_dump_change"] = st.number_input("Saham turun harian <=", -100.0, 0.0, float(rules["stock_dump_change"]), 1.0)
            rules["stock_min_rel_volume"] = st.number_input("Saham rel volume >=", 0.0, 20.0, float(rules["stock_min_rel_volume"]), 0.1)
            st.markdown("**Crypto IDR**")
            rules["cex_min_score"] = st.number_input("IDR min score", 0.0, 100.0, float(rules["cex_min_score"]), 1.0)
            rules["cex_min_24h_change"] = st.number_input("IDR pump 24h >=", -100.0, 1000.0, float(rules["cex_min_24h_change"]), 1.0)
            rules["cex_dump_24h_change"] = st.number_input("IDR dump 24h <=", -100.0, 0.0, float(rules["cex_dump_24h_change"]), 1.0)
            if st.form_submit_button("Simpan Rules", width="stretch"):
                save_alert_rules(rules)
                st.success("Alert rules tersimpan.")

    render_deploy_readiness_panel()

    with st.expander("Backup / Restore Data Lokal", expanded=False):
        backup_payload = build_local_backup_payload()
        backup_bytes = json.dumps(backup_payload, ensure_ascii=False, indent=2).encode("utf-8")
        b1, b2 = st.columns([1, 2])
        with b1:
            st.download_button(
                "Export Backup JSON",
                backup_bytes,
                file_name=f"market_screener_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch",
            )
        with b2:
            uploaded_backup = st.file_uploader("Restore backup JSON", type=["json"], key="restore_backup_json")
            merge_restore = st.checkbox("Merge dengan data sekarang", value=True)
            if uploaded_backup is not None and st.button("Restore Backup", width="stretch"):
                try:
                    payload = json.loads(uploaded_backup.read().decode("utf-8"))
                    restore_local_backup_payload(payload, merge=merge_restore)
                    st.success("Backup berhasil direstore.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Restore gagal: {exc}")
        st.caption("Di Streamlit Cloud, file lokal bisa hilang saat app restart/redeploy. Export backup sebelum deploy ulang kalau watchlist/history penting.")

    if not items:
        st.info("Watchlist masih kosong. Tambahkan saham dari Saham BEI Auto Scanner atau coin IDR dari Crypto Market/Meme Coin Radar.")
        return

    st.subheader("Daftar Watchlist")
    watch_df = pd.DataFrame(items)
    show_cols = [c for c in ["type", "label", "company", "sector", "source", "symbol", "quote", "added_at"] if c in watch_df.columns]
    st.dataframe(watch_df[show_cols], width="stretch")

    a1, a2 = st.columns([1, 2])
    with a1:
        if st.button("Refresh Watchlist", width="stretch"):
            with st.spinner("Mengambil snapshot terbaru saham dan crypto..."):
                snapshot, alerts, saved_count = refresh_watchlist_snapshot(run_security=False)
            st.session_state["watchlist_snapshot"] = snapshot
            st.session_state["watchlist_alerts"] = alerts
            st.success(f"{saved_count} snapshot tersimpan ke history.")
    with a2:
        remove_options = {
            f"{item.get('label', item.get('id'))} ({item.get('type', '-')})": item.get("id")
            for item in items
        }
        selected_remove = st.selectbox("Hapus item", ["-"] + list(remove_options.keys()))
        if selected_remove != "-" and st.button("Hapus dari Watchlist", width="stretch"):
            remove_watchlist_item(remove_options[selected_remove])
            st.rerun()

    snapshot = st.session_state.get("watchlist_snapshot")
    alerts = st.session_state.get("watchlist_alerts")
    if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
        st.subheader("Snapshot Terbaru")
        display = snapshot.copy()
        for col in ["price", "score", "risk", "security", "volume", "value_traded", "change", "rel_volume"]:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce")
        st.dataframe(display, width="stretch")
    else:
        st.info("Klik Refresh Watchlist untuk menarik data terbaru dan menyimpan history.")

    st.subheader("Alert Aktif")
    if isinstance(alerts, pd.DataFrame) and not alerts.empty:
        st.dataframe(alerts, width="stretch")
    else:
        st.caption("Belum ada alert aktif dari snapshot terakhir.")

    st.subheader("History")
    history = pd.DataFrame(load_history())
    if history.empty:
        st.info("History masih kosong.")
        return
    history["ts"] = pd.to_datetime(history["ts"], errors="coerce")
    id_to_label = {item.get("id"): item.get("label", item.get("id")) for item in items}
    labels = [id_to_label.get(item_id, item_id) for item_id in history["id"].dropna().unique()]
    reverse = {id_to_label.get(item_id, item_id): item_id for item_id in history["id"].dropna().unique()}
    selected_label = st.selectbox("Asset history", labels)
    selected_id = reverse[selected_label]
    h = history[history["id"] == selected_id].sort_values("ts").copy()
    metric_cols = [c for c in ["score", "risk", "change", "rel_volume"] if c in h.columns]
    if metric_cols:
        st.line_chart(h.set_index("ts")[metric_cols])
    st.dataframe(h.sort_values("ts", ascending=False), width="stretch")

# ── SCORING ──
@st.cache_data(show_spinner=False)
def load_table(uploaded_file):
    name = uploaded_file.name.lower()
    bio  = io.BytesIO(uploaded_file.read())
    df   = pd.read_excel(bio) if name.endswith((".xlsx",".xls")) else pd.read_csv(bio)
    return normalize_columns(df)

def compute_scores(saham_df, broker_df, trade_df, daftar_df, weights=None):
    if weights is None: weights = DEFAULT_WEIGHTS
    d = saham_df.copy()
    d["kode saham"]      = d[find_col(d,"kode saham")].astype(str).str.upper().str.strip()
    d["nama perusahaan"] = d[find_col(d,"nama perusahaan")].astype(str).str.strip()
    for col,alias in [("penutupan","penutupan_num"),("open price","open_num"),("tertinggi","high_num"),
                      ("terendah","low_num"),("sebelumnya","prev_num"),("selisih","selisih_num"),
                      ("nilai","nilai_num"),("volume","volume_num"),("frekuensi","frekuensi_num"),
                      ("foreign buy","foreign_buy_num"),("foreign sell","foreign_sell_num"),
                      ("bid volume","bid_vol_num"),("offer volume","offer_vol_num")]:
        d[alias] = safe_col(d, col)
    d["change_pct"]         = np.where(d["prev_num"]!=0,(d["selisih_num"]/d["prev_num"])*100,np.nan)
    d["foreign_net"]        = d["foreign_buy_num"]-d["foreign_sell_num"]
    d["foreign_net_ratio"]  = np.where(d["nilai_num"]!=0,d["foreign_net"]/d["nilai_num"],0)
    d["bid_offer_pressure"] = np.where(
        (d["bid_vol_num"]+d["offer_vol_num"])!=0,
        (d["bid_vol_num"]-d["offer_vol_num"])/(d["bid_vol_num"]+d["offer_vol_num"]),0)
    d["true_range"] = np.maximum(d["high_num"]-d["low_num"],
                      np.maximum(abs(d["high_num"]-d["prev_num"]),abs(d["low_num"]-d["prev_num"])))
    d["atr_pct"]    = np.where(d["prev_num"]!=0,d["true_range"]/d["prev_num"]*100,np.nan)
    d["close_position"] = np.where(
        (d["high_num"]-d["low_num"])!=0,
        (d["penutupan_num"]-d["low_num"])/(d["high_num"]-d["low_num"]),0.5)
    m = daftar_df.copy()
    if has_columns(m, REQUIRED_COLUMNS["daftar_saham"]):
        m["kode"]             = m[find_col(m,"kode")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m,"papan pencatatan")].astype(str).str.strip()
        m["tanggal pencatatan"]=m[find_col(m,"tanggal pencatatan")]
    elif has_columns(m, ALTERNATE_COLUMNS["daftar_saham"]):
        m["kode"]             = m[find_col(m,"id instrument")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m,"id board")].astype(str).str.strip()
        m["tanggal pencatatan"]=pd.NaT
        m = m.groupby("kode",dropna=False).agg(
            {"papan pencatatan":lambda x:",".join(sorted(set(v for v in x if pd.notna(v)))),"tanggal pencatatan":"first"}).reset_index()
    else:
        m = pd.DataFrame({"kode":d["kode saham"],"papan pencatatan":pd.NA,"tanggal pencatatan":pd.NaT})
    d = d.merge(m[["kode","papan pencatatan","tanggal pencatatan"]],left_on="kode saham",right_on="kode",how="left")
    t = trade_df.copy()
    t["id instrument"]   = t[find_col(t,"id instrument")].astype(str).str.upper().str.strip()
    t["trade_nilai"]     = safe_col(t,"nilai")
    t["trade_frekuensi"] = safe_col(t,"frekuensi")
    t_agg = t.groupby("id instrument",dropna=False)[["trade_nilai","trade_frekuensi"]].sum().reset_index().rename(columns={"id instrument":"kode saham"})
    d = d.merge(t_agg,on="kode saham",how="left")
    b = broker_df.copy()
    b["kode perusahaan"] = b[find_col(b,"kode perusahaan")].astype(str).str.upper().str.strip()
    b["broker_nilai"]    = safe_col(b,"nilai")
    b["broker_frekuensi"]= safe_col(b,"frekuensi")
    ov = len(set(d["kode saham"].dropna().astype(str).str.upper())&set(b["kode perusahaan"].dropna().astype(str).str.upper()))
    if ov/max(1,min(len(d),len(b)))>=0.2:
        b_agg=b.groupby("kode perusahaan",dropna=False)[["broker_nilai","broker_frekuensi"]].sum().reset_index().rename(columns={"kode perusahaan":"kode saham"})
        d=d.merge(b_agg,on="kode saham",how="left")
        d["broker_score"]=percentile_series(d["broker_nilai"].fillna(0))*0.6+percentile_series(d["broker_frekuensi"].fillna(0))*0.4
    else: d["broker_score"]=50.0
    w=weights
    d["momentum_score"]       =percentile_series(d["change_pct"])
    d["liquidity_score"]      =percentile_series(d["nilai_num"])*0.5+percentile_series(d["volume_num"])*0.25+percentile_series(d["frekuensi_num"])*0.25
    d["flow_score"]           =percentile_series(d["foreign_net_ratio"])*0.6+percentile_series(d["bid_offer_pressure"])*0.4
    d["market_activity_score"]=percentile_series(d["trade_nilai"].fillna(0))*0.6+percentile_series(d["trade_frekuensi"].fillna(0))*0.4
    d["vol_per_freq"]         =np.where(d["frekuensi_num"]!=0,d["volume_num"]/d["frekuensi_num"],np.nan)
    d["volume_trend_score"]   =percentile_series(d["vol_per_freq"])*0.5+percentile_series(d["bid_offer_pressure"])*0.5
    d["price_structure_score"]=percentile_series(d["close_position"])
    d["final_score"]=(d["momentum_score"]*w["momentum"]+d["liquidity_score"]*w["liquidity"]+
                      d["flow_score"]*w["flow"]+d["market_activity_score"]*w["market_activity"]+
                      d["volume_trend_score"]*w["volume_trend"]+d["price_structure_score"]*w["price_structure"]+
                      d["broker_score"]*w["broker"])
    d["kategori"]=pd.cut(d["final_score"],bins=[-np.inf,40,60,75,np.inf],labels=["Rendah","Menarik","Tinggi","Sangat Tinggi"])
    cols=["kode saham","nama perusahaan","papan pencatatan","tanggal pencatatan",
          "penutupan_num","open_num","high_num","low_num","prev_num",
          "change_pct","volume_num","nilai_num","frekuensi_num",
          "foreign_net","foreign_net_ratio","bid_offer_pressure",
          "atr_pct","close_position","final_score","kategori",
          "momentum_score","liquidity_score","flow_score",
          "market_activity_score","volume_trend_score","price_structure_score","broker_score"]
    return d[cols].sort_values("final_score",ascending=False).reset_index(drop=True)

def compute_multiday_signals(scored_days, day_labels):
    if len(scored_days)==1:
        df=scored_days[0].copy()
        df["trend_slope"]=0.0; df["score_consistency"]=0.0
        df["trend_score_norm"]=df["final_score"]; df["signal_strength"]=df["final_score"]
        df["signal_label"]=df["kategori"].astype(str); df["days_data"]=1
        df["score_day_labels"]=day_labels[0] if day_labels else "D1"
        return df
    all_s=[]
    for i,(dd,lbl) in enumerate(zip(scored_days,day_labels)):
        tmp=dd[["kode saham","final_score","foreign_net_ratio","nilai_num"]].copy()
        tmp["day_idx"]=i; all_s.append(tmp)
    panel=pd.concat(all_s,ignore_index=True)
    latest=scored_days[-1].copy()
    def calc_slope(g):
        sc=g.sort_values("day_idx")["final_score"].values
        if len(sc)<2: return pd.Series({"trend_slope":0.0,"score_consistency":0.0,"foreign_acc_mean":0.0,"vol_growth":0.0})
        slope=float(np.polyfit(np.arange(len(sc),dtype=float),sc,1)[0])
        cons=float(np.mean(sc>=60)*100)
        fv=g.sort_values("day_idx")["foreign_net_ratio"].values
        vv=g.sort_values("day_idx")["nilai_num"].values
        vg=float((vv[-1]-vv[0])/(vv[0]+1e-9)*100) if len(vv)>=2 else 0.0
        return pd.Series({"trend_slope":slope,"score_consistency":cons,"foreign_acc_mean":float(np.mean(fv)),"vol_growth":vg})
    trends=panel.groupby("kode saham").apply(calc_slope).reset_index()
    latest=latest.merge(trends,on="kode saham",how="left")
    latest["trend_score_norm"]=np.clip(50+latest["trend_slope"]*(50/5.0),0,100)
    latest["foreign_acc_norm"]=np.clip(50+latest["foreign_acc_mean"]*500,0,100)
    latest["vol_growth_norm"] =np.clip(50+latest["vol_growth"]/4,0,100)
    latest["signal_strength"] =(latest["final_score"]*0.35+latest["trend_score_norm"]*0.30+
                                 latest["score_consistency"]*0.20+latest["foreign_acc_norm"]*0.10+
                                 latest["vol_growth_norm"]*0.05)
    def sl(row):
        fs,ts,co=row["final_score"],row.get("trend_score_norm",50),row.get("score_consistency",0)
        if fs>=70 and ts>=60 and co>=60: return "\U0001f525 Breakout Candidate"
        if fs>=60 and ts>=60:            return "\U0001f4c8 Trending Up"
        if fs>=60 and ts<45:             return "\u26a0\ufe0f Fading Momentum"
        if fs<50  and ts>=60:            return "\U0001f440 Emerging"
        if fs>=75:                       return "\u2705 Sangat Tinggi"
        if fs>=60:                       return "\u2705 Tinggi"
        return "\u2b1c Watch Only"
    latest["signal_label"]=latest.apply(sl,axis=1)
    latest["days_data"]=len(scored_days)
    latest["score_day_labels"]=" -> ".join(day_labels)
    return latest.sort_values("signal_strength",ascending=False).reset_index(drop=True)

def compute_market_regime(index_df):
    d=index_df.copy()
    d["kode indeks"]=d[find_col(d,"kode indeks")].astype(str).str.upper().str.strip()
    d["sebelumnya_num"]=safe_col(d,"sebelumnya"); d["selisih_num"]=safe_col(d,"selisih")
    d["nilai_num"]=safe_col(d,"nilai"); d["frekuensi_num"]=safe_col(d,"frekuensi")
    pref=d[d["kode indeks"].str.contains("IHSG|COMPOSITE|JCI",regex=True,na=False)]
    row=pref.iloc[0] if not pref.empty else d.sort_values("nilai_num",ascending=False).iloc[0]
    prev=float(row["sebelumnya_num"]) if pd.notna(row["sebelumnya_num"]) else 0.0
    diff=float(row["selisih_num"])    if pd.notna(row["selisih_num"]) else 0.0
    chg=(diff/prev)*100 if prev!=0 else 0.0
    mom=float(np.clip(50+(chg*8),0,100))
    act=percentile_series(d["nilai_num"]).loc[row.name]*0.6+percentile_series(d["frekuensi_num"]).loc[row.name]*0.4
    rs=float(np.clip(mom*0.7+act*0.3,0,100))
    return {"index_code":str(row["kode indeks"]),"index_change_pct":float(chg),"regime_score":rs,
            "regime_label":"Risk-On" if rs>=60 else "Netral" if rs>=45 else "Risk-Off"}

# -- APP NAVIGATION --
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if st.session_state["page"] not in APP_PAGES:
    st.session_state["page"] = "Home"

with st.sidebar:
    st.header("Navigasi")
    nav_choice = st.selectbox(
        "Mode",
        APP_PAGES,
        index=APP_PAGES.index(st.session_state["page"]),
        key=f"nav_mode_{st.session_state['page'].replace(' ', '_')}",
    )
    if nav_choice != st.session_state["page"]:
        st.session_state["page"] = nav_choice
        st.rerun()
    st.markdown("---")

if st.session_state["page"] == "Home":
    render_home()
    st.stop()

if st.session_state["page"] == "Crypto Market":
    render_crypto_market_page()
    st.stop()

if st.session_state["page"] == "Meme Coin Radar":
    render_meme_coin_page()
    st.stop()

if st.session_state["page"] == "Watchlist & Alerts":
    render_watchlist_page()
    st.stop()


# ── SIDEBAR UI ──
st.title("BEI Screener v3 — Auto + Flow + AI")
st.caption("Default tanpa upload memakai TradingView scanner. Upload BEI tetap tersedia untuk foreign/broker flow.")

with st.sidebar:
    st.header("Setup")
    st.markdown("**OpenRouter API Key**")
    static_openrouter_key = get_static_openrouter_key()
    llm_model = get_openrouter_model()
    st.caption(f"Model aktif: {llm_model}")
    if static_openrouter_key:
        st.success("API key statis terdeteksi (secrets/env).")
        with st.expander("Override API Key (opsional)"):
            override_key = st.text_input("Override Key", type="password", placeholder="sk-or-v1-...", key="or_key_override")
        openrouter_key = override_key.strip() if override_key else static_openrouter_key
    else:
        st.caption("Ambil key di: https://openrouter.ai/keys")
        openrouter_key = st.text_input("API Key", type="password", placeholder="sk-or-v1-...", key="or_key")
    st.markdown("---")
    st.header("Sumber Data Saham")
    stock_data_mode = st.radio(
        "Mode",
        ["Auto TradingView", "Upload BEI Advanced"],
        index=0,
        key="stock_data_mode",
    )
    if stock_data_mode == "Upload BEI Advanced":
        st.markdown("---")
        st.header("Upload Data BEI")
        n_days=st.number_input("Jumlah hari",min_value=1,max_value=5,value=1,step=1)
        day_data=[]
        for i in range(int(n_days)):
            with st.expander(f"Hari {i+1}",expanded=(i==int(n_days)-1)):
                lbl=st.text_input("Label",value=f"D{i+1}",key=f"lbl_{i}")
                sf =st.file_uploader("Ringkasan Saham",      type=["xlsx","xls","csv"],key=f"s_{i}")
                bf =st.file_uploader("Ringkasan Broker",     type=["xlsx","xls","csv"],key=f"b_{i}")
                pf =st.file_uploader("Ringkasan Perdagangan",type=["xlsx","xls","csv"],key=f"p_{i}")
                df_=st.file_uploader("Daftar Saham",         type=["xlsx","xls","csv"],key=f"d_{i}")
                day_data.append({"label":lbl,"saham":sf,"broker":bf,"perdagangan":pf,"daftar":df_})
        st.markdown("---")
        indeks_file=st.file_uploader("Ringkasan Indeks (Opsional)",type=["xlsx","xls","csv"])
        st.markdown("---")
        with st.expander("Bobot Scoring"):
            wm=st.slider("Momentum",       0,100,25); wl=st.slider("Likuiditas",     0,100,20)
            wf=st.slider("Flow",           0,100,20); wa=st.slider("Market Activity",0,100,15)
            wv=st.slider("Volume Trend",   0,100,10); wp=st.slider("Price Structure",0,100, 5)
            wb=st.slider("Broker",         0,100, 5)
            tw=max(wm+wl+wf+wa+wv+wp+wb,1)
            custom_weights={"momentum":wm/tw,"liquidity":wl/tw,"flow":wf/tw,"market_activity":wa/tw,
                            "volume_trend":wv/tw,"price_structure":wp/tw,"broker":wb/tw}

if stock_data_mode == "Auto TradingView":
    render_stock_auto_page(openrouter_key, llm_model)
    st.stop()

# ── DATA PROCESSING ──
complete_days=[d for d in day_data if all([d["saham"],d["broker"],d["perdagangan"],d["daftar"]])]
if not complete_days:
    st.info("Upload minimal 1 set lengkap (4 file) untuk memulai.")
    st.stop()

scored_days,day_labels,market_regime=[],[],None
with st.spinner("Memproses data BEI..."):
    for day in complete_days:
        try:
            s_df=load_table(day["saham"]); b_df=load_table(day["broker"])
            p_df=load_table(day["perdagangan"]); d_df=load_table(day["daftar"])
            ok_all=True
            for key,df in [("ringkasan_saham",s_df),("ringkasan_broker",b_df),("ringkasan_perdagangan",p_df),("daftar_saham",d_df)]:
                ok,missing=validate_columns(df,REQUIRED_COLUMNS[key])
                if key in ALTERNATE_COLUMNS and not ok: ok=has_columns(df,ALTERNATE_COLUMNS[key])
                if not ok: ok_all=False; st.error(f"[{day['label']}] {key}: kurang {', '.join(missing)}")
            if ok_all:
                scored_days.append(compute_scores(s_df,b_df,p_df,d_df,weights=custom_weights))
                day_labels.append(day["label"])
        except Exception as e: st.error(f"[{day['label']}] Gagal: {e}")
    if indeks_file:
        try:
            idx_df=load_table(indeks_file)
            ok_i,miss_i=validate_columns(idx_df,INDEX_REQUIRED_COLUMNS)
            if ok_i: market_regime=compute_market_regime(idx_df)
            else: st.warning(f"Indeks kolom kurang: {', '.join(miss_i)}")
        except Exception as e: st.warning(f"Indeks error: {e}")

if not scored_days: st.error("Tidak ada data valid."); st.stop()

combined=compute_multiday_signals(scored_days,day_labels)
if market_regime:
    combined["final_score"]    =combined["final_score"]*0.90+market_regime["regime_score"]*0.10
    combined["signal_strength"]=combined["signal_strength"]*0.90+market_regime["regime_score"]*0.10

# ── MAIN UI ──
if market_regime:
    r1,r2,r3,r4=st.columns(4)
    r1.metric("Index",market_regime["index_code"]); r2.metric("IHSG",f"{market_regime['index_change_pct']:+.2f}%")
    r3.metric("Regime",market_regime["regime_label"]); r4.metric("Regime Score",f"{market_regime['regime_score']:.1f}")

n_loaded=len(scored_days)
if n_loaded > 1:
    st.success(f"{n_loaded} hari data: {' -> '.join(day_labels)}")
else:
    st.info("1 hari data.")

c1,c2,c3,c4=st.columns([2,1,1,1])
with c1: min_sig=st.slider("Min Signal Strength",0,100,60)
with c2:
    po=sorted([x for x in combined["papan pencatatan"].dropna().unique() if str(x).strip()])
    papan_f=st.selectbox("Papan",["Semua"]+po)
with c3: top_n=st.number_input("Top N",5,300,25,5)
with c4: sig_f=st.selectbox("Sinyal",["Semua","Breakout Candidate","Trending Up","Fading Momentum","Emerging"])

view=combined[combined["signal_strength"]>=min_sig].copy()
if papan_f!="Semua": view=view[view["papan pencatatan"]==papan_f]
if sig_f!="Semua":
    view=view[view["signal_label"].str.contains(sig_f,na=False)]
view=view.head(int(top_n))

st.subheader("Kandidat Saham")
dcols=["kode saham","nama perusahaan","papan pencatatan","penutupan_num","change_pct","foreign_net","final_score","signal_strength","signal_label"]
if n_loaded>1: dcols+=["trend_slope","score_consistency"]
disp=view[[c for c in dcols if c in view.columns]].rename(columns={
    "penutupan_num":"close","foreign_net":"net_foreign","signal_strength":"signal",
    "signal_label":"sinyal","trend_slope":"tren/hari","score_consistency":"konsistensi%"})
num_cols=[c for c in ["signal","final_score"] if c in disp.columns]
if num_cols: render_df_with_style_fallback(disp, num_cols)
else: st.dataframe(disp,width="stretch")
st.download_button("Export CSV",view.to_csv(index=False).encode("utf-8"),"bei_v3.csv","text/csv")

if view.empty: st.warning("Tidak ada kandidat."); st.stop()

st.markdown("---")
selected_code=st.selectbox("Pilih saham untuk analisis",options=view["kode saham"].tolist())
sel=combined[combined["kode saham"]==selected_code].iloc[0]
company=str(sel.get("nama perusahaan",""))
sig_lbl=str(sel.get("signal_label",""))

st.subheader(f"{selected_code}  {company}  {sig_lbl}")
m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Signal",f"{sel.get('signal_strength',sel['final_score']):.1f}")
m2.metric("Score",f"{sel['final_score']:.1f}")
m3.metric("Change %",f"{sel['change_pct']:+.2f}%")
m4.metric("Net Foreign",f"{sel['foreign_net']:,.0f}")
m5.metric("ATR-1d",f"{sel.get('atr_pct',0):.2f}%")

tab_score,tab_chart,tab_ai=st.tabs(["Score","Chart","AI Analisis"])

with tab_score:
    bd=pd.DataFrame({
        "Faktor":["Momentum","Likuiditas","Flow","Market Activity","Vol Trend","Price Structure","Broker"],
        "Skor":[safe_num(sel.get(c),np.nan) for c in
               ["momentum_score","liquidity_score","flow_score","market_activity_score",
                "volume_trend_score","price_structure_score","broker_score"]]})
    bd["Status"]=bd["Skor"].apply(factor_label)
    render_df_with_style_fallback(bd, ["Skor"])
    if n_loaded>1:
        x1,x2,x3=st.columns(3)
        x1.metric("Tren/Hari",f"{sel.get('trend_slope',0):+.1f} pts")
        x2.metric("Konsistensi",f"{sel.get('score_consistency',0):.0f}%")
        x3.metric("Hari Data",f"{sel.get('days_data',1)}")
    close=safe_num(sel.get("penutupan_num"),np.nan); atr_p=max(safe_num(sel.get("atr_pct"),2.0),0.5)
    if not pd.isna(close):
        atr_abs=close*atr_p/100
        st.markdown("**Level Eksekusi**")
        l1,l2,l3,l4,l5=st.columns(5)
        l1.metric("Entry Low",f"{close*0.995:.2f}"); l2.metric("Entry High",f"{close+atr_abs*0.3:.2f}")
        l3.metric("Stop",f"{close-atr_abs*1.5:.2f}"); l4.metric("TP1",f"{close+atr_abs*2.0:.2f}"); l5.metric("TP2",f"{close+atr_abs*3.5:.2f}")
        st.caption(f"R/R: {(atr_abs*2.0)/(atr_abs*1.5):.2f}x | ATR={atr_abs:.2f} ({atr_p:.2f}%)")

with tab_chart:
    tv_sym=f"IDX:{selected_code}"; tv_slug=f"IDX-{selected_code}"
    ct1,ct2,ct3,ct4=st.tabs(["Chart","Technicals","Financials","News"])
    with ct1:
        render_tradingview_advanced_chart(tv_sym, interval="D", height=580)
    with ct2:
        components.html(f"""<div class="tradingview-widget-container" style="width:100%;height:700px;">
          <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
          <script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
          {json.dumps({"interval":"1D","width":"100%","height":700,"symbol":tv_sym,"showIntervalTabs":True,"locale":"id","colorTheme":"dark"})}
          </script></div>""",height=720)
    with ct3:
        components.iframe(f"https://id.tradingview.com/symbols/{tv_slug}/financials-overview/",height=700,scrolling=True)
    with ct4:
        components.html(f"""<div class="tradingview-widget-container" style="width:100%;height:700px;">
          <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
          <script src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
          {json.dumps({"feedMode":"symbol","symbol":tv_sym,"displayMode":"regular","width":"100%","height":700,"colorTheme":"dark","locale":"id","isTransparent":False})}
          </script></div>""",height=720)

with tab_ai:
    if not openrouter_key:
        st.warning("Masukkan OpenRouter API Key di sidebar.")
        st.markdown("Buat key di: https://openrouter.ai/keys")
    else:
        tv_status=st.empty()
        tv_status.info(f"Mengambil data teknikal {selected_code} dari TradingView...")
        tv_raw=fetch_tv_data(selected_code)
        tv_parsed=parse_tv_data(tv_raw)
        if tv_parsed:
            tv_status.success("Data TradingView berhasil dimuat.")
            with st.expander("Data Teknikal TradingView",expanded=True):
                rec_all=tv_parsed.get("rec_all","N/A")
                ta1,ta2,ta3=st.columns(3)
                ta1.metric("Rekomendasi",f"{tv_rec_emoji(rec_all)} {rec_all}")
                ta2.metric("MA Signal",  f"{tv_rec_emoji(tv_parsed.get('rec_ma',''))} {tv_parsed.get('rec_ma','N/A')}")
                ta3.metric("Oscillator", f"{tv_rec_emoji(tv_parsed.get('rec_other',''))} {tv_parsed.get('rec_other','N/A')}")
                tb1,tb2,tb3,tb4=st.columns(4)
                tb1.metric("RSI(14)",f"{tv_parsed.get('rsi',0):.1f}")
                tb2.metric("ADX",    f"{tv_parsed.get('adx',0):.1f}")
                tb3.metric("Stoch K",f"{tv_parsed.get('stoch_k',0):.1f}")
                tb4.metric("Vol Rel",f"{tv_parsed.get('rel_volume',0):.2f}x")
                tc1,tc2,tc3=st.columns(3)
                ef=lambda x:"di atas" if x else "di bawah" if x is False else "-"
                tc1.metric("EMA20", f"{tv_parsed.get('ema20',0):.2f}",delta=ef(tv_parsed.get("above_ema20")))
                tc2.metric("EMA50", f"{tv_parsed.get('ema50',0):.2f}",delta=ef(tv_parsed.get("above_ema50")))
                tc3.metric("EMA200",f"{tv_parsed.get('ema200',0):.2f}",delta=ef(tv_parsed.get("above_ema200")))
                if tv_parsed.get("bb_position") is not None:
                    st.progress(min(int(tv_parsed["bb_position"]),100),
                                text=f"Bollinger Position: {tv_parsed['bb_position']:.1f}% (0=lower band, 100=upper band)")
        else:
            tv_status.warning("Data TradingView tidak tersedia. AI analisis dari data BEI saja.")
        md_ctx=None
        if n_loaded>1:
            md_ctx={"trend_slope":safe_num(sel.get("trend_slope"),0),
                    "score_consistency":safe_num(sel.get("score_consistency"),0),
                    "days_data":int(safe_num(sel.get("days_data"),1))}
        with st.spinner("OpenRouter menganalisis..."):
            prompt=build_prompt(selected_code,company,sel,tv_parsed,market_regime,md_ctx)
            narasi=call_openrouter(prompt,openrouter_key,llm_model)
        st.markdown("---")
        st.markdown(f"### Analisis AI — OpenRouter ({llm_model})")
        st.markdown(narasi)
        st.caption("Output AI berdasarkan data kuantitatif. Bukan rekomendasi investasi.")
