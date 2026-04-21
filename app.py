import io
import json
import os
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="BEI Screener v3", layout="wide")

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
        st.dataframe(df.style.applymap(color_signal, subset=subset_cols), use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

def factor_label(score):
    if pd.isna(score): return "N/A"
    v = float(score)
    if v>=75: return "Sangat Kuat"
    if v>=60: return "Kuat"
    if v>=45: return "Netral"
    return "Lemah"


def get_static_openrouter_key() -> str:
    # Hardcoded fallback key (least secure; use only if you accept source exposure risk).
    hardcoded_key = " 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"

    key = ""
    try:
        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key and hardcoded_key != "YOUR_OPENROUTER_API_KEY":
        key = hardcoded_key.strip()
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
def call_openrouter(prompt, api_key, model):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.35,
        "max_tokens": 1200,
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

# ── SIDEBAR UI ──
st.title("BEI Screener v3 — Flow + Teknikal + AI")
st.caption("Data BEI (upload) x TradingView real-time x OpenRouter AI narasi otomatis saat pilih saham.")

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
else: st.dataframe(disp,use_container_width=True)
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
        components.html(f"""<div class="tradingview-widget-container" style="height:560px;width:100%;">
          <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
          <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {json.dumps({"autosize":True,"symbol":tv_sym,"interval":"D","timezone":"Asia/Jakarta","theme":"dark","style":"1","locale":"id","allow_symbol_change":True})}
          </script></div>""",height=580)
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
