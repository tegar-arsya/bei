import io
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="BEI Screener v2", layout="wide")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "ringkasan_saham": [
        "kode saham", "nama perusahaan", "sebelumnya", "open price",
        "tanggal perdagangan terakhir", "first trade", "tertinggi", "terendah",
        "penutupan", "selisih", "volume", "nilai", "frekuensi",
        "offer", "offer volume", "bid", "bid volume", "listed shares",
        "tradeble shares", "weight for index", "foreign sell", "foreign buy",
        "non regular volume", "non regular value", "non regular frequency",
    ],
    "ringkasan_broker": ["kode perusahaan", "nama perusahaan", "volume", "nilai", "frekuensi"],
    "ringkasan_perdagangan": ["id instrument", "id board", "volume", "nilai", "frekuensi"],
    "daftar_saham": ["kode", "nama perusahaan", "tanggal pencatatan", "saham", "papan pencatatan"],
}

ALTERNATE_COLUMNS = {
    "daftar_saham": ["id instrument", "id board", "volume", "nilai", "frekuensi"],
}

INDEX_REQUIRED_COLUMNS = [
    "kode indeks", "sebelumnya", "tertinggi", "terendah",
    "penutupan", "selisih", "volume", "nilai", "frekuensi",
]

# Score weights — bisa dituning user
DEFAULT_WEIGHTS = {
    "momentum":        0.25,
    "liquidity":       0.20,
    "flow":            0.20,
    "market_activity": 0.15,
    "volume_trend":    0.10,
    "price_structure": 0.05,
    "broker":          0.05,
}

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def normalize_column_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: normalize_column_name(c) for c in df.columns})

def find_col(df: pd.DataFrame, target: str) -> str:
    target_norm = normalize_column_name(target)
    for col in df.columns:
        if normalize_column_name(col) == target_norm:
            return col
    raise KeyError(f"Kolom '{target}' tidak ditemukan")

def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    raw = series.astype(str).str.strip()
    raw = raw.replace({"": np.nan, "-": np.nan, "--": np.nan})
    raw = raw.str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    parsed_direct = pd.to_numeric(raw, errors="coerce")
    parsed_id = pd.to_numeric(
        raw.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce"
    )
    parsed_en = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
    return parsed_direct.fillna(parsed_id).fillna(parsed_en)

def percentile_series(series: pd.Series) -> pd.Series:
    s = series.copy().replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series([50.0] * len(s), index=s.index)
    return s.rank(pct=True) * 100

def safe_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    return to_numeric(df[find_col(df, col_name)])

def safe_num(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    existing = {normalize_column_name(c) for c in df.columns}
    return {normalize_column_name(c) for c in cols}.issubset(existing)

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    existing = [normalize_column_name(c) for c in df.columns]
    missing = [c for c in required_cols if c not in existing]
    return len(missing) == 0, missing

# ─────────────────────────────────────────────
# FILE LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_table(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    bio = io.BytesIO(uploaded_file.read())
    if file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    elif file_name.endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        raise ValueError("Format file tidak didukung. Gunakan xlsx/xls/csv.")
    return normalize_columns(df)

# ─────────────────────────────────────────────
# SINGLE-DAY SCORE ENGINE
# ─────────────────────────────────────────────
def compute_scores(
    saham_df: pd.DataFrame,
    broker_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    daftar_df: pd.DataFrame,
    weights: Optional[Dict] = None,
) -> pd.DataFrame:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    d = saham_df.copy()
    d["kode saham"]     = d[find_col(d, "kode saham")].astype(str).str.upper().str.strip()
    d["nama perusahaan"]= d[find_col(d, "nama perusahaan")].astype(str).str.strip()

    d["penutupan_num"]  = safe_col(d, "penutupan")
    d["open_num"]       = safe_col(d, "open price")
    d["high_num"]       = safe_col(d, "tertinggi")
    d["low_num"]        = safe_col(d, "terendah")
    d["prev_num"]       = safe_col(d, "sebelumnya")
    d["selisih_num"]    = safe_col(d, "selisih")
    d["nilai_num"]      = safe_col(d, "nilai")
    d["volume_num"]     = safe_col(d, "volume")
    d["frekuensi_num"]  = safe_col(d, "frekuensi")
    d["foreign_buy_num"]= safe_col(d, "foreign buy")
    d["foreign_sell_num"]= safe_col(d, "foreign sell")
    d["bid_vol_num"]    = safe_col(d, "bid volume")
    d["offer_vol_num"]  = safe_col(d, "offer volume")

    # Derived fields
    d["change_pct"]     = np.where(d["prev_num"] != 0, (d["selisih_num"] / d["prev_num"]) * 100, np.nan)
    d["foreign_net"]    = d["foreign_buy_num"] - d["foreign_sell_num"]
    d["foreign_net_ratio"] = np.where(d["nilai_num"] != 0, d["foreign_net"] / d["nilai_num"], 0)
    d["bid_offer_pressure"] = np.where(
        (d["bid_vol_num"] + d["offer_vol_num"]) != 0,
        (d["bid_vol_num"] - d["offer_vol_num"]) / (d["bid_vol_num"] + d["offer_vol_num"]),
        0,
    )

    # True Range proxy (ATR-1d) — gunakan untuk risk level
    d["true_range"] = np.maximum(
        d["high_num"] - d["low_num"],
        np.maximum(
            abs(d["high_num"] - d["prev_num"]),
            abs(d["low_num"] - d["prev_num"])
        )
    )
    d["atr_pct"] = np.where(d["prev_num"] != 0, d["true_range"] / d["prev_num"] * 100, np.nan)

    # Price structure: close position dalam range hari ini (0=low, 1=high)
    d["close_position"] = np.where(
        (d["high_num"] - d["low_num"]) != 0,
        (d["penutupan_num"] - d["low_num"]) / (d["high_num"] - d["low_num"]),
        0.5
    )

    # Daftar saham
    m = daftar_df.copy()
    if has_columns(m, REQUIRED_COLUMNS["daftar_saham"]):
        m["kode"] = m[find_col(m, "kode")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m, "papan pencatatan")].astype(str).str.strip()
        m["tanggal pencatatan"] = m[find_col(m, "tanggal pencatatan")]
    elif has_columns(m, ALTERNATE_COLUMNS["daftar_saham"]):
        m["kode"] = m[find_col(m, "id instrument")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m, "id board")].astype(str).str.strip()
        m["tanggal pencatatan"] = pd.NaT
        m = (m.groupby("kode", dropna=False).agg(
            {"papan pencatatan": lambda x: ",".join(sorted(set(v for v in x if pd.notna(v)))),
             "tanggal pencatatan": "first"}).reset_index())
    else:
        m = pd.DataFrame({"kode": d["kode saham"], "papan pencatatan": pd.NA, "tanggal pencatatan": pd.NaT})

    d = d.merge(m[["kode", "papan pencatatan", "tanggal pencatatan"]], left_on="kode saham", right_on="kode", how="left")

    # Trade data
    t = trade_df.copy()
    t["id instrument"] = t[find_col(t, "id instrument")].astype(str).str.upper().str.strip()
    t["trade_nilai"]     = safe_col(t, "nilai")
    t["trade_frekuensi"] = safe_col(t, "frekuensi")
    t_agg = (t.groupby("id instrument", dropna=False)[["trade_nilai", "trade_frekuensi"]]
             .sum().reset_index().rename(columns={"id instrument": "kode saham"}))
    d = d.merge(t_agg, on="kode saham", how="left")

    # Broker data
    b = broker_df.copy()
    b["kode perusahaan"] = b[find_col(b, "kode perusahaan")].astype(str).str.upper().str.strip()
    b["broker_nilai"]     = safe_col(b, "nilai")
    b["broker_frekuensi"] = safe_col(b, "frekuensi")

    ticker_set     = set(d["kode saham"].dropna().astype(str).str.upper())
    broker_code_set= set(b["kode perusahaan"].dropna().astype(str).str.upper())
    overlap_ratio  = len(ticker_set & broker_code_set) / max(1, min(len(ticker_set), len(broker_code_set)))

    if overlap_ratio >= 0.2:
        b_agg = (b.groupby("kode perusahaan", dropna=False)[["broker_nilai", "broker_frekuensi"]]
                 .sum().reset_index().rename(columns={"kode perusahaan": "kode saham"}))
        d = d.merge(b_agg, on="kode saham", how="left")
        d["broker_score"] = (percentile_series(d["broker_nilai"].fillna(0)) * 0.6 +
                             percentile_series(d["broker_frekuensi"].fillna(0)) * 0.4)
    else:
        d["broker_score"] = 50.0

    # ── SCORES ──
    d["momentum_score"]       = percentile_series(d["change_pct"])
    d["liquidity_score"]      = (percentile_series(d["nilai_num"]) * 0.5 +
                                  percentile_series(d["volume_num"]) * 0.25 +
                                  percentile_series(d["frekuensi_num"]) * 0.25)
    d["flow_score"]           = (percentile_series(d["foreign_net_ratio"]) * 0.6 +
                                  percentile_series(d["bid_offer_pressure"]) * 0.4)
    d["market_activity_score"]= (percentile_series(d["trade_nilai"].fillna(0)) * 0.6 +
                                  percentile_series(d["trade_frekuensi"].fillna(0)) * 0.4)
    # Volume trend proxy: frekuensi per lot
    d["vol_per_freq"]         = np.where(d["frekuensi_num"] != 0, d["volume_num"] / d["frekuensi_num"], np.nan)
    d["volume_trend_score"]   = (percentile_series(d["vol_per_freq"]) * 0.5 +
                                  percentile_series(d["bid_offer_pressure"]) * 0.5)
    # Price structure: close di upper range = bullish
    d["price_structure_score"]= percentile_series(d["close_position"])

    w = weights
    d["final_score"] = (
        d["momentum_score"]        * w["momentum"]        +
        d["liquidity_score"]       * w["liquidity"]       +
        d["flow_score"]            * w["flow"]            +
        d["market_activity_score"] * w["market_activity"] +
        d["volume_trend_score"]    * w["volume_trend"]    +
        d["price_structure_score"] * w["price_structure"] +
        d["broker_score"]          * w["broker"]
    )

    d["kategori"] = pd.cut(
        d["final_score"],
        bins=[-np.inf, 40, 60, 75, np.inf],
        labels=["Rendah", "Menarik", "Tinggi", "Sangat Tinggi"],
    )

    cols = [
        "kode saham", "nama perusahaan", "papan pencatatan", "tanggal pencatatan",
        "penutupan_num", "open_num", "high_num", "low_num", "prev_num",
        "change_pct", "volume_num", "nilai_num", "frekuensi_num",
        "foreign_net", "foreign_net_ratio", "bid_offer_pressure",
        "atr_pct", "close_position",
        "final_score", "kategori",
        "momentum_score", "liquidity_score", "flow_score",
        "market_activity_score", "volume_trend_score", "price_structure_score", "broker_score",
    ]
    return d[cols].sort_values("final_score", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# MULTI-DAY ENGINE — CORE IMPROVEMENT
# ─────────────────────────────────────────────
def compute_multiday_signals(scored_days: List[pd.DataFrame], day_labels: List[str]) -> pd.DataFrame:
    """
    Menggabungkan N hari data menjadi sinyal yang lebih kuat:
    - Score trend (naik/turun)
    - Sinyal konsisten (berapa hari score >= threshold)
    - Foreign flow akumulasi
    - Volume akumulasi relatif
    - Signal strength (composite)
    """
    if len(scored_days) == 1:
        df = scored_days[0].copy()
        df["trend_score"]        = df["final_score"]
        df["signal_consistency"] = 0.0
        df["foreign_acc"]        = df["foreign_net_ratio"]
        df["vol_trend_pct"]      = 0.0
        df["signal_strength"]    = df["final_score"]
        df["signal_label"]       = df["kategori"].astype(str)
        df["days_data"]          = 1
        df["score_day_labels"]   = day_labels[0] if day_labels else "D1"
        return df

    # Buat panel data: kode saham × hari
    all_scores = []
    for i, (day_df, label) in enumerate(zip(scored_days, day_labels)):
        tmp = day_df[["kode saham", "final_score", "foreign_net_ratio",
                       "volume_num", "nilai_num", "change_pct",
                       "momentum_score", "liquidity_score", "flow_score",
                       "market_activity_score", "volume_trend_score",
                       "price_structure_score", "broker_score"]].copy()
        tmp["day_idx"]   = i
        tmp["day_label"] = label
        all_scores.append(tmp)

    panel = pd.concat(all_scores, ignore_index=True)

    # Gunakan hari terakhir sebagai base
    latest = scored_days[-1].copy()

    # Trend score: slope dari final_score linear regression across days
    def calc_slope(group):
        scores = group.sort_values("day_idx")["final_score"].values
        if len(scores) < 2:
            return pd.Series({"trend_slope": 0.0, "score_consistency": 0.0,
                              "foreign_acc_mean": 0.0, "vol_growth": 0.0,
                              "score_latest": scores[-1] if len(scores) else 50.0})
        x = np.arange(len(scores), dtype=float)
        slope = np.polyfit(x, scores, 1)[0]
        # Berapa hari score >= 60
        consistency = float(np.mean(scores >= 60) * 100)
        foreign_vals = group.sort_values("day_idx")["foreign_net_ratio"].values
        foreign_acc  = float(np.mean(foreign_vals))
        vol_vals     = group.sort_values("day_idx")["nilai_num"].values
        vol_growth   = float((vol_vals[-1] - vol_vals[0]) / (vol_vals[0] + 1e-9) * 100) if len(vol_vals) >= 2 else 0.0
        return pd.Series({"trend_slope": float(slope), "score_consistency": consistency,
                          "foreign_acc_mean": foreign_acc, "vol_growth": vol_growth,
                          "score_latest": float(scores[-1])})

    trends = panel.groupby("kode saham").apply(calc_slope).reset_index()

    latest = latest.merge(trends, on="kode saham", how="left")

    # Normalize trend slope ke 0-100 (slope ~0 = 50, +5/hari = 100)
    max_slope = 5.0
    latest["trend_score_norm"] = np.clip(50 + (latest["trend_slope"] * (50 / max_slope)), 0, 100)

    # Signal strength: gabungkan final_score terbaru + trend + konsistensi + foreign acc
    latest["foreign_acc_norm"] = np.clip(50 + latest["foreign_acc_mean"] * 500, 0, 100)
    latest["vol_growth_norm"]  = np.clip(50 + latest["vol_growth"] / 4, 0, 100)

    n_days = len(scored_days)
    latest["signal_strength"] = (
        latest["final_score"]       * 0.35 +
        latest["trend_score_norm"]  * 0.30 +
        latest["score_consistency"] * 0.20 +
        latest["foreign_acc_norm"]  * 0.10 +
        latest["vol_growth_norm"]   * 0.05
    )

    # Label sinyal
    def make_signal_label(row):
        fs  = row["final_score"]
        ts  = row.get("trend_score_norm", 50)
        con = row.get("score_consistency", 0)
        if fs >= 70 and ts >= 60 and con >= 60:
            return "🔥 Breakout Candidate"
        if fs >= 60 and ts >= 60:
            return "📈 Trending Up"
        if fs >= 60 and ts < 45:
            return "⚠️ Fading Momentum"
        if fs < 50 and ts >= 60:
            return "👀 Emerging"
        if fs >= 75:
            return "✅ Sangat Tinggi"
        if fs >= 60:
            return "✅ Tinggi"
        return "⬜ Watch Only"

    latest["signal_label"] = latest.apply(make_signal_label, axis=1)
    latest["days_data"]    = n_days
    latest["score_day_labels"] = " → ".join(day_labels)

    return latest.sort_values("signal_strength", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# MARKET REGIME
# ─────────────────────────────────────────────
def compute_market_regime(index_df: pd.DataFrame) -> Dict:
    d = index_df.copy()
    d["kode indeks"]   = d[find_col(d, "kode indeks")].astype(str).str.upper().str.strip()
    d["sebelumnya_num"]= safe_col(d, "sebelumnya")
    d["selisih_num"]   = safe_col(d, "selisih")
    d["nilai_num"]     = safe_col(d, "nilai")
    d["frekuensi_num"] = safe_col(d, "frekuensi")

    pref = d[d["kode indeks"].str.contains("IHSG|COMPOSITE|JCI", regex=True, na=False)]
    row  = pref.iloc[0] if not pref.empty else d.sort_values("nilai_num", ascending=False).iloc[0]

    prev   = float(row["sebelumnya_num"]) if pd.notna(row["sebelumnya_num"]) else 0.0
    diff   = float(row["selisih_num"]) if pd.notna(row["selisih_num"]) else 0.0
    change_pct = (diff / prev) * 100 if prev != 0 else 0.0

    momentum = float(np.clip(50 + (change_pct * 8), 0, 100))
    val_rank = float(percentile_series(d["nilai_num"]).loc[row.name])
    fr_rank  = float(percentile_series(d["frekuensi_num"]).loc[row.name])
    activity = (val_rank * 0.6) + (fr_rank * 0.4)
    regime_score = float(np.clip((momentum * 0.7) + (activity * 0.3), 0, 100))
    regime_label = "Risk-On" if regime_score >= 60 else "Netral" if regime_score >= 45 else "Risk-Off"

    return {
        "index_code": str(row["kode indeks"]),
        "index_change_pct": float(change_pct),
        "regime_score": regime_score,
        "regime_label": regime_label,
    }


# ─────────────────────────────────────────────
# DETAILED ANALYSIS (IMPROVED)
# ─────────────────────────────────────────────
def factor_label(score: float) -> str:
    if pd.isna(score): return "N/A"
    if score >= 75: return "Sangat Kuat"
    if score >= 60: return "Kuat"
    if score >= 45: return "Netral"
    return "Lemah"

def generate_detailed_analysis(
    row: pd.Series,
    universe_df: pd.DataFrame,
    market_regime: Optional[Dict] = None,
) -> Dict:
    factors = {
        "Momentum":       safe_num(row.get("momentum_score"), np.nan),
        "Likuiditas":     safe_num(row.get("liquidity_score"), np.nan),
        "Flow":           safe_num(row.get("flow_score"), np.nan),
        "Market Activity":safe_num(row.get("market_activity_score"), np.nan),
        "Vol Trend":      safe_num(row.get("volume_trend_score"), np.nan),
        "Price Structure":safe_num(row.get("price_structure_score"), np.nan),
        "Broker":         safe_num(row.get("broker_score"), np.nan),
    }

    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    strengths  = [f"{n} ({s:.1f})" for n, s in sorted_factors if s >= 60]
    weaknesses = [f"{n} ({s:.1f})" for n, s in sorted_factors if s < 45]

    final_score   = safe_num(row.get("final_score", 0))
    signal_str    = safe_num(row.get("signal_strength", final_score))
    change_pct    = safe_num(row.get("change_pct", 0))
    foreign_net   = safe_num(row.get("foreign_net", 0))
    trend_slope   = safe_num(row.get("trend_slope", 0))
    consistency   = safe_num(row.get("score_consistency", 0))
    days_data     = int(safe_num(row.get("days_data", 1)))

    final_rank_pct    = safe_num(universe_df["final_score"].rank(pct=True).loc[row.name] * 100, 50)
    signal_rank_pct   = safe_num(universe_df.get("signal_strength", universe_df["final_score"]).rank(pct=True).loc[row.name] * 100, 50)

    factor_values  = np.array([safe_num(v, 50) for v in factors.values()], dtype=float)
    factor_std     = float(np.std(factor_values))
    consistency_score = float(np.clip(100 - (factor_std * 2.2), 0, 100))

    regime_score = safe_num(market_regime.get("regime_score") if market_regime else np.nan, 50)
    conviction_score = float(np.clip(
        signal_str * 0.45 + signal_rank_pct * 0.20 + consistency_score * 0.20 + regime_score * 0.15,
        0, 100
    ))

    # ATR-based risk levels (lebih akurat dari fixed %)
    close_price = safe_num(row.get("penutupan_num"), np.nan)
    atr_pct     = safe_num(row.get("atr_pct"), 2.0)  # default 2% jika tidak ada
    atr_pct     = max(atr_pct, 0.5)  # floor 0.5%

    if pd.notna(close_price) and close_price > 0:
        atr_abs         = close_price * atr_pct / 100
        entry_zone_low  = close_price * 0.995
        entry_zone_high = close_price + atr_abs * 0.3
        invalidation    = close_price - atr_abs * 1.5   # 1.5× ATR stop
        tp1             = close_price + atr_abs * 2.0   # 2× ATR TP1
        tp2             = close_price + atr_abs * 3.5   # 3.5× ATR TP2
        rr_ratio        = (tp1 - close_price) / max(close_price - invalidation, 1e-9)
    else:
        entry_zone_low = entry_zone_high = invalidation = tp1 = tp2 = np.nan
        rr_ratio = np.nan

    # Risk score
    risk_score = 30.0
    risk_score += min(abs(change_pct) * 2.5, 20)
    if foreign_net < 0: risk_score += 15
    if factors["Flow"] < 45: risk_score += 10
    if factors["Likuiditas"] < 45: risk_score += 8
    if factors["Momentum"] > 85: risk_score += 5
    if trend_slope < 0 and days_data > 1: risk_score += 10  # skor menurun antar hari
    risk_score = float(np.clip(risk_score, 0, 100))

    rr_profile = "Agresif" if conviction_score >= 75 and risk_score <= 55 else \
                 "Moderat"  if conviction_score >= 60 else "Defensif"

    # Action suggestion
    if days_data >= 3 and consistency >= 60 and final_score >= 70:
        action = "Sinyal kuat & konsisten ≥3 hari. Prioritas masuk bertahap, SL ketat di bawah invalidation."
    elif days_data >= 2 and trend_slope > 0 and final_score >= 60:
        action = "Momentum terkonfirmasi naik. Pantau volume sebagai konfirmasi; entry jika volume meningkat."
    elif final_score >= 75:
        action = "Score tinggi hari ini. Butuh konfirmasi multi-hari. Watchlist prioritas."
    elif final_score >= 60:
        action = "Cukup menarik. Tunggu 1-2 hari konfirmasi tren sebelum entry penuh."
    else:
        action = "Belum prioritas. Simpan watchlist, tunggu perbaikan score dan tren."

    risks = []
    if change_pct >= 8:
        risks.append("Harga sudah naik cepat; risiko mengejar harga tinggi.")
    if foreign_net < 0:
        risks.append("Net foreign negatif; tekanan distribusi dari asing masih ada.")
    if days_data >= 2 and trend_slope < -1:
        risks.append("Score menurun hari-ke-hari; momentum melemah.")
    if len(weaknesses) >= 2:
        risks.append("Beberapa faktor inti lemah; sinyal belum solid.")
    if not pd.isna(rr_ratio) and rr_ratio < 1.5:
        risks.append(f"R/R ratio {rr_ratio:.1f}× — di bawah ideal 2×. Pertimbangkan ulang sizing.")
    if not risks:
        risks.append("Tidak ada red flag besar dari kombinasi faktor saat ini.")

    return {
        "factors": factors,
        "strengths":  strengths  if strengths  else ["Belum ada faktor dominan."],
        "weaknesses": weaknesses if weaknesses else ["Tidak ada faktor lemah."],
        "action": action,
        "risks": risks,
        "final_rank_pct":   final_rank_pct,
        "signal_rank_pct":  signal_rank_pct,
        "consistency_score": consistency_score,
        "conviction_score": conviction_score,
        "risk_score":       risk_score,
        "rr_profile":       rr_profile,
        "rr_ratio":         rr_ratio,
        "atr_pct":          atr_pct,
        "entry_zone_low":   entry_zone_low,
        "entry_zone_high":  entry_zone_high,
        "invalidation":     invalidation,
        "tp1": tp1, "tp2": tp2,
        "trend_slope":  trend_slope,
        "score_consistency": consistency,
        "days_data": days_data,
    }


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("BEI Screener v2 — Multi-Hari")
st.caption("Upload 1–5 hari data. Semakin banyak hari, sinyal semakin terkonfirmasi.")

with st.sidebar:
    st.header("📁 Upload File BEI")
    st.markdown("**Per Hari:** upload 4 file berikut")

    n_days = st.number_input("Jumlah hari data", min_value=1, max_value=5, value=1, step=1)

    day_data = []
    for i in range(int(n_days)):
        with st.expander(f"Hari {i+1}", expanded=(i == int(n_days) - 1)):
            day_label  = st.text_input(f"Label hari {i+1}", value=f"D{i+1}", key=f"label_{i}")
            saham_f    = st.file_uploader(f"Ringkasan Saham",      type=["xlsx","xls","csv"], key=f"saham_{i}")
            broker_f   = st.file_uploader(f"Ringkasan Broker",     type=["xlsx","xls","csv"], key=f"broker_{i}")
            perdagang_f= st.file_uploader(f"Ringkasan Perdagangan",type=["xlsx","xls","csv"], key=f"perdagang_{i}")
            daftar_f   = st.file_uploader(f"Daftar Saham",         type=["xlsx","xls","csv"], key=f"daftar_{i}")
            day_data.append({"label": day_label, "saham": saham_f, "broker": broker_f,
                             "perdagangan": perdagang_f, "daftar": daftar_f})

    st.markdown("---")
    indeks_file = st.file_uploader("Ringkasan Indeks (Opsional)", type=["xlsx","xls","csv"])

    st.markdown("---")
    st.markdown("**⚙️ Bobot Scoring**")
    with st.expander("Sesuaikan bobot"):
        w_mom  = st.slider("Momentum",        0, 100, 25)
        w_liq  = st.slider("Likuiditas",      0, 100, 20)
        w_flow = st.slider("Flow",            0, 100, 20)
        w_ma   = st.slider("Market Activity", 0, 100, 15)
        w_vt   = st.slider("Volume Trend",    0, 100, 10)
        w_ps   = st.slider("Price Structure", 0, 100,  5)
        w_br   = st.slider("Broker",          0, 100,  5)
        total_w = w_mom + w_liq + w_flow + w_ma + w_vt + w_ps + w_br
        if total_w == 0: total_w = 1
        custom_weights = {
            "momentum": w_mom/total_w, "liquidity": w_liq/total_w,
            "flow": w_flow/total_w, "market_activity": w_ma/total_w,
            "volume_trend": w_vt/total_w, "price_structure": w_ps/total_w,
            "broker": w_br/total_w,
        }
        st.caption(f"Total bobot: {total_w} (dinormalisasi ke 100%)")

# Check semua hari punya 4 file
complete_days = [d for d in day_data if all([d["saham"], d["broker"], d["perdagangan"], d["daftar"]])]
if not complete_days:
    st.info("Upload minimal 1 set lengkap (4 file) untuk memulai screening.")
    st.stop()

# Load & score per hari
scored_days  = []
day_labels   = []
market_regime= None

with st.spinner("Memproses data..."):
    for day in complete_days:
        try:
            s_df = load_table(day["saham"])
            b_df = load_table(day["broker"])
            p_df = load_table(day["perdagangan"])
            d_df = load_table(day["daftar"])

            # Validate
            checks = {"ringkasan_saham": s_df, "ringkasan_broker": b_df,
                      "ringkasan_perdagangan": p_df, "daftar_saham": d_df}
            all_ok = True
            for key, df in checks.items():
                ok, missing = validate_columns(df, REQUIRED_COLUMNS[key])
                if key in ALTERNATE_COLUMNS and not ok:
                    ok = has_columns(df, ALTERNATE_COLUMNS[key])
                if not ok:
                    all_ok = False
                    st.error(f"[{day['label']}] Kolom wajib {key} tidak lengkap: {', '.join(missing)}")

            if all_ok:
                sc = compute_scores(s_df, b_df, p_df, d_df, weights=custom_weights)
                scored_days.append(sc)
                day_labels.append(day["label"])
        except Exception as exc:
            st.error(f"[{day['label']}] Gagal proses: {exc}")

    if indeks_file is not None:
        try:
            indeks_df = load_table(indeks_file)
            ok_idx, missing_idx = validate_columns(indeks_df, INDEX_REQUIRED_COLUMNS)
            if ok_idx:
                market_regime = compute_market_regime(indeks_df)
            else:
                st.warning(f"Ringkasan Indeks kolom tidak lengkap: {', '.join(missing_idx)}")
        except Exception as exc:
            st.warning(f"Ringkasan Indeks tidak terbaca: {exc}")

if not scored_days:
    st.error("Tidak ada data valid untuk diproses.")
    st.stop()

# Multi-day signal computation
combined = compute_multiday_signals(scored_days, day_labels)

# Apply market regime adjustment
if market_regime is not None:
    combined["final_score"]   = combined["final_score"]   * 0.90 + market_regime["regime_score"] * 0.10
    combined["signal_strength"]= combined["signal_strength"] * 0.90 + market_regime["regime_score"] * 0.10

# ── MARKET REGIME BAR ──
if market_regime is not None:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Index",         str(market_regime["index_code"]))
    r2.metric("Change %",      f"{market_regime['index_change_pct']:.2f}%")
    r3.metric("Regime",        str(market_regime["regime_label"]))
    r4.metric("Regime Score",  f"{market_regime['regime_score']:.2f}")

days_loaded = len(scored_days)
if days_loaded > 1:
    st.success(f"✅ {days_loaded} hari data dimuat: {' → '.join(day_labels)}. Sinyal multi-hari aktif.")
else:
    st.info(f"ℹ️ 1 hari data dimuat. Upload lebih banyak hari untuk sinyal yang lebih kuat.")

# ── FILTERS ──
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    min_score = st.slider("Minimum Signal Strength", 0, 100, 60)
with col2:
    papan_options = sorted([x for x in combined["papan pencatatan"].dropna().unique().tolist() if str(x).strip()])
    papan_filter  = st.selectbox("Filter Papan", ["Semua"] + papan_options)
with col3:
    top_n = st.number_input("Top N", min_value=5, max_value=300, value=25, step=5)
with col4:
    signal_filter = st.selectbox("Filter Sinyal", ["Semua", "🔥 Breakout Candidate", "📈 Trending Up",
                                                    "⚠️ Fading Momentum", "👀 Emerging"])

view = combined[combined["signal_strength"] >= min_score].copy()
if papan_filter != "Semua":
    view = view[view["papan pencatatan"] == papan_filter]
if signal_filter != "Semua" and "signal_label" in view.columns:
    view = view[view["signal_label"] == signal_filter]
view = view.head(int(top_n))

# ── MAIN TABLE ──
st.subheader("Kandidat Saham")
display_cols = ["kode saham", "nama perusahaan", "papan pencatatan",
                "penutupan_num", "change_pct", "foreign_net",
                "final_score", "signal_strength", "signal_label"]
if days_loaded > 1:
    display_cols += ["trend_slope", "score_consistency"]

display_df = view[[c for c in display_cols if c in view.columns]].rename(columns={
    "penutupan_num": "close", "foreign_net": "net_foreign",
    "signal_strength": "signal", "signal_label": "sinyal",
    "trend_slope": "tren/hari", "score_consistency": "konsistensi%",
})

def color_signal(val):
    if isinstance(val, float):
        if val >= 75: return "background-color: #1a472a; color: #90ee90"
        if val >= 60: return "background-color: #1e3a5f; color: #87ceeb"
        if val >= 45: return "background-color: #3d2b00; color: #ffd700"
        return "background-color: #3d0000; color: #ff9999"
    return ""



st.dataframe(
    display_df.style.map(color_signal, subset=[c for c in ["signal", "final_score"] if c in display_df.columns]),
    use_container_width=True,
)


# Export
csv_data = view.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export CSV", data=csv_data, file_name="bei_screener_result.csv", mime="text/csv")

if view.empty:
    st.warning("Tidak ada kandidat sesuai filter.")
    st.stop()

# ── DETAIL KANDIDAT ──
selected_code = st.selectbox("Pilih saham kandidat", options=view["kode saham"].tolist())
selected_row  = combined[combined["kode saham"] == selected_code].iloc[0]

st.markdown("---")
st.subheader(f"Detail: {selected_code} — {str(selected_row.get('signal_label',''))}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Signal Strength", f"{selected_row.get('signal_strength', selected_row['final_score']):.1f}")
m2.metric("Final Score",     f"{selected_row['final_score']:.1f}")
m3.metric("Change %",        f"{selected_row['change_pct']:.2f}%")
m4.metric("Net Foreign",     f"{selected_row['foreign_net']:,.0f}")
if days_loaded > 1:
    m5.metric("Tren/Hari",   f"{selected_row.get('trend_slope', 0):.1f} pts")
    st.caption(f"Data: {selected_row.get('score_day_labels','')}")

# Score breakdown
st.markdown("#### Breakdown Score")
breakdown_data = {
    "Faktor":  ["Momentum", "Likuiditas", "Flow", "Market Activity", "Vol Trend", "Price Structure", "Broker"],
    "Skor":    [selected_row.get(c, np.nan) for c in [
                "momentum_score","liquidity_score","flow_score",
                "market_activity_score","volume_trend_score","price_structure_score","broker_score"]],
}
bd_df = pd.DataFrame(breakdown_data)
bd_df["Status"] = bd_df["Skor"].apply(factor_label)
st.dataframe(bd_df.style.map(color_signal, subset=["Skor"]), use_container_width=True)

# ── DETAILED ANALYSIS ──
analysis = generate_detailed_analysis(selected_row, combined, market_regime)

st.markdown("#### Analisis Komposit")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Signal Rank",    f"{analysis['signal_rank_pct']:.1f}%")
d2.metric("Conviction",     f"{analysis['conviction_score']:.1f}")
d3.metric("Risk Score",     f"{analysis['risk_score']:.1f}")
d4.metric("Profil",         str(analysis["rr_profile"]))
if not pd.isna(analysis.get("rr_ratio")):
    d5.metric("R/R Ratio",  f"{analysis['rr_ratio']:.2f}×")

if days_loaded > 1:
    st.markdown("**Signal Multi-Hari**")
    mc1, mc2 = st.columns(2)
    mc1.metric("Tren Skor",    f"{analysis['trend_slope']:+.1f} pts/hari")
    mc2.metric("Konsistensi",  f"{analysis['score_consistency']:.0f}% hari score ≥ 60")

a1, a2 = st.columns(2)
with a1:
    st.write("**Kekuatan:**")
    for item in analysis["strengths"]:
        st.write(f"✅ {item}")
    st.write("**Kelemahan:**")
    for item in analysis["weaknesses"]:
        st.write(f"⚠️ {item}")
with a2:
    st.write("**Risiko:**")
    for item in analysis["risks"]:
        st.write(f"🔴 {item}")
    st.markdown("**Rencana Aksi:**")
    st.info(str(analysis["action"]))

# ATR-based levels
st.markdown("#### Level Eksekusi (Berbasis ATR)")
st.caption(f"ATR-1d: {analysis['atr_pct']:.2f}% dari close. Stop = 1.5× ATR, TP1 = 2× ATR, TP2 = 3.5× ATR.")
lv1, lv2, lv3, lv4, lv5 = st.columns(5)
fmt = lambda v: f"{v:.2f}" if not pd.isna(v) else "-"
lv1.metric("Entry Low",   fmt(analysis["entry_zone_low"]))
lv2.metric("Entry High",  fmt(analysis["entry_zone_high"]))
lv3.metric("Invalidation",fmt(analysis["invalidation"]))
lv4.metric("TP1",         fmt(analysis["tp1"]))
lv5.metric("TP2",         fmt(analysis["tp2"]))

# ── TRADINGVIEW EMBED ──
st.markdown("---")
st.subheader("Chart & Analisis TradingView")

tv_code = st.text_input("Kode Saham IDX", value=selected_code, key="tv_input").strip().upper()
if not tv_code: tv_code = selected_code

tv_symbol    = f"IDX:{tv_code}"
tv_sym_slug  = f"IDX-{tv_code}"

tab_chart, tab_fin, tab_news, tab_tech = st.tabs(["Chart", "Financials", "News", "Technicals"])

with tab_chart:
    widget_config = {
        "autosize": True, "symbol": tv_symbol, "interval": "D",
        "timezone": "Asia/Jakarta", "theme": "dark", "style": "1",
        "locale": "id", "allow_symbol_change": True, "calendar": False,
    }
    components.html(f"""
    <div class="tradingview-widget-container" style="height:560px;width:100%;">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {json.dumps(widget_config)}
      </script>
    </div>
    """, height=580)

with tab_fin:
    components.iframe(f"https://id.tradingview.com/symbols/{tv_sym_slug}/financials-overview/", height=700, scrolling=True)

with tab_news:
    components.html(f"""
    <div class="tradingview-widget-container" style="width:100%;height:700px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {json.dumps({"feedMode":"symbol","symbol":tv_symbol,"isTransparent":False,
                   "displayMode":"regular","width":"100%","height":700,"colorTheme":"dark","locale":"id"})}
      </script>
    </div>
    """, height=720)

with tab_tech:
    components.html(f"""
    <div class="tradingview-widget-container" style="width:100%;height:700px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {json.dumps({"interval":"1D","width":"100%","isTransparent":False,"height":700,
                   "symbol":tv_symbol,"showIntervalTabs":True,"locale":"id","colorTheme":"dark"})}
      </script>
    </div>
    """, height=720)