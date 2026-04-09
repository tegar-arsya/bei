import io
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="BEI Screener (4 File)", layout="wide")

REQUIRED_COLUMNS = {
    "ringkasan_saham": [
        "kode saham",
        "nama perusahaan",
        "sebelumnya",
        "open price",
        "tanggal perdagangan terakhir",
        "first trade",
        "tertinggi",
        "terendah",
        "penutupan",
        "selisih",
        "volume",
        "nilai",
        "frekuensi",
        "offer",
        "offer volume",
        "bid",
        "bid volume",
        "listed shares",
        "tradeble shares",
        "weight for index",
        "foreign sell",
        "foreign buy",
        "non regular volume",
        "non regular value",
        "non regular frequency",
    ],
    "ringkasan_broker": [
        "kode perusahaan",
        "nama perusahaan",
        "volume",
        "nilai",
        "frekuensi",
    ],
    "ringkasan_perdagangan": [
        "id instrument",
        "id board",
        "volume",
        "nilai",
        "frekuensi",
    ],
    "daftar_saham": [
        "kode",
        "nama perusahaan",
        "tanggal pencatatan",
        "saham",
        "papan pencatatan",
    ],
}

ALTERNATE_COLUMNS = {
    "daftar_saham": ["id instrument", "id board", "volume", "nilai", "frekuensi"],
}

INDEX_REQUIRED_COLUMNS = [
    "kode indeks",
    "sebelumnya",
    "tertinggi",
    "terendah",
    "penutupan",
    "selisih",
    "volume",
    "nilai",
    "frekuensi",
]


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


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    existing = [normalize_column_name(c) for c in df.columns]
    missing = [c for c in required_cols if c not in existing]
    return len(missing) == 0, missing


def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    existing = {normalize_column_name(c) for c in df.columns}
    required = {normalize_column_name(c) for c in cols}
    return required.issubset(existing)


def safe_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    return to_numeric(df[find_col(df, col_name)])


def compute_market_regime(index_df: pd.DataFrame) -> Dict[str, float | str]:
    d = index_df.copy()
    d["kode indeks"] = d[find_col(d, "kode indeks")].astype(str).str.upper().str.strip()
    d["sebelumnya_num"] = safe_col(d, "sebelumnya")
    d["selisih_num"] = safe_col(d, "selisih")
    d["nilai_num"] = safe_col(d, "nilai")
    d["frekuensi_num"] = safe_col(d, "frekuensi")

    pref = d[d["kode indeks"].str.contains("IHSG|COMPOSITE|JCI", regex=True, na=False)]
    row = pref.iloc[0] if not pref.empty else d.sort_values("nilai_num", ascending=False).iloc[0]

    prev = float(row["sebelumnya_num"]) if pd.notna(row["sebelumnya_num"]) else 0.0
    diff = float(row["selisih_num"]) if pd.notna(row["selisih_num"]) else 0.0
    change_pct = (diff / prev) * 100 if prev != 0 else 0.0

    momentum = float(np.clip(50 + (change_pct * 8), 0, 100))
    val_rank = float(percentile_series(d["nilai_num"]).loc[row.name])
    fr_rank = float(percentile_series(d["frekuensi_num"]).loc[row.name])
    activity = (val_rank * 0.6) + (fr_rank * 0.4)
    regime_score = float(np.clip((momentum * 0.7) + (activity * 0.3), 0, 100))

    regime_label = "Risk-On" if regime_score >= 60 else "Netral" if regime_score >= 45 else "Risk-Off"
    return {
        "index_code": str(row["kode indeks"]),
        "index_change_pct": float(change_pct),
        "regime_score": regime_score,
        "regime_label": regime_label,
    }


def compute_scores(
    saham_df: pd.DataFrame,
    broker_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    daftar_df: pd.DataFrame,
) -> pd.DataFrame:
    d = saham_df.copy()

    d["kode saham"] = d[find_col(d, "kode saham")].astype(str).str.upper().str.strip()
    d["nama perusahaan"] = d[find_col(d, "nama perusahaan")].astype(str).str.strip()

    d["penutupan_num"] = safe_col(d, "penutupan")
    d["selisih_num"] = safe_col(d, "selisih")
    d["nilai_num"] = safe_col(d, "nilai")
    d["volume_num"] = safe_col(d, "volume")
    d["frekuensi_num"] = safe_col(d, "frekuensi")
    d["foreign_buy_num"] = safe_col(d, "foreign buy")
    d["foreign_sell_num"] = safe_col(d, "foreign sell")
    d["bid_vol_num"] = safe_col(d, "bid volume")
    d["offer_vol_num"] = safe_col(d, "offer volume")

    d["change_pct"] = np.where(d["penutupan_num"] != 0, (d["selisih_num"] / d["penutupan_num"]) * 100, np.nan)
    d["foreign_net"] = d["foreign_buy_num"] - d["foreign_sell_num"]
    d["foreign_net_ratio"] = np.where(d["nilai_num"] != 0, d["foreign_net"] / d["nilai_num"], 0)
    d["bid_offer_pressure"] = np.where(
        (d["bid_vol_num"] + d["offer_vol_num"]) != 0,
        (d["bid_vol_num"] - d["offer_vol_num"]) / (d["bid_vol_num"] + d["offer_vol_num"]),
        0,
    )

    m = daftar_df.copy()
    if has_columns(m, REQUIRED_COLUMNS["daftar_saham"]):
        m["kode"] = m[find_col(m, "kode")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m, "papan pencatatan")].astype(str).str.strip()
        m["tanggal pencatatan"] = m[find_col(m, "tanggal pencatatan")]
    elif has_columns(m, ALTERNATE_COLUMNS["daftar_saham"]):
        m["kode"] = m[find_col(m, "id instrument")].astype(str).str.upper().str.strip()
        m["papan pencatatan"] = m[find_col(m, "id board")].astype(str).str.strip()
        m["tanggal pencatatan"] = pd.NaT
        m = (
            m.groupby("kode", dropna=False)
            .agg(
                {
                    "papan pencatatan": lambda x: ",".join(sorted(set(v for v in x if pd.notna(v)))),
                    "tanggal pencatatan": "first",
                }
            )
            .reset_index()
        )
    else:
        m = pd.DataFrame({"kode": d["kode saham"], "papan pencatatan": pd.NA, "tanggal pencatatan": pd.NaT})

    d = d.merge(m[["kode", "papan pencatatan", "tanggal pencatatan"]], left_on="kode saham", right_on="kode", how="left")

    t = trade_df.copy()
    t["id instrument"] = t[find_col(t, "id instrument")].astype(str).str.upper().str.strip()
    t["trade_nilai"] = safe_col(t, "nilai")
    t["trade_frekuensi"] = safe_col(t, "frekuensi")
    t_agg = (
        t.groupby("id instrument", dropna=False)[["trade_nilai", "trade_frekuensi"]]
        .sum()
        .reset_index()
        .rename(columns={"id instrument": "kode saham"})
    )
    d = d.merge(t_agg, on="kode saham", how="left")

    b = broker_df.copy()
    b["kode perusahaan"] = b[find_col(b, "kode perusahaan")].astype(str).str.upper().str.strip()
    b["broker_nilai"] = safe_col(b, "nilai")
    b["broker_frekuensi"] = safe_col(b, "frekuensi")

    ticker_set = set(d["kode saham"].dropna().astype(str).str.upper().tolist())
    broker_code_set = set(b["kode perusahaan"].dropna().astype(str).str.upper().tolist())
    overlap = len(ticker_set.intersection(broker_code_set))
    overlap_ratio = overlap / max(1, min(len(ticker_set), len(broker_code_set)))

    if overlap_ratio >= 0.2:
        b_agg = (
            b.groupby("kode perusahaan", dropna=False)[["broker_nilai", "broker_frekuensi"]]
            .sum()
            .reset_index()
            .rename(columns={"kode perusahaan": "kode saham"})
        )
        d = d.merge(b_agg, on="kode saham", how="left")
        d["broker_score"] = percentile_series(d["broker_nilai"].fillna(0)) * 0.6 + percentile_series(
            d["broker_frekuensi"].fillna(0)
        ) * 0.4
    else:
        d["broker_score"] = 50.0

    d["momentum_score"] = percentile_series(d["change_pct"])
    d["liquidity_score"] = (
        percentile_series(d["nilai_num"]) * 0.5
        + percentile_series(d["volume_num"]) * 0.25
        + percentile_series(d["frekuensi_num"]) * 0.25
    )
    d["flow_score"] = percentile_series(d["foreign_net_ratio"]) * 0.6 + percentile_series(d["bid_offer_pressure"]) * 0.4
    d["market_activity_score"] = percentile_series(d["trade_nilai"].fillna(0)) * 0.6 + percentile_series(
        d["trade_frekuensi"].fillna(0)
    ) * 0.4

    d["final_score"] = (
        d["momentum_score"] * 0.30
        + d["liquidity_score"] * 0.30
        + d["flow_score"] * 0.20
        + d["market_activity_score"] * 0.15
        + d["broker_score"] * 0.05
    )

    d["kategori"] = pd.cut(
        d["final_score"],
        bins=[-np.inf, 40, 60, 75, np.inf],
        labels=["Rendah", "Menarik", "Tinggi", "Sangat Tinggi"],
    )

    cols = [
        "kode saham",
        "nama perusahaan",
        "papan pencatatan",
        "tanggal pencatatan",
        "penutupan_num",
        "change_pct",
        "volume_num",
        "nilai_num",
        "foreign_net",
        "final_score",
        "kategori",
        "momentum_score",
        "liquidity_score",
        "flow_score",
        "market_activity_score",
        "broker_score",
    ]
    return d[cols].sort_values("final_score", ascending=False)


def factor_label(score: float) -> str:
    if pd.isna(score):
        return "N/A"
    if score >= 75:
        return "Sangat Kuat"
    if score >= 60:
        return "Kuat"
    if score >= 45:
        return "Netral"
    return "Lemah"


def safe_num(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def generate_detailed_analysis(row: pd.Series, universe_df: pd.DataFrame, market_regime: Dict | None = None) -> Dict[str, object]:
    factors = {
        "Momentum": safe_num(row.get("momentum_score", np.nan), np.nan),
        "Likuiditas": safe_num(row.get("liquidity_score", np.nan), np.nan),
        "Flow": safe_num(row.get("flow_score", np.nan), np.nan),
        "Market Activity": safe_num(row.get("market_activity_score", np.nan), np.nan),
        "Broker": safe_num(row.get("broker_score", np.nan), np.nan),
    }

    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    strengths = [f"{name} ({score:.1f})" for name, score in sorted_factors if score >= 60]
    weaknesses = [f"{name} ({score:.1f})" for name, score in sorted_factors if score < 45]

    final_score = safe_num(row.get("final_score", 0))
    change_pct = safe_num(row.get("change_pct", 0))
    foreign_net = safe_num(row.get("foreign_net", 0))

    final_rank_pct = safe_num(universe_df["final_score"].rank(pct=True).loc[row.name] * 100, 50)
    momentum_rank_pct = safe_num(universe_df["momentum_score"].rank(pct=True).loc[row.name] * 100, 50)
    flow_rank_pct = safe_num(universe_df["flow_score"].rank(pct=True).loc[row.name] * 100, 50)
    liq_rank_pct = safe_num(universe_df["liquidity_score"].rank(pct=True).loc[row.name] * 100, 50)

    factor_values = np.array([safe_num(v, 50) for v in factors.values()], dtype=float)
    factor_std = float(np.std(factor_values))
    consistency_score = float(np.clip(100 - (factor_std * 2.2), 0, 100))

    regime_score = safe_num(market_regime.get("regime_score") if market_regime else np.nan, 50)
    conviction_score = float(
        np.clip(
            (final_score * 0.45)
            + (final_rank_pct * 0.20)
            + (consistency_score * 0.20)
            + (regime_score * 0.15),
            0,
            100,
        )
    )

    risk_score = 35.0
    risk_score += min(abs(change_pct) * 3, 25)
    if foreign_net < 0:
        risk_score += 15
    if factors["Flow"] < 45:
        risk_score += 10
    if factors["Likuiditas"] < 45:
        risk_score += 8
    if factors["Momentum"] > 85:
        risk_score += 6
    risk_score = float(np.clip(risk_score, 0, 100))

    rr_profile = "Agresif" if conviction_score >= 75 and risk_score <= 55 else "Moderat" if conviction_score >= 60 else "Defensif"

    if final_score >= 75:
        action = "Kandidat kuat. Fokus ke timing entry bertahap dan disiplin risk management."
    elif final_score >= 60:
        action = "Menarik untuk watchlist prioritas. Tunggu konfirmasi lanjutan (harga/volume) sebelum entry penuh."
    else:
        action = "Belum prioritas. Simpan di watchlist dan tunggu perbaikan faktor kunci."

    risks = []
    if change_pct >= 8:
        risks.append("Harga sudah naik cepat; risiko kejar harga meningkat.")
    if foreign_net < 0:
        risks.append("Net foreign masih negatif; ada tekanan distribusi dari asing.")
    if len(weaknesses) >= 2:
        risks.append("Beberapa faktor inti masih lemah; sinyal belum kompak.")
    if not risks:
        risks.append("Tidak ada red flag besar dari kombinasi faktor saat ini.")

    close_price = safe_num(row.get("penutupan_num", np.nan), np.nan)
    if pd.notna(close_price):
        entry_zone_low = close_price * 0.985
        entry_zone_high = close_price * 1.010
        invalidation = close_price * 0.965
        tp1 = close_price * 1.04
        tp2 = close_price * 1.08
    else:
        entry_zone_low, entry_zone_high, invalidation, tp1, tp2 = [np.nan] * 5

    return {
        "factors": factors,
        "strengths": strengths if strengths else ["Belum ada faktor dominan di atas ambang kuat."],
        "weaknesses": weaknesses if weaknesses else ["Tidak ada faktor yang masuk kategori lemah."],
        "action": action,
        "risks": risks,
        "final_rank_pct": final_rank_pct,
        "momentum_rank_pct": momentum_rank_pct,
        "flow_rank_pct": flow_rank_pct,
        "liq_rank_pct": liq_rank_pct,
        "consistency_score": consistency_score,
        "conviction_score": conviction_score,
        "risk_score": risk_score,
        "rr_profile": rr_profile,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "invalidation": invalidation,
        "tp1": tp1,
        "tp2": tp2,
    }


st.title("BEI Screener (4 File Saja)")
st.caption("Versi ringkas: 4 file BEI sebagai core. Ringkasan Indeks opsional sebagai filter market.")

with st.sidebar:
    st.header("Upload 4 File BEI")
    saham_file = st.file_uploader("1) Ringkasan Saham", type=["xlsx", "xls", "csv"])
    broker_file = st.file_uploader("2) Ringkasan Broker", type=["xlsx", "xls", "csv"])
    perdagangan_file = st.file_uploader("3) Ringkasan Perdagangan & Rekap", type=["xlsx", "xls", "csv"])
    daftar_file = st.file_uploader("4) Daftar Saham", type=["xlsx", "xls", "csv"])
    indeks_file = st.file_uploader("5) Ringkasan Indeks (Opsional)", type=["xlsx", "xls", "csv"])

if not all([saham_file, broker_file, perdagangan_file, daftar_file]):
    st.info("Silakan upload ke-4 file terlebih dahulu.")
    st.stop()

try:
    saham_df = load_table(saham_file)
    broker_df = load_table(broker_file)
    perdagangan_df = load_table(perdagangan_file)
    daftar_df = load_table(daftar_file)
except Exception as exc:
    st.error(f"Gagal membaca file: {exc}")
    st.stop()

checks = {
    "ringkasan_saham": saham_df,
    "ringkasan_broker": broker_df,
    "ringkasan_perdagangan": perdagangan_df,
    "daftar_saham": daftar_df,
}

all_ok = True
for key, df in checks.items():
    ok, missing = validate_columns(df, REQUIRED_COLUMNS[key])
    if key in ALTERNATE_COLUMNS and not ok:
        ok = has_columns(df, ALTERNATE_COLUMNS[key])
        missing = [] if ok else missing
    if not ok:
        all_ok = False
        st.error(f"Kolom wajib pada {key} belum lengkap. Missing: {', '.join(missing)}")

if not all_ok:
    st.stop()

scored = compute_scores(saham_df, broker_df, perdagangan_df, daftar_df)
market_regime = None

if indeks_file is not None:
    try:
        indeks_df = load_table(indeks_file)
        ok_idx, missing_idx = validate_columns(indeks_df, INDEX_REQUIRED_COLUMNS)
        if not ok_idx:
            st.warning(f"Kolom Ringkasan Indeks belum lengkap. Missing: {', '.join(missing_idx)}")
        else:
            market_regime = compute_market_regime(indeks_df)
            scored["final_score"] = (scored["final_score"] * 0.90) + (market_regime["regime_score"] * 0.10)
    except Exception as exc:
        st.warning(f"Ringkasan Indeks tidak terbaca: {exc}")

if has_columns(daftar_df, ALTERNATE_COLUMNS["daftar_saham"]):
    st.info(
        "File daftar saham terdeteksi dalam format alternatif (ID Instrument/ID Board). "
        "Metadata papan tetap dipakai, tetapi nama perusahaan/tanggal pencatatan dari daftar saham tidak tersedia."
    )

if market_regime is not None:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Index", str(market_regime["index_code"]))
    r2.metric("Index Change %", f"{market_regime['index_change_pct']:.2f}%")
    r3.metric("Regime", str(market_regime["regime_label"]))
    r4.metric("Regime Score", f"{market_regime['regime_score']:.2f}")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    min_score = st.slider("Minimum Final Score", 0, 100, 60)
with col2:
    papan_options = sorted([x for x in scored["papan pencatatan"].dropna().unique().tolist() if str(x).strip()])
    papan_filter = st.selectbox("Filter Papan", ["Semua"] + papan_options)
with col3:
    top_n = st.number_input("Top N", min_value=5, max_value=300, value=25, step=5)

view = scored[scored["final_score"] >= min_score].copy()
if papan_filter != "Semua":
    view = view[view["papan pencatatan"] == papan_filter]
view = view.head(int(top_n))

st.subheader("Kandidat Saham Potensi")
st.dataframe(
    view[
        [
            "kode saham",
            "nama perusahaan",
            "papan pencatatan",
            "penutupan_num",
            "change_pct",
            "foreign_net",
            "final_score",
            "kategori",
        ]
    ].rename(columns={"penutupan_num": "close", "foreign_net": "net_foreign"}),
    use_container_width=True,
)

if view.empty:
    st.warning("Belum ada kandidat sesuai filter.")
    st.stop()

selected_code = st.selectbox("Pilih saham kandidat", options=view["kode saham"].tolist())
selected_row = scored[scored["kode saham"] == selected_code].iloc[0]

st.markdown("---")
st.subheader(f"Detail Kandidat: {selected_code}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Final Score", f"{selected_row['final_score']:.2f}")
m2.metric("Kategori", str(selected_row["kategori"]))
m3.metric("Change %", f"{selected_row['change_pct']:.2f}%")
m4.metric("Net Foreign", f"{selected_row['foreign_net']:,.0f}")

st.write("Breakdown score:")
st.dataframe(
    pd.DataFrame(
        {
            "faktor": ["Momentum", "Likuiditas", "Flow", "Market Activity", "Broker"],
            "skor": [
                selected_row["momentum_score"],
                selected_row["liquidity_score"],
                selected_row["flow_score"],
                selected_row["market_activity_score"],
                selected_row["broker_score"],
            ],
        }
    ),
    use_container_width=True,
)

analysis = generate_detailed_analysis(selected_row, scored, market_regime)

st.markdown("### Analisis Detail")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Universe Rank", f"{analysis['final_rank_pct']:.1f}%")
d2.metric("Conviction Score", f"{analysis['conviction_score']:.1f}")
d3.metric("Risk Score", f"{analysis['risk_score']:.1f}")
d4.metric("Profil", str(analysis["rr_profile"]))

a1, a2 = st.columns(2)
with a1:
    st.write("Kekuatan utama:")
    for item in analysis["strengths"]:
        st.write(f"- {item}")
    st.write("Titik lemah:")
    for item in analysis["weaknesses"]:
        st.write(f"- {item}")

with a2:
    st.write("Risiko yang perlu diperhatikan:")
    for item in analysis["risks"]:
        st.write(f"- {item}")
    st.write("Rencana aksi:")
    st.info(str(analysis["action"]))

st.write("Rencana level eksekusi (berbasis close EOD):")
plan_cols = st.columns(5)
plan_cols[0].metric("Entry Low", f"{analysis['entry_zone_low']:.2f}" if pd.notna(analysis["entry_zone_low"]) else "-")
plan_cols[1].metric("Entry High", f"{analysis['entry_zone_high']:.2f}" if pd.notna(analysis["entry_zone_high"]) else "-")
plan_cols[2].metric("Invalidation", f"{analysis['invalidation']:.2f}" if pd.notna(analysis["invalidation"]) else "-")
plan_cols[3].metric("TP1", f"{analysis['tp1']:.2f}" if pd.notna(analysis["tp1"]) else "-")
plan_cols[4].metric("TP2", f"{analysis['tp2']:.2f}" if pd.notna(analysis["tp2"]) else "-")

rank_view = pd.DataFrame(
    {
        "indikator": ["Final Rank", "Momentum Rank", "Flow Rank", "Likuiditas Rank", "Consistency"],
        "nilai": [
            analysis["final_rank_pct"],
            analysis["momentum_rank_pct"],
            analysis["flow_rank_pct"],
            analysis["liq_rank_pct"],
            analysis["consistency_score"],
        ],
    }
)
st.dataframe(rank_view, use_container_width=True)

factor_view = pd.DataFrame(
    {
        "faktor": list(analysis["factors"].keys()),
        "score": list(analysis["factors"].values()),
        "status": [factor_label(v) for v in analysis["factors"].values()],
    }
).sort_values("score", ascending=False)
st.dataframe(factor_view, use_container_width=True)

st.markdown("### 3A. TradingView Dinamis")
st.caption("TradingView ditampilkan langsung di sistem (embed), bukan redirect.")

tv_code = st.text_input("Input kode saham IDX", value=selected_code, key="tv_code_input").strip().upper()
if not tv_code:
    tv_code = selected_code

tv_symbol = f"IDX:{tv_code}"
tv_symbol_slug = f"IDX-{tv_code}"
tv_link = f"https://id.tradingview.com/chart/O3oGhaCr/?symbol=IDX%3A{tv_code}"
tv_financial_url = f"https://id.tradingview.com/symbols/{tv_symbol_slug}/financials-overview/"
tv_news_url = f"https://id.tradingview.com/symbols/{tv_symbol_slug}/news/"
tv_technicals_url = f"https://id.tradingview.com/symbols/{tv_symbol_slug}/technicals/"
tv_etf_url = f"https://id.tradingview.com/symbols/{tv_symbol_slug}/etfs/"

info_col, link_col = st.columns([1, 1])
with info_col:
    st.text_input("Symbol aktif", value=tv_symbol, disabled=True)
with link_col:
    st.link_button("Link Resmi Chart", tv_link)

tab_chart, tab_fin, tab_news, tab_tech, tab_etf = st.tabs(["Chart", "Financials", "News", "Technicals", "ETFs"])

with tab_chart:
    widget_config = {
        "autosize": True,
        "symbol": tv_symbol,
        "interval": "D",
        "timezone": "Asia/Jakarta",
        "theme": "dark",
        "style": "1",
        "locale": "id",
        "allow_symbol_change": True,
        "calendar": False,
        "support_host": "https://www.tradingview.com",
    }
    widget_html = f"""
    <div class=\"tradingview-widget-container\" style=\"height:560px;width:100%;\">
      <div class=\"tradingview-widget-container__widget\" style=\"height:calc(100% - 32px);width:100%;\"></div>
      <script type=\"text/javascript\" src=\"https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js\" async>
      {json.dumps(widget_config)}
      </script>
    </div>
    """
    components.html(widget_html, height=580)

with tab_fin:
    st.caption("Financials overview")
    components.iframe(tv_financial_url, height=700, scrolling=True)

with tab_news:
    news_widget = {
        "feedMode": "symbol",
        "symbol": tv_symbol,
        "isTransparent": False,
        "displayMode": "regular",
        "width": "100%",
        "height": 700,
        "colorTheme": "dark",
        "locale": "id",
    }
    news_html = f"""
    <div class=\"tradingview-widget-container\" style=\"width:100%;height:700px;\">
      <div class=\"tradingview-widget-container__widget\" style=\"width:100%;height:100%;\"></div>
      <script type=\"text/javascript\" src=\"https://s3.tradingview.com/external-embedding/embed-widget-timeline.js\" async>
      {json.dumps(news_widget)}
      </script>
    </div>
    """
    components.html(news_html, height=720)

with tab_tech:
    tech_widget = {
        "interval": "1D",
        "width": "100%",
        "isTransparent": False,
        "height": 700,
        "symbol": tv_symbol,
        "showIntervalTabs": True,
        "displayMode": "single",
        "locale": "id",
        "colorTheme": "dark",
    }
    tech_html = f"""
    <div class=\"tradingview-widget-container\" style=\"width:100%;height:700px;\">
      <div class=\"tradingview-widget-container__widget\" style=\"width:100%;height:100%;\"></div>
      <script type=\"text/javascript\" src=\"https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js\" async>
      {json.dumps(tech_widget)}
      </script>
    </div>
    """
    components.html(tech_html, height=720)

with tab_etf:
    st.caption("ETF references untuk symbol terkait")
    components.iframe(tv_etf_url, height=700, scrolling=True)
