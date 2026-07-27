"""Plotly rendering for the deterministic AI chart."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - handled by the Streamlit fallback
    go = None
    make_subplots = None


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def create_ai_chart(history: pd.DataFrame, quant: dict[str, Any] | None = None, analysis: dict[str, Any] | None = None):
    """Create a candlestick chart with only validated, data-backed overlays."""

    if go is None or make_subplots is None:
        raise RuntimeError("Plotly belum terpasang. Jalankan pip install -r requirements.txt.")
    data = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
    if data.empty:
        raise ValueError("Belum ada OHLC untuk membuat AI Chart.")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    x = data["date"].astype(str).tolist() if "date" in data.columns else list(range(len(data)))
    quant = quant or {}
    indicators = quant.get("indicators", {})
    levels = quant.get("levels", {})
    execution = quant.get("execution", {})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25], subplot_titles=("AI Quant Chart", "Volume"))
    fig.add_trace(go.Candlestick(x=x, open=data["open"], high=data["high"], low=data["low"], close=data["close"], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=data["volume"], name="Volume", marker_color="#4b78a8", opacity=0.65), row=2, col=1)

    for period, color in ((20, "#f6c945"), (50, "#4dabf7"), (200, "#ff7f50")):
        column = f"ema{period}"
        if column in data.columns and data[column].notna().any():
            fig.add_trace(go.Scatter(x=x, y=data[column], mode="lines", name=f"EMA {period}", line={"color": color, "width": 1.4}), row=1, col=1)

    def add_hline(value: Any, name: str, color: str, dash: str = "dot"):
        if _finite(value):
            fig.add_hline(y=float(value), row=1, col=1, line_dash=dash, line_color=color,
                          annotation_text=f"{name} {float(value):.2f}", annotation_position="top left")

    for support in levels.get("support", [])[-4:]:
        add_hline(support, "S", "#35c48d")
    for resistance in levels.get("resistance", [])[:4]:
        add_hline(resistance, "R", "#ff6b6b")
    for item in levels.get("fibonacci", []):
        if isinstance(item, dict):
            add_hline(item.get("price"), item.get("label", "Fib"), "#b18cff", "dash")
    for key, label, color in (("entry_low", "Entry", "#2dd4bf"), ("entry_high", "Entry", "#2dd4bf"),
                               ("stop", "Stop", "#ef4444"), ("tp1", "TP1", "#22c55e"), ("tp2", "TP2", "#16a34a")):
        add_hline(execution.get(key), label, color, "solid")

    # AI drawings are applied only after the validator has removed malformed
    # or out-of-range instructions.
    for drawing in (analysis or {}).get("drawings", []):
        dtype = drawing.get("type")
        if dtype == "horizontal_line":
            add_hline(drawing.get("y_start"), drawing.get("name", "AI"), "#f59e0b", "dashdot")
        elif dtype == "trendline" and _finite(drawing.get("y_start")) and _finite(drawing.get("y_end")):
            x0 = drawing.get("x_start") or x[0]
            x1 = drawing.get("x_end") or x[-1]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[drawing["y_start"], drawing["y_end"]], mode="lines",
                                     name=f"AI: {drawing.get('name', 'Trendline')}", line={"color": "#f59e0b", "width": 2}), row=1, col=1)
        elif dtype == "rectangle" and _finite(drawing.get("y_start")) and _finite(drawing.get("y_end")):
            fig.add_shape(type="rect", x0=drawing.get("x_start") or x[0], x1=drawing.get("x_end") or x[-1],
                          y0=min(float(drawing["y_start"]), float(drawing["y_end"])),
                          y1=max(float(drawing["y_start"]), float(drawing["y_end"])),
                          line={"color": "#f59e0b", "dash": "dash"}, fillcolor="rgba(245,158,11,0.10)", row=1, col=1)
        elif dtype == "label" and _finite(drawing.get("y_start")):
            fig.add_annotation(x=drawing.get("x_start") or x[-1], y=drawing["y_start"], text=drawing.get("name", "AI"),
                               bgcolor="#f59e0b", font={"color": "#111827"}, row=1, col=1)

    fig.update_layout(height=720, template="plotly_dark", margin={"l": 20, "r": 20, "t": 55, "b": 20},
                      xaxis_rangeslider_visible=False, legend_orientation="h", legend_y=1.02,
                      hovermode="x unified")
    fig.update_yaxes(title_text="Harga", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

