"""Structured OpenRouter analysis for BEI and TradingView data."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import requests

from .schema import ANALYSIS_SCHEMA


SYSTEM_PROMPT = """Anda adalah AI equity research dan technical analyst untuk saham Bursa Efek Indonesia.

Gunakan hanya data yang diberikan aplikasi. Data tersebut adalah fakta yang harus dijelaskan berdasarkan sumber dan cakupan datanya. Anda tidak membaca laporan PDF/Excel dan tidak boleh menggunakan data eksternal yang tidak diberikan.

Aturan wajib:
- Jangan mengarang harga, candle, volume, indikator, tanggal, support, resistance, Fibonacci, atau target.
- Bedakan REPORTED (langsung dari data), CALCULATED (hasil formula), INTERPRETED (kesimpulan), dan MISSING.
- Jika histori kurang, tulis insufficient_data. Jangan menyebut EMA20/50/200, RSI14, atau ATR14 valid jika observasinya belum cukup.
- TradingView adalah sumber snapshot teknikal. Jika konteks menyatakan `auto_tradingview`, jangan mengklaim tersedia foreign flow, order book, atau histori candle lengkap kecuali field GOAPI yang sesuai memang tersedia. Broker summary hanya boleh dipakai jika `goapi_broker_available` bernilai true.
- Jika terdapat data BEI dan TradingView sekaligus, jelaskan perbedaan sumbernya dan jangan memilih diam-diam ketika nilainya berbeda.
- Level entry, stop, TP, support, resistance, dan Fibonacci hanya boleh memakai kandidat yang ada di quant_context atau dihitung dari OHLC yang diberikan.
- Drawing adalah instruksi visual saja. Setiap drawing wajib menyertakan alasan dan evidence. Jangan membuat drawing jika data tidak cukup.
- Skenario adalah kondisi bersyarat, bukan janji keuntungan. Kesimpulan boleh WAIT atau INSUFFICIENT_DATA.
- Tulis dalam bahasa Indonesia profesional, objektif, dan ringkas.
- Kembalikan JSON yang mengikuti schema, tanpa markdown fence dan tanpa teks tambahan."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def build_runtime_context(ticker: str, company: str, bei: dict[str, Any], tradingview: dict[str, Any] | None,
                          quant: dict[str, Any], market_regime: dict[str, Any] | None = None,
                          multi_day: dict[str, Any] | None = None) -> dict[str, Any]:
    multi_day = multi_day or {}
    is_auto = multi_day.get("mode") == "auto_tradingview"
    has_goapi_history = bool(multi_day.get("goapi_available"))
    if is_auto and has_goapi_history:
        data_scope = "GOAPI historical OHLC + TradingView scanner snapshot"
    elif is_auto:
        data_scope = "TradingView scanner snapshot only; no full OHLC history"
    else:
        data_scope = "BEI upload + TradingView scanner snapshot"
    return _jsonable({
        "ticker": ticker,
        "company": company,
        "data_scope": data_scope,
        "bei_snapshot": bei,
        "tradingview_snapshot": tradingview or {},
        "quant_context": quant,
        "market_regime": market_regime or {},
        "multi_day_context": multi_day,
    })


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Analisis data berikut dan kembalikan schema JSON yang diminta:\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
    ]


def _extract_json(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    if not isinstance(content, str):
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None


def call_openrouter_structured(context: dict[str, Any], api_key: str, model: str, max_tokens: int = 3200) -> dict[str, Any]:
    """Call OpenRouter with JSON schema and return a stable result envelope."""

    if not api_key:
        return {"ok": False, "error": "API key OpenRouter belum diisi."}
    payload = {
        "model": model,
        "messages": build_messages(context),
        "temperature": 0.15,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "bei_stock_analysis", "strict": True, "schema": ANALYSIS_SCHEMA},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sahambei.streamlit.app",
        "X-Title": "BEI Stock Screener Structured Analyst",
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=90)
        body = response.json() if response.content else {}
        if not response.ok:
            message = body.get("error", {}).get("message") if isinstance(body, dict) else None
            return {"ok": False, "error": f"OpenRouter HTTP {response.status_code}: {message or 'request ditolak'}"}
        message = ((body.get("choices") or [{}])[0].get("message") or {})
        parsed = _extract_json(message.get("content"))
        if parsed is None:
            return {"ok": False, "error": "Model tidak mengembalikan JSON valid. Pilih model OpenRouter yang mendukung structured output."}
        ticker = str(context.get("ticker", "")).upper()
        if str(parsed.get("ticker", "")).upper() != ticker:
            return {"ok": False, "error": "JSON AI tidak cocok dengan ticker yang dianalisis."}
        return {"ok": True, "data": parsed}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "OpenRouter timeout setelah 90 detik. Coba model yang lebih ringan atau ulangi."}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "error": f"Gagal terhubung ke OpenRouter: {exc}"}
    except (ValueError, TypeError) as exc:
        return {"ok": False, "error": f"Respons OpenRouter tidak dapat dibaca: {exc}"}
