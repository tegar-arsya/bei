"""Small strict schema shared by the OpenRouter request and UI validator."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "as_of_date": {"type": "string"},
        "verdict": {"type": "string", "enum": ["BUY_ON_PULLBACK", "BREAKOUT_WATCH", "WAIT", "AVOID", "INSUFFICIENT_DATA"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "summary": {"type": "string"},
        "data_quality": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_DATA"]},
                "explanation": {"type": "string"},
                "missing": {"type": "array", "items": {"type": "string"}},
                "conflicts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "explanation", "missing", "conflicts"],
            "additionalProperties": False,
        },
        "market_structure": {
            "type": "object",
            "properties": {
                "trend": {"type": "string"},
                "momentum": {"type": "string"},
                "flow": {"type": "string"},
                "technical": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["trend", "momentum", "flow", "technical", "evidence"],
            "additionalProperties": False,
        },
        "levels": {
            "type": "object",
            "properties": {
                "support": {"type": "array", "items": {"type": "number"}},
                "resistance": {"type": "array", "items": {"type": "number"}},
                "entry_low": {"type": "number"},
                "entry_high": {"type": "number"},
                "stop": {"type": "number"},
                "tp1": {"type": "number"},
                "tp2": {"type": "number"},
            },
            "required": ["support", "resistance", "entry_low", "entry_high", "stop", "tp1", "tp2"],
            "additionalProperties": False,
        },
        "scenarios": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["BEARISH", "BASE", "BULLISH"]},
                    "condition": {"type": "string"},
                    "trigger": {"type": "string"},
                    "invalidation": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 100},
                    "action": {"type": "string"},
                },
                "required": ["name", "condition", "trigger", "invalidation", "probability", "action"],
                "additionalProperties": False,
            },
        },
        "drawings": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["horizontal_line", "trendline", "rectangle", "label"]},
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "x_start": {"type": "string"},
                    "x_end": {"type": "string"},
                    "y_start": {"type": "number"},
                    "y_end": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "name", "purpose", "x_start", "x_end", "y_start", "y_end", "evidence"],
                "additionalProperties": False,
            },
        },
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "monitoring_triggers": {"type": "array", "items": {"type": "string"}},
        "conclusion": {"type": "string"},
        "disclaimer": {"type": "string"},
    },
    "required": ["ticker", "as_of_date", "verdict", "confidence", "summary", "data_quality", "market_structure", "levels", "scenarios", "drawings", "risk_flags", "monitoring_triggers", "conclusion", "disclaimer"],
    "additionalProperties": False,
}

