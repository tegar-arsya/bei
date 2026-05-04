# Streamlit Deployment Checklist

## Required files

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `.env.example`
- `README.md`

## Do not commit

- `.env`
- `.streamlit/secrets.toml`
- `data/`

## Streamlit Cloud secrets

Add these in the Streamlit Cloud app settings:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
```

## Data persistence

The app stores watchlist, alert rules, and history in local JSON files under `data/`.
This is fine for local use, but cloud filesystem storage should not be treated as permanent.

Before redeploying or restarting a cloud app, use:

`Watchlist & Alerts` -> `Backup / Restore Data Lokal` -> `Export Backup JSON`

To restore, upload the JSON backup from the same panel.
