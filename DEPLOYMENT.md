# Streamlit Deployment Checklist — BEI Stock Screener

## Required files

- app.py
- requirements.txt
- .streamlit/config.toml
- .streamlit/secrets.toml.example
- .env.example
- README.md

## Do not commit

- .env
- .streamlit/secrets.toml
- data/

## Streamlit Cloud secrets

~~~toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
~~~

## Data persistence

Watchlist saham, alert rules, dan history disimpan sebagai JSON di folder data/.

Untuk VPS/Docker, gunakan volume persisten, misalnya:

~~~yaml
volumes:
  - ./data:/app/data
~~~

Untuk Streamlit Cloud, filesystem tidak boleh dianggap database permanen. Gunakan Watchlist & Alerts lalu Backup / Restore Data Saham sebelum redeploy.

## Cek setelah deploy

1. Buka halaman Home.
2. Buka panel Deploy Readiness.
3. Pastikan folder data/ dapat ditulis.
4. Uji Auto TradingView.
5. Uji Upload BEI Advanced dengan satu set file.
6. Uji export backup watchlist.
