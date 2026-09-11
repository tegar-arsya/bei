# Meme Coin Crypto Screener

Aplikasi Streamlit untuk screening dan analisis crypto meme coin. Mendukung **Indodax (IDR)** dan **TradingView Binance (USDT)**.

## Fitur Utama

### Sumber Data
1. **Indodax (IDR)** - Default
   - Data langsung dari API Indodax (tanpa API key).
   - Harga dalam Rupiah, cocok untuk trader Indonesia.
   - Meme coin: DOGE, SHIB, PEPE, BONK, WIF, TRUMP, PENGU, FARTCOIN, dll.
   - Teknikal: RSI, MACD, EMA dihitung dari data trades Indodax.
   - Info tambahan: spread, min order, fee per pair.

2. **TradingView Binance (USDT)**
   - Scan semua crypto pair USDT di Binance via TradingView scanner.
   - Indikator lengkap: RSI, MACD, EMA, Bollinger, ADX, Stochastic.
   - Rekomendasi TradingView (Strong Buy/Sell).

### AI Analysis (9Router)
- Menggunakan AI via 9Router endpoint untuk analisis teknikal.
- AI bertindak sebagai analis crypto swing trading.
- Output: verdict, market structure, support/resistance, scenario, danger flags.

### Watchlist
- Simpan coin pilihan ke folder `data/` lokal.
- History harga/score disimpan otomatis.

## Flow

~~~text
Pilih Sumber (Indodax/TradingView)
        ↓
Fetch data & normalisasi
        ↓
Hitung indikator teknikal & scoring
        ↓
Filter berdasarkan score & volume
        ↓
Analisis AI (opsional) + Chart
        ↓
Watchlist & history
~~~

## Menjalankan

~~~bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~

Buka: http://localhost:8501

## Konfigurasi

### 9Router (AI Analysis)

~~~bash
# .env atau .streamlit/secrets.toml
NINEROUTER_API_KEY=your-key-here
NINEROUTER_MODEL=combo-1
~~~

## Source Code

- `app.py` — UI utama Streamlit
- `analytics/indodax.py` — Data source Indodax API
- `analytics/ai.py` — AI analysis via 9Router
- `analytics/validator.py` — Validasi output AI
- `storage/persistence.py` — Simpan analisis ke file lokal
- `stock_app_disabled.py` — Code lama BEI (nonaktif)
