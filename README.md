# BEI Stock Screener

Aplikasi Streamlit khusus untuk screening dan analisis saham BEI.

## Flow aplikasi

~~~text
TradingView scanner / upload BEI
            ↓
Normalisasi data
            ↓
Indikator dan scoring
            ↓
Filter kandidat saham
            ↓
Chart, teknikal, flow, broker, dan AI
            ↓
Watchlist, history, dan alert saham
~~~

## Mode yang tersedia

1. Auto TradingView
   - Scan saham IDX tanpa upload file.
   - Mengambil harga, volume, value traded, market cap, RSI, MACD, EMA, Bollinger, ADX, relative volume, performa mingguan/bulanan, dan rekomendasi TradingView.
   - Jika `GOAPI_API_KEY` tersedia, quote, historical OHLC, dan broker summary saham terpilih diambil dari GOAPI untuk quant engine dan AI; TradingView tetap menjadi chart interaktif.
   - Foreign flow dan order book tidak tersedia; broker summary hanya tersedia jika endpoint GOAPI trial Anda mengizinkannya.
   - Hasil scanner dapat diunduh sebagai Excel dengan seluruh row/kolom, header freeze, dan filter.

2. Upload BEI Advanced
   - Upload data BEI untuk foreign flow, bid-offer pressure, broker activity, market activity, dan analisis 1-5 hari.
   - Ringkasan indeks dapat digunakan untuk menghitung market regime.

3. Watchlist & Alerts
   - Menyimpan saham pilihan ke folder data/.
   - Refresh snapshot saham dari TradingView.
   - Menyimpan history score.
   - Menampilkan alert score, perubahan harga, relative volume, dan breakout candidate.
   - Memiliki backup/restore JSON.

## Menjalankan aplikasi

~~~bash
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~

Jika environment belum ada:

~~~bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~

Buka:

~~~text
http://localhost:8501
~~~

## AI opsional

AI memakai OpenRouter. Key dapat dimasukkan dari sidebar atau diset melalui .env:

~~~bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openrouter/auto
~~~

Pada Upload BEI Advanced, AI menerima data BEI dan snapshot indikator TradingView,
lalu meminta respons `json_schema`. Respons divalidasi sebelum ditampilkan sebagai
kartu, skenario, level, drawing, dan AI Quant Chart. Bagian `AI Analisis` juga
menampilkan chart candlestick berbasis OHLC, volume, EMA, support/resistance,
Fibonacci, level eksekusi, serta drawing AI yang tervalidasi. Tersedia juga
download detail analisis ke Excel dengan sheet ringkasan, quant indicators,
levels, scenarios, evidence, drawings, dan data chart.

Perhitungan quant dan integrasi AI dipisahkan dari `app.py`:

- `analytics/quant.py`: indikator, swing, Fibonacci, support/resistance, dan level proxy.
- `analytics/chart.py`: Plotly candlestick, volume, EMA, level, dan drawing tervalidasi.
- `analytics/ai.py`: system prompt profesional dan OpenRouter structured output.
- `analytics/validator.py`: menolak drawing atau level yang tidak konsisten dengan OHLC.
- `storage/persistence.py`: menyimpan hasil ke JSON lokal dan opsional mirror ke Supabase.

Supabase tidak wajib untuk menjalankan aplikasi. Jika `SUPABASE_URL` dan key sudah
diisi, hasil analisis dicoba disimpan ke tabel `ai_analyses`; kegagalan Supabase tidak
menghentikan analisis lokal. Modul ini tidak menghapus atau mengubah tabel lama.

## Supabase untuk upload BEI

1. Jalankan `supabase/migrations/20260723_bei_streamlit_imports.sql` di Supabase SQL Editor.
   Jika tabel sudah terlanjur dibuat tetapi muncul `permission denied`, jalankan juga
   `supabase/migrations/20260723_bei_streamlit_imports_permissions.sql`.
2. Isi `SUPABASE_URL` dan server-only `SUPABASE_SERVICE_ROLE_KEY` atau `SUPABASE_SECRET_KEY`.
3. Pastikan bucket private `saham` tersedia; migrasi akan membuatnya jika belum ada.
4. Di halaman Upload BEI Advanced, pilih tanggal terlebih dahulu lalu upload Ringkasan Saham,
   Ringkasan Broker, Ringkasan Perdagangan, dan Daftar Saham.
5. Klik `Simpan batch upload ke Supabase`.

Satu tanggal adalah satu batch. Upload ulang pada tanggal yang sama akan mengganti baris
untuk tanggal tersebut dan meng-upsert file di bucket `saham`; tanggal lain tidak disentuh.
Gunakan service-role/secret key hanya di server atau Streamlit Secrets, jangan di frontend.

Jika sudah ada data tersimpan, halaman Advanced otomatis menawarkan mode `Database Supabase`.
Pilih satu sampai lima tanggal dari database untuk menjalankan scoring tanpa upload ulang file.
Mode `Upload file baru` tetap tersedia untuk menambah atau memperbarui batch.

Alternatif Streamlit secrets:

~~~toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
~~~

## GOAPI IDX (opsional)

Tambahkan di `.streamlit/secrets.toml` atau environment server:

~~~toml
GOAPI_API_KEY = "API_KEY_GOAPI_ANDA"
GOAPI_BASE_URL = "https://api.goapi.io"
~~~

Key dibaca server-side melalui header `X-API-KEY` dan tidak dikirim ke browser. Mode Auto tetap berjalan dengan
TradingView jika GOAPI belum diatur atau trial sedang tidak memiliki akses. GOAPI
digunakan untuk quote, historical OHLC, dan perhitungan quant pada saham terpilih;
chart tetap memakai TradingView agar drawing dan Fibonacci tersedia.

## Upload BEI Advanced

Untuk setiap hari, siapkan:

1. Ringkasan saham
2. Ringkasan broker
3. Ringkasan perdagangan
4. Daftar saham
5. Ringkasan indeks (opsional)

Format file yang didukung: xlsx, xls, dan csv.

Format utama Daftar Saham:

- Kode
- Nama Perusahaan
- Tanggal Pencatatan
- Saham
- Papan Pencatatan

Format alternatif:

- ID Instrument
- ID Board
- Volume
- Nilai
- Frekuensi

## Formula scoring Advanced

Default bobot:

- Momentum: 25%
- Likuiditas: 20%
- Flow: 20%
- Market Activity: 15%
- Volume Trend: 10%
- Price Structure: 5%
- Broker: 5%

Jika ringkasan indeks diupload:

~~~text
Final Score Adjusted = 0.90 × Final Score + 0.10 × Regime Score
~~~

Kategori score:

- Rendah
- Menarik
- Tinggi
- Sangat Tinggi

## Penyimpanan lokal

File runtime dibuat di folder data/:

- stock_watchlist.json
- stock_history.json
- stock_alert_rules.json

Untuk Streamlit Cloud, storage lokal tidak permanen. Gunakan fitur backup/restore sebelum redeploy. Pada VPS, mount folder data/ sebagai Docker volume.

## Deploy

1. Siapkan app.py, requirements.txt, .streamlit/config.toml, .env.example, dan README.
2. Jangan commit .env, .streamlit/secrets.toml, atau folder data/.
3. Isi OPENROUTER_API_KEY dan OPENROUTER_MODEL melalui Secrets jika memakai Streamlit Cloud.
4. Jika memakai Docker/VPS, pastikan folder data/ dipersistenkan dengan volume.

## Catatan

- Endpoint scanner TradingView adalah endpoint publik dan dapat berubah atau terkena rate limit.
- Chart, financials, technical analysis, dan news ditampilkan melalui widget TradingView.
- Aplikasi tidak mengirim order ke broker.
- Output AI adalah alat bantu analisis, bukan rekomendasi investasi final.
