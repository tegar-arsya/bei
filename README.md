# Market Screener

Aplikasi ini punya beberapa mode:

1. Saham BEI
2. Crypto Market
3. Meme Coin Radar
4. Watchlist & Alerts

Mode Saham BEI menggabungkan:

- Auto scanner TradingView untuk saham IDX tanpa upload file
- Upload BEI advanced untuk foreign flow dan broker summary (1-5 hari)
- Data teknikal TradingView (scanner/chart)
- Narasi otomatis dengan OpenRouter (opsional, pakai API key)

Mode Crypto Market memakai data gratis dari Indodax public API untuk pair crypto IDR yang tersedia di market Indonesia. CoinGecko IDR tersedia sebagai pembanding harga Rupiah global, bukan bukti listing lokal. Detail coin terpilih punya tab `Outlook`, chart TradingView, data OHLCV/indikator chart untuk AI, news, dan community check.

Mode Meme Coin Radar menampilkan meme coin yang punya pair IDR di Indodax, seperti DOGE, SHIB, PEPE, FLOKI, BONK, WIF, dan sejenisnya jika tersedia. Modul ini tidak memakai pair DEX global. AI membaca market IDR, OHLCV chart Indodax, berita terbaru, dan ukuran komunitas jika metadata CoinGecko tersedia.

Mode Watchlist & Alerts menyimpan kandidat saham IDX dan crypto IDR ke `data/`, refresh snapshot, menyimpan history score, dan menampilkan alert sederhana.

## Data yang dibutuhkan

Mode Saham BEI default tidak membutuhkan upload file karena memakai TradingView scanner otomatis.

Untuk mode `Upload BEI Advanced`, siapkan:

1. Ringkasan saham
2. Ringkasan broker
3. Ringkasan perdagangan dan rekapitulasi
4. Daftar saham
5. (Opsional) Ringkasan indeks

Format file: `xlsx`, `xls`, atau `csv`.

## Menjalankan aplikasi

```bash
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run app.py
```

Lalu buka di browser:

```bash
http://localhost:8501
```

Di halaman awal, pilih mode:

- `Saham BEI` untuk auto scanner saham IDX tanpa upload, plus upload BEI advanced
- `Crypto Market` untuk coin besar/altcoin dengan pair IDR di Indonesia
- `Meme Coin Radar` untuk meme coin dengan pair IDR di Indonesia
- `Watchlist & Alerts` untuk memantau kandidat saham/crypto yang disimpan

Untuk fitur AI narasi, isi OpenRouter API key di sidebar (didapat dari OpenRouter).

Jika ingin tanpa ketik manual, set API key statis lewat `.env`, Streamlit secrets, atau environment variable.

Opsi paling simpel untuk lokal adalah `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openrouter/auto
```

File `.env` sudah masuk `.gitignore`. Gunakan `.env.example` sebagai contoh format.

Alternatif Streamlit secrets:

```toml
# .streamlit/secrets.toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
```

Atau environment variable shell:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="openrouter/auto"
```

Jika key statis tersedia, app otomatis memakainya di sidebar.

## Deploy ke Streamlit Cloud

Checklist sebelum deploy:

1. Push file ini ke repo: `app.py`, `requirements.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `.env.example`, dan README.
2. Jangan push `.env`, `.streamlit/secrets.toml`, atau folder `data/`.
3. Di Streamlit Cloud, pilih repo, branch, dan main file `app.py`.
4. Isi Secrets di dashboard Streamlit:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
```

5. Setelah app hidup, buka halaman `Home` lalu cek panel `Deploy Readiness`.
6. Jika memakai Watchlist & Alerts, export backup JSON secara berkala dari halaman `Watchlist & Alerts`.

Catatan penting deploy:

- `.env` hanya untuk lokal. Streamlit Cloud memakai Secrets dashboard.
- `data/` dipakai untuk watchlist/history lokal, tapi jangan dianggap database permanen di cloud.
- Untuk penyimpanan permanen lintas redeploy, gunakan backup/restore JSON atau nanti sambungkan database eksternal seperti Supabase/Firebase.

## Catatan format

- `Ringkasan Saham` mengikuti kolom IDX harian.
- `Ringkasan Broker` dipakai sebagai komponen broker activity. Jika kode broker tidak map ke ticker, skornya dibuat netral.
- `Daftar Saham` mendukung dua format:
  - Utama: `Kode`, `Nama Perusahaan`, `Tanggal Pencatatan`, `Saham`, `Papan Pencatatan`
  - Alternatif: `ID Instrument`, `ID Board`, `Volume`, `Nilai`, `Frekuensi`

## Formula skor inti

Final score menggunakan bobot yang bisa diubah di sidebar, default:

- Momentum: 25%
- Likuiditas: 20%
- Flow: 20%
- Market Activity: 15%
- Volume Trend: 10%
- Price Structure: 5%
- Broker: 5%

Jika `Ringkasan Indeks` diupload:

- Sistem menghitung `Regime Score` market (Risk-On / Netral / Risk-Off)
- Penyesuaian akhir: `Final Score Adjusted = 0.90 * Final Score + 0.10 * Regime Score`

Format minimum `Ringkasan Indeks`:

- `No`
- `Kode Indeks`
- `Sebelumnya`
- `Tertinggi`
- `Terendah`
- `Penutupan`
- `Selisih`
- `Volume`
- `Nilai`
- `Frekuensi`

Kolom lain seperti `# Stock` dan `Kapitalisasi Pasar*` boleh ada (opsional).

Kategori:

- `Rendah`
- `Menarik`
- `Tinggi`
- `Sangat Tinggi`

## Catatan penting

- Data TradingView dan OpenRouter tergantung koneksi internet.
- Auto scanner saham tidak membaca net foreign dan broker summary. Gunakan `Upload BEI Advanced` jika butuh flow detail.
- Data Crypto Market dan Meme Coin Radar tergantung Indodax public API. CoinGecko IDR hanya pembanding harga Rupiah global.
- News memakai Google News RSS, sedangkan community check memakai metadata CoinGecko jika coin punya `coingecko_id`.
- Watchlist, alert rules, dan history tersimpan lokal di folder `data/`.
- Watchlist saham bisa memberi alert score, perubahan harian, dan relative volume dari snapshot TradingView.
- Jika TradingView gagal diambil, mode auto saham tidak bisa refresh; mode upload BEI tetap bisa dipakai.
- Output AI bersifat asisten analisis, bukan rekomendasi investasi final.
- Meme coin tetap sangat spekulatif walaupun sudah punya pair IDR.
