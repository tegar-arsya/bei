# Market Screener

Aplikasi ini punya beberapa mode:

1. Saham BEI
2. Crypto Market
3. Meme Coin Radar
4. Watchlist & Alerts

Mode Saham BEI menggabungkan:

- Scoring BEI dari file upload (1-5 hari)
- Data teknikal TradingView (scanner)
- Narasi otomatis dengan OpenRouter (opsional, pakai API key)

Mode Crypto Market memakai data gratis dari Indodax public API untuk pair crypto IDR yang tersedia di market Indonesia. CoinGecko IDR tersedia sebagai pembanding harga Rupiah global, bukan bukti listing lokal. Detail coin terpilih punya tab `Outlook` untuk forward score, range 24h, scenario map, dan checklist praktis.

Mode Meme Coin Radar menampilkan meme coin yang punya pair IDR di Indodax, seperti DOGE, SHIB, PEPE, FLOKI, BONK, WIF, dan sejenisnya jika tersedia. Modul ini tidak memakai pair DEX global.

Mode Watchlist & Alerts menyimpan kandidat crypto ke `data/`, refresh snapshot, menyimpan history score, dan menampilkan alert sederhana.

## Data yang dibutuhkan

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

- `Saham BEI` untuk workflow lama berbasis upload data BEI
- `Crypto Market` untuk coin besar/altcoin dengan pair IDR di Indonesia
- `Meme Coin Radar` untuk meme coin dengan pair IDR di Indonesia
- `Watchlist & Alerts` untuk memantau kandidat yang disimpan

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
- Data Crypto Market dan Meme Coin Radar tergantung Indodax public API. CoinGecko IDR hanya pembanding harga Rupiah global.
- Watchlist, alert rules, dan history tersimpan lokal di folder `data/`.
- Jika TradingView gagal diambil, app tetap jalan dengan analisis BEI saja.
- Output AI bersifat asisten analisis, bukan rekomendasi investasi final.
- Meme coin tetap sangat spekulatif walaupun sudah punya pair IDR.
