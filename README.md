# Saham BEI Screener v3

Aplikasi ini menggabungkan:

- Scoring BEI dari file upload (1-5 hari)
- Data teknikal TradingView (scanner)
- Narasi otomatis dengan OpenRouter (opsional, pakai API key)

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

Untuk fitur AI narasi, isi OpenRouter API key di sidebar (didapat dari OpenRouter).

Jika ingin tanpa ketik manual, set API key statis lewat secrets atau environment variable:

```toml
# .streamlit/secrets.toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openrouter/auto"
```

Atau via environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="openrouter/auto"
```

Jika key statis tersedia, app otomatis memakainya di sidebar.

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
- Jika TradingView gagal diambil, app tetap jalan dengan analisis BEI saja.
- Output AI bersifat asisten analisis, bukan rekomendasi investasi final.
