# Saham BEI Screener (4 File)

Aplikasi ini sekarang fokus ke mode ringkas:

- Hanya pakai 4 file data BEI
- Tanpa orderbook
- Tanpa broksum manual
- Tanpa input fundamental manual

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

## Catatan format

- `Ringkasan Saham` mengikuti kolom IDX harian.
- `Ringkasan Broker` dipakai sebagai komponen broker activity. Jika kode broker tidak map ke ticker, skornya dibuat netral.
- `Daftar Saham` mendukung dua format:
  - Utama: `Kode`, `Nama Perusahaan`, `Tanggal Pencatatan`, `Saham`, `Papan Pencatatan`
  - Alternatif: `ID Instrument`, `ID Board`, `Volume`, `Nilai`, `Frekuensi`

## Formula skor

- `Final Score = 0.30 * Momentum + 0.30 * Likuiditas + 0.20 * Flow + 0.15 * Market Activity + 0.05 * Broker`

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

## Rencana tambahan

Jika ingin ditambah nanti, file `Ringkasan Indeks` bisa dijadikan modul tambahan untuk filter kondisi market (misalnya risk-on/risk-off).
