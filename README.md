# Klasifikasi Keparahan Kecelakaan Lalu Lintas — Streamlit App

Hasil deployment dari notebook `Untitled8.ipynb` (Decision Tree, klasifikasi
kecelakaan **Berat** vs **Tidak Berat**).

## Struktur File

```
streamlit_app/
├── app.py            # Aplikasi utama Streamlit (UI: training & prediksi)
├── preprocessing.py  # Semua fungsi feature engineering (replikasi notebook)
└── requirements.txt  # Daftar dependency
```

## Cara Menjalankan di Komputer Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Lalu buka browser ke `http://localhost:8501`.

## Cara Deploy ke Streamlit Community Cloud (gratis)

1. Buat repository GitHub baru, lalu unggah ketiga file di atas (`app.py`,
   `preprocessing.py`, `requirements.txt`) ke repo tersebut.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **"New app"**, pilih repository dan branch yang sesuai.
4. Isi **Main file path** dengan `app.py`.
5. Klik **Deploy**. Tunggu beberapa menit hingga aplikasi siap.

## Cara Pakai Aplikasi

1. **Tab "Latih Model"** — unggah file Excel data kecelakaan mentah (format
   kolom harus sama seperti data sumber di notebook). Klik "Mulai Training".
   Model, metrik evaluasi, dan feature importance akan langsung ditampilkan,
   serta bisa diunduh sebagai file `.joblib`.
2. **Tab "Prediksi"** — gunakan model yang baru dilatih, atau unggah file
   `.joblib` model lama. Lalu isi form data kasus baru untuk prediksi
   satu-per-satu, atau unggah file Excel untuk prediksi banyak data sekaligus
   (batch), hasilnya bisa diunduh sebagai CSV.
3. **Tab "Tentang"** — penjelasan pipeline & daftar kolom wajib pada file Excel.

## Kolom Wajib pada File Excel Sumber

```
KORBAN MD, KORBAN LB, KORBAN LR, HARI / TGL / JAM KEJADIAN,
IDENTITAS TERSANGKA, IDENTITAS KORBAN, TYPE KECELAKAAN,
MATERI, TKP, TAHUN DATA
```

## Catatan Teknis

- Logika preprocessing di `preprocessing.py` adalah replikasi 1:1 dari
  notebook (kategori umur, ekstraksi jenis kelamin/pendidikan/pekerjaan,
  klasifikasi jenis & risiko tabrakan, kategori kerugian, clustering TKP, dll).
- Model & daftar kolom hasil one-hot encoding disimpan bersama dalam satu
  file `.joblib` (`model`, `train_columns`, `top_tkp`) agar prediksi data
  baru selalu konsisten dengan struktur kolom saat training.
- Tidak ada lagi dependensi `google.colab` — upload file ditangani lewat
  `st.file_uploader`.
