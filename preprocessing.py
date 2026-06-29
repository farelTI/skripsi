"""
Modul preprocessing & feature engineering.
Direplikasi 1:1 dari notebook (Untitled8.ipynb) supaya logika training
dan logika prediksi konsisten.
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fungsi-fungsi ekstraksi (sama seperti di notebook)
# ---------------------------------------------------------------------------

def ekstrak_umur(text):
    text = str(text)
    match = re.search(r"(\d+)\s*TH", text)
    if match:
        return int(match.group(1))
    return np.nan


def kategori_umur(x):
    if pd.isna(x):
        return "Unknown"
    elif x <= 17:
        return "Anak"
    elif x <= 25:
        return "Remaja"
    elif x <= 45:
        return "Dewasa"
    elif x <= 60:
        return "Pra Lansia"
    else:
        return "Lansia"


def ekstrak_jk(text):
    text = str(text)
    if ">> LK >>" in text:
        return "L"
    elif ">> PR >>" in text:
        return "P"
    return "Unknown"


def ekstrak_pendidikan(text):
    text = str(text).upper()
    daftar = ["SD", "SLTP", "SLTA", "D3", "S1", "S2"]
    for p in daftar:
        if p in text:
            return p
    return "LAINNYA"


def ekstrak_pekerjaan(text):
    text = str(text).upper()
    pekerjaan = [
        "PELAJAR", "MAHASISWA", "BURUH", "KARYAWAN",
        "WIRASWASTA", "PNS", "IRT",
    ]
    for p in pekerjaan:
        if p in text:
            return p
    return "LAINNYA"


def jenis_tabrakan(x):
    x = str(x).upper()
    if "MANUSIA" in x:
        return "TABRAK MANUSIA"
    elif "DEPAN" in x:
        return "DEPAN"
    elif "BELAKANG" in x:
        return "BELAKANG"
    elif "SAMPING" in x:
        return "SAMPING"
    elif "TUNGGAL" in x:
        return "TUNGGAL"
    else:
        return "LAINNYA"


def risiko_tabrakan(x):
    if x == "TABRAK MANUSIA":
        return "TINGGI"
    elif x == "DEPAN":
        return "TINGGI"
    elif x == "TUNGGAL":
        return "SEDANG"
    else:
        return "RENDAH"


def konversi_materi(x):
    angka = re.sub(r"[^0-9]", "", str(x))
    if angka == "":
        return 0
    return int(angka)


def kategori_kerugian(x):
    if x < 500_000:
        return "Sangat Rendah"
    elif x < 2_000_000:
        return "Rendah"
    elif x < 5_000_000:
        return "Sedang"
    elif x < 10_000_000:
        return "Tinggi"
    else:
        return "Sangat Tinggi"


def kategori_waktu(jam):
    if pd.isna(jam):
        return "Unknown"
    elif 0 <= jam < 6:
        return "Dini Hari"
    elif 6 <= jam < 12:
        return "Pagi"
    elif 12 <= jam < 18:
        return "Siang"
    else:
        return "Malam"


def jam_sibuk(jam):
    if pd.isna(jam):
        return "Normal"
    elif 6 <= jam <= 9:
        return "Pagi Sibuk"
    elif 16 <= jam <= 19:
        return "Sore Sibuk"
    else:
        return "Normal"


# Kolom fitur final yang dipakai untuk training (sama seperti notebook)
FITUR = [
    "UMUR_TERSANGKA",
    "UMUR_KORBAN",
    "KAT_UMUR_TERSANGKA",
    "KAT_UMUR_KORBAN",
    "JK_TERSANGKA",
    "JK_KORBAN",
    "PENDIDIKAN_TERSANGKA",
    "PENDIDIKAN_KORBAN",
    "PEKERJAAN_TERSANGKA",
    "PEKERJAAN_KORBAN",
    "JENIS_TABRAKAN",
    "RISIKO_TABRAKAN",
    "KATEGORI_WAKTU",
    "JAM_SIBUK",
    "WEEKEND",
    "BULAN",
    "TKP_CLUSTER",
    "KATEGORI_KERUGIAN",
    "TAHUN DATA",
]

REQUIRED_COLUMNS = [
    "KORBAN MD", "KORBAN LB", "KORBAN LR",
    "HARI / TGL / JAM KEJADIAN",
    "IDENTITAS TERSANGKA", "IDENTITAS KORBAN",
    "TYPE KECELAKAAN", "MATERI", "TKP", "TAHUN DATA",
]


def validate_columns(df: pd.DataFrame):
    """Mengecek kolom wajib ada di dataframe mentah. Mengembalikan list kolom yang hilang."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def preprocess(df_raw: pd.DataFrame, top_tkp=None):
    """
    Mengubah dataframe mentah (hasil baca Excel) menjadi dataframe
    yang sudah lengkap dengan seluruh fitur hasil engineering.

    Parameter
    ---------
    df_raw : dataframe mentah dari file excel
    top_tkp : daftar 10 TKP teratas yang dipakai saat training.
              Jika None, akan dihitung dari df_raw ini sendiri (dipakai saat training awal).
              Saat prediksi data baru, WAJIB diberikan top_tkp dari hasil training agar konsisten.

    Return
    ------
    df : dataframe dengan kolom-kolom fitur lengkap + TARGET (jika kolom KORBAN ada)
    top_tkp : daftar TKP yang dipakai (untuk disimpan bersama model)
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # --- Kolom korban & target ---
    for col in ["KORBAN MD", "KORBAN LB", "KORBAN LR"]:
        if col in df.columns:
            df[col] = df[col].replace("-", 0).fillna(0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if all(c in df.columns for c in ["KORBAN MD", "KORBAN LB"]):
        df["TARGET"] = np.where(
            (df["KORBAN MD"] > 0) | (df["KORBAN LB"] > 0),
            "Berat",
            "Tidak Berat",
        )

    # --- Waktu ---
    df["DATETIME"] = pd.to_datetime(df["HARI / TGL / JAM KEJADIAN"], errors="coerce")
    df["JAM"] = df["DATETIME"].dt.hour
    df["BULAN"] = df["DATETIME"].dt.month
    df["HARI"] = df["DATETIME"].dt.day_name()

    df["KATEGORI_WAKTU"] = df["JAM"].apply(kategori_waktu)
    df["JAM_SIBUK"] = df["JAM"].apply(jam_sibuk)
    df["WEEKEND"] = np.where(df["HARI"].isin(["Saturday", "Sunday"]), "Weekend", "Weekday")

    # --- Identitas tersangka / korban ---
    df["UMUR_TERSANGKA"] = df["IDENTITAS TERSANGKA"].apply(ekstrak_umur)
    df["UMUR_KORBAN"] = df["IDENTITAS KORBAN"].apply(ekstrak_umur)

    df["KAT_UMUR_TERSANGKA"] = df["UMUR_TERSANGKA"].apply(kategori_umur)
    df["KAT_UMUR_KORBAN"] = df["UMUR_KORBAN"].apply(kategori_umur)

    df["JK_TERSANGKA"] = df["IDENTITAS TERSANGKA"].apply(ekstrak_jk)
    df["JK_KORBAN"] = df["IDENTITAS KORBAN"].apply(ekstrak_jk)

    df["PENDIDIKAN_TERSANGKA"] = df["IDENTITAS TERSANGKA"].apply(ekstrak_pendidikan)
    df["PENDIDIKAN_KORBAN"] = df["IDENTITAS KORBAN"].apply(ekstrak_pendidikan)

    df["PEKERJAAN_TERSANGKA"] = df["IDENTITAS TERSANGKA"].apply(ekstrak_pekerjaan)
    df["PEKERJAAN_KORBAN"] = df["IDENTITAS KORBAN"].apply(ekstrak_pekerjaan)

    # --- Jenis & risiko tabrakan ---
    df["JENIS_TABRAKAN"] = df["TYPE KECELAKAAN"].apply(jenis_tabrakan)
    df["RISIKO_TABRAKAN"] = df["JENIS_TABRAKAN"].apply(risiko_tabrakan)

    # --- Kerugian materi ---
    df["KERUGIAN"] = df["MATERI"].apply(konversi_materi)
    df["KATEGORI_KERUGIAN"] = df["KERUGIAN"].apply(kategori_kerugian)

    # --- Cluster TKP ---
    if top_tkp is None:
        top_tkp = df["TKP"].value_counts().nlargest(10).index.tolist()

    df["TKP_CLUSTER"] = np.where(df["TKP"].isin(top_tkp), df["TKP"], "LAINNYA")

    return df, top_tkp


def build_features(df: pd.DataFrame, train_columns=None):
    """
    Mengambil kolom FITUR dari dataframe lalu melakukan one-hot encoding.

    Jika train_columns diberikan (kolom hasil get_dummies saat training),
    maka hasil encoding akan di-reindex agar kolomnya identik dengan saat
    training (kolom baru dibuang, kolom yang hilang diisi 0). Ini penting
    untuk data baru pada saat prediksi.
    """
    X = df[FITUR].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)

    return X
