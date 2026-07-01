"""
Aplikasi Streamlit: Klasifikasi Keparahan Kecelakaan Lalu Lintas
(Berat vs Tidak Berat) menggunakan Decision Tree.

Direplikasi dari notebook Untitled8.ipynb.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preprocess, build_features, FITUR, validate_columns

st.set_page_config(
    page_title="Klasifikasi Keparahan Kecelakaan",
    layout="wide",
)

st.title("🚦 Klasifikasi Keparahan Kecelakaan Lalu Lintas")
st.caption("Decision Tree — replikasi dari notebook analisis data kecelakaan")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.train_columns = None
    st.session_state.top_tkp = None
    st.session_state.metrics = None

tab_train, tab_klasifikasi, tab_about = st.tabs(
    ["📊 Latih Model", "🔮 Klasifikasi", "ℹ️ Tentang"]
)

# ---------------------------------------------------------------------------
# TAB 1 — TRAINING
# ---------------------------------------------------------------------------
with tab_train:
    st.header("Latih Model dari Data Excel")
    st.write(
        "Unggah file Excel mentah (format & nama kolom sama seperti data sumber "
        "di notebook). Aplikasi akan melakukan seluruh feature engineering "
        "secara otomatis lalu melatih model Decision Tree dengan GridSearchCV."
    )

    uploaded_file = st.file_uploader(
        "Pilih file Excel (.xlsx)", type=["xlsx", "xls"], key="train_uploader"
    )

    colA, colB = st.columns(2)
    with colA:
        test_size = st.slider("Proporsi data test", 0.1, 0.4, 0.2, 0.05)
    with colB:
        cv_folds = st.slider("Jumlah fold cross-validation", 3, 10, 5, 1)

    run_training = st.button("🚀 Mulai Training", type="primary", disabled=uploaded_file is None)

    if run_training and uploaded_file is not None:
        with st.spinner("Membaca dan membersihkan data..."):
            df_raw = pd.read_excel(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip()

            missing = validate_columns(df_raw)
            if missing:
                st.error(
                    "File tidak sesuai format. Kolom berikut tidak ditemukan: "
                    + ", ".join(missing)
                )
                st.stop()

            df, top_tkp = preprocess(df_raw)

        st.success(f"Data berhasil diproses. Jumlah baris: {df.shape[0]}")

        with st.expander("Lihat distribusi target"):
            st.bar_chart(df["TARGET"].value_counts())

        X = build_features(df)
        y = df["TARGET"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        with st.spinner("Melatih model dengan GridSearchCV (bisa beberapa menit)..."):
            param_grid = {
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [2, 5, 10],
                "ccp_alpha": [0, 0.001, 0.005],
            }
            grid = GridSearchCV(
                DecisionTreeClassifier(
                    criterion="entropy", random_state=42, class_weight="balanced"
                ),
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        with st.spinner("Menghitung cross-validation score..."):
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")

        st.session_state.model = model
        st.session_state.train_columns = X.columns.tolist()
        st.session_state.top_tkp = top_tkp
        st.session_state.metrics = {
            "acc": acc,
            "report": report,
            "cm": cm,
            "classes": model.classes_,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
        }

        st.success("Model selesai dilatih ✅")

    if st.session_state.model is not None:
        st.divider()
        st.subheader("Hasil Evaluasi")
        m = st.session_state.metrics

        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Test", f"{m['acc']*100:.2f}%")
        c2.metric("CV Accuracy (mean)", f"{m['cv_mean']*100:.2f}%", f"±{m['cv_std']*100:.2f}%")
        c3.metric("Train Accuracy", f"{m['train_acc']*100:.2f}%")

        st.write("**Classification Report**")
        st.dataframe(pd.DataFrame(m["report"]).transpose())

        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            m["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=m["classes"], yticklabels=m["classes"], ax=ax,
        )
        ax.set_xlabel("Hasil Klasifikasi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        st.write("**Feature Importance (Top 20)**")
        importance = pd.DataFrame({
            "Fitur": st.session_state.train_columns,
            "Importance": st.session_state.model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(20)
        st.bar_chart(importance.set_index("Fitur"))

# ---------------------------------------------------------------------------
# TAB 2 — KLASIFIKASI
# ---------------------------------------------------------------------------
with tab_klasifikasi:
    st.header("Klasifikasi Kasus Baru")

    st.write(
        "Latih model terlebih dahulu di tab '📊 Latih Model', "
        "kemudian isi form di bawah untuk melakukan klasifikasi."
    )

    if st.session_state.model is None:
        st.info("Belum ada model. Latih model di tab '📊 Latih Model' terlebih dahulu.")
    else:
        st.subheader("Input Data Kasus")

        with st.form("klasifikasi_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Tersangka**")
                umur_tersangka = st.number_input("Umur Tersangka", 0, 100, 30)
                jk_tersangka = st.selectbox("Jenis Kelamin Tersangka", ["L", "P", "Unknown"])
                pendidikan_tersangka = st.selectbox(
                    "Pendidikan Tersangka", ["SD", "SLTP", "SLTA", "D3", "S1", "S2", "LAINNYA"]
                )
                pekerjaan_tersangka = st.selectbox(
                    "Pekerjaan Tersangka",
                    ["PELAJAR", "MAHASISWA", "BURUH", "KARYAWAN", "WIRASWASTA", "PNS", "IRT", "LAINNYA"],
                )
            with c2:
                st.markdown("**Korban**")
                umur_korban = st.number_input("Umur Korban", 0, 100, 30)
                jk_korban = st.selectbox("Jenis Kelamin Korban", ["L", "P", "Unknown"])
                pendidikan_korban = st.selectbox(
                    "Pendidikan Korban", ["SD", "SLTP", "SLTA", "D3", "S1", "S2", "LAINNYA"]
                )
                pekerjaan_korban = st.selectbox(
                    "Pekerjaan Korban",
                    ["PELAJAR", "MAHASISWA", "BURUH", "KARYAWAN", "WIRASWASTA", "PNS", "IRT", "LAINNYA"],
                )

            st.markdown("**Kecelakaan**")
            c3, c4, c5 = st.columns(3)
            with c3:
                jenis_tabrakan = st.selectbox(
                    "Jenis Tabrakan",
                    ["TABRAK MANUSIA", "DEPAN", "BELAKANG", "SAMPING", "TUNGGAL", "LAINNYA"],
                )
                jam = st.slider("Jam Kejadian (0-23)", 0, 23, 12)
            with c4:
                bulan = st.selectbox("Bulan", list(range(1, 13)))
                hari_weekend = st.selectbox("Hari", ["Weekday", "Weekend"])
            with c5:
                kerugian = st.number_input(
                    "Kerugian Materi (Rp)", 0, 1_000_000_000, 1_000_000, step=100_000
                )
                tahun_data = st.number_input("Tahun Data", 2000, 2100, 2024)

            tkp = st.text_input("TKP (lokasi kejadian)", "")
            submitted = st.form_submit_button("🔍 Klasifikasikan")

        if submitted:
            from preprocessing import (
                kategori_umur,
                risiko_tabrakan,
                kategori_kerugian,
                kategori_waktu,
                jam_sibuk,
            )

            risiko = risiko_tabrakan(jenis_tabrakan)
            tkp_cluster = tkp if tkp in (st.session_state.top_tkp or []) else "LAINNYA"

            row = {
                "UMUR_TERSANGKA": umur_tersangka,
                "UMUR_KORBAN": umur_korban,
                "KAT_UMUR_TERSANGKA": kategori_umur(umur_tersangka),
                "KAT_UMUR_KORBAN": kategori_umur(umur_korban),
                "JK_TERSANGKA": jk_tersangka,
                "JK_KORBAN": jk_korban,
                "PENDIDIKAN_TERSANGKA": pendidikan_tersangka,
                "PENDIDIKAN_KORBAN": pendidikan_korban,
                "PEKERJAAN_TERSANGKA": pekerjaan_tersangka,
                "PEKERJAAN_KORBAN": pekerjaan_korban,
                "JENIS_TABRAKAN": jenis_tabrakan,
                "RISIKO_TABRAKAN": risiko,
                "KATEGORI_WAKTU": kategori_waktu(jam),
                "JAM_SIBUK": jam_sibuk(jam),
                "WEEKEND": hari_weekend,
                "BULAN": bulan,
                "TKP_CLUSTER": tkp_cluster,
                "KATEGORI_KERUGIAN": kategori_kerugian(kerugian),
                "TAHUN DATA": tahun_data,
            }

            df_input = pd.DataFrame([row])[FITUR]
            X_input = build_features(df_input, train_columns=st.session_state.train_columns)

            hasil = st.session_state.model.predict(X_input)[0]
            proba = st.session_state.model.predict_proba(X_input)[0]
            classes = st.session_state.model.classes_

            st.divider()
            if hasil == "Berat":
                st.error(f"### Hasil Klasifikasi: **{hasil}**")
            else:
                st.success(f"### Hasil Klasifikasi: **{hasil}**")

            proba_df = pd.DataFrame(
                {"Kelas": classes, "Probabilitas": proba}
            ).set_index("Kelas")
            st.bar_chart(proba_df)

        st.divider()
        st.subheader("Klasifikasi banyak data sekaligus (batch)")
        batch_file = st.file_uploader(
            "Unggah file Excel data baru (format sama seperti data training)",
            type=["xlsx", "xls"],
            key="batch_uploader",
        )
        if batch_file is not None:
            df_batch_raw = pd.read_excel(batch_file)
            df_batch_raw.columns = df_batch_raw.columns.str.strip()

            missing = validate_columns(df_batch_raw)
            if missing:
                st.error("File tidak sesuai format. Kolom hilang: " + ", ".join(missing))
            else:
                df_batch, _ = preprocess(df_batch_raw, top_tkp=st.session_state.top_tkp)
                X_batch = build_features(
                    df_batch, train_columns=st.session_state.train_columns
                )
                hasil_batch = st.session_state.model.predict(X_batch)
                df_result = df_batch_raw.copy()
                df_result["HASIL KLASIFIKASI"] = hasil_batch

                st.write("Hasil klasifikasi:")
                st.dataframe(df_result)

                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil (.csv)", csv, "hasil_klasifikasi.csv", "text/csv"
                )

# ---------------------------------------------------------------------------
# TAB 3 — ABOUT
# ---------------------------------------------------------------------------
with tab_about:
    st.header("Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini merupakan deployment dari notebook analisis data kecelakaan
        lalu lintas. Model mengklasifikasikan setiap kasus kecelakaan menjadi:

        - **Berat** — jika ada korban Meninggal Dunia (MD) atau Luka Berat (LB)
        - **Tidak Berat** — selain itu

        **Pipeline yang direplikasi dari notebook:**
        1. Pembersihan kolom korban (MD/LB/LR)
        2. Ekstraksi fitur waktu (jam, bulan, hari, kategori waktu, jam sibuk, weekend)
        3. Ekstraksi umur, jenis kelamin, pendidikan, pekerjaan dari kolom identitas
        4. Klasifikasi jenis & risiko tabrakan
        5. Konversi & kategorisasi kerugian materi
        6. Clustering TKP (10 lokasi teratas, sisanya "LAINNYA")
        7. One-hot encoding fitur kategorikal
        8. Decision Tree + GridSearchCV (cross-validation 5-fold)

        **Catatan:** kolom yang wajib ada pada file Excel sumber:
        """
    )
    st.code(
        "KORBAN MD, KORBAN LB, KORBAN LR, HARI / TGL / JAM KEJADIAN,\n"
        "IDENTITAS TERSANGKA, IDENTITAS KORBAN, TYPE KECELAKAAN,\n"
        "MATERI, TKP, TAHUN DATA"
    )
