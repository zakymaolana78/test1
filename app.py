import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Stunting Kabupaten Indramayu",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

model_rf = joblib.load(
    "model_random_forest_indramayu.pkl"
)

model_lr = joblib.load(
    "model_linear_regression_indramayu.pkl"
)

fitur = joblib.load(
    "fitur_model_indramayu.pkl"
)

# ============================================================
# JUDUL
# ============================================================

st.title(
    "📊 Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu"
)

st.write(
    "Aplikasi ini membandingkan model "
    "Random Forest Regressor dan Linear Regression "
    "untuk memprediksi jumlah kasus balita stunting "
    "di Kabupaten Indramayu."
)

st.info(
    "Wilayah penelitian: Kabupaten Indramayu"
)

# ============================================================
# INPUT DATA
# ============================================================

st.header("Input Data Prediksi")

col1, col2 = st.columns(2)

with col1:

    tahun = st.number_input(
        "Tahun Prediksi",
        min_value=2025,
        max_value=2026,
        value=2025,
        step=1
    )

    kemiskinan = st.number_input(
        "Persentase Penduduk Miskin (%)",
        min_value=0.0,
        value=11.50,
        step=0.01
    )

    garis_kemiskinan = st.number_input(
        "Garis Kemiskinan (Rp)",
        min_value=0.0,
        value=580000.0,
        step=1000.0
    )

with col2:

    sanitasi = st.number_input(
        "Persentase Sanitasi Layak (%)",
        min_value=0.0,
        max_value=100.0,
        value=98.50,
        step=0.01
    )

    tenaga_gizi = st.number_input(
        "Jumlah Tenaga Kesehatan Gizi",
        min_value=0,
        value=52,
        step=1
    )

# ============================================================
# DATA INPUT
# ============================================================

data_input = pd.DataFrame({
    "tahun": [tahun],
    "persentase_penduduk_miskin": [kemiskinan],
    "garis_kemiskinan": [garis_kemiskinan],
    "persentase_sanitasi_layak": [sanitasi],
    "jumlah_nakes_gizi": [tenaga_gizi]
})

# ============================================================
# TOMBOL PREDIKSI
# ============================================================

if st.button(
    "🔮 Lakukan Prediksi",
    use_container_width=True
):

    # Random Forest
    prediksi_rf = model_rf.predict(
        data_input[fitur]
    )[0]

    # Linear Regression
    prediksi_lr = model_lr.predict(
        data_input[fitur]
    )[0]

    st.header(
        f"Hasil Prediksi Tahun {tahun}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌳 Random Forest Regressor"
        )

        st.metric(
            "Prediksi Jumlah Balita Stunting",
            f"{prediksi_rf:,.0f}"
        )

    with col2:

        st.subheader(
            "📈 Linear Regression"
        )

        st.metric(
            "Prediksi Jumlah Balita Stunting",
            f"{prediksi_lr:,.0f}"
        )

    # ========================================================
    # PERBANDINGAN
    # ========================================================

    st.subheader(
        "Perbandingan Hasil Prediksi"
    )

    perbandingan = pd.DataFrame({
        "Model": [
            "Random Forest Regressor",
            "Linear Regression"
        ],
        "Prediksi Jumlah Balita Stunting": [
            prediksi_rf,
            prediksi_lr
        ]
    })

    st.dataframe(
        perbandingan,
        use_container_width=True
    )

    # ========================================================
    # GRAFIK
    # ========================================================

    st.subheader(
        "Visualisasi Perbandingan Prediksi"
    )

    chart_data = perbandingan.set_index(
        "Model"
    )

    st.bar_chart(
        chart_data
    )

    # ========================================================
    # SELISIH PREDIKSI
    # ========================================================

    selisih = abs(
        prediksi_rf - prediksi_lr
    )

    st.info(
        f"Selisih prediksi kedua model: "
        f"{selisih:,.2f} kasus"
    )
