import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Stunting Indramayu",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# JUDUL APLIKASI
# ============================================================

st.title(
    "Prediksi Jumlah Balita Stunting Kabupaten Indramayu"
)

st.markdown(
    """
    Aplikasi ini menggunakan data Kabupaten Indramayu
    tahun 2018–2024 untuk membangun model prediksi
    jumlah balita stunting tahun 2025–2026.

    Model yang dibandingkan:
    - Random Forest Regressor
    - Linear Regression
    """
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "dataset_indramayu_2018_2024.csv"
    )


df = load_data()


# ============================================================
# FILTER DATA INDRAMAYU
# ============================================================

df["nama_kabupaten_kota"] = (
    df["nama_kabupaten_kota"]
    .astype(str)
    .str.strip()
)

df = df[
    df["nama_kabupaten_kota"]
    .str.contains(
        "Indramayu",
        case=False,
        na=False
    )
].copy()


# ============================================================
# FILTER TAHUN
# ============================================================

df["tahun"] = pd.to_numeric(
    df["tahun"],
    errors="coerce"
)

df = df[
    (df["tahun"] >= 2018) &
    (df["tahun"] <= 2024)
].copy()


df = df.sort_values(
    "tahun"
).reset_index(drop=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    model_rf = joblib.load(
        "model_random_forest.pkl"
    )

    model_lr = joblib.load(
        "model_linear_regression.pkl"
    )

    return model_rf, model_lr


model_rf, model_lr = load_models()


# ============================================================
# FITUR DAN TARGET
# ============================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

target = "jumlah_balita_stunting"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Menu Aplikasi"
)

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Data Dataset",
        "Evaluasi Model",
        "Feature Importance",
        "Prediksi 2025–2026"
    ]
)


# ============================================================
# MENU 1 — DATASET
# ============================================================

if menu == "Data Dataset":

    st.header(
        "Dataset Kabupaten Indramayu 2018–2024"
    )

    st.write(
        "Data yang digunakan dalam penelitian:"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "Informasi Dataset"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        len(df)
    )

    col2.metric(
        "Tahun Awal",
        int(df["tahun"].min())
    )

    col3.metric(
        "Tahun Akhir",
        int(df["tahun"].max())
    )


# ============================================================
# MENU 2 — EVALUASI MODEL
# ============================================================

elif menu == "Evaluasi Model":

    st.header(
        "Perbandingan Performa Model"
    )

    X = df[fitur]

    y = df[target]


    # Prediksi data historis
    pred_rf = model_rf.predict(X)

    pred_lr = model_lr.predict(X)


    # Evaluasi
    hasil = pd.DataFrame({

        "Model": [
            "Random Forest Regressor",
            "Linear Regression"
        ],

        "R² Score": [
            r2_score(y, pred_rf),
            r2_score(y, pred_lr)
        ],

        "MAE": [
            mean_absolute_error(
                y,
                pred_rf
            ),

            mean_absolute_error(
                y,
                pred_lr
            )
        ],

        "MSE": [
            mean_squared_error(
                y,
                pred_rf
            ),

            mean_squared_error(
                y,
                pred_lr
            )
        ],

        "RMSE": [
            np.sqrt(
                mean_squared_error(
                    y,
                    pred_rf
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y,
                    pred_lr
                )
            )
        ]

    })


    st.dataframe(
        hasil.style.format({
            "R² Score": "{:.4f}",
            "MAE": "{:,.2f}",
            "MSE": "{:,.2f}",
            "RMSE": "{:,.2f}"
        }),
        use_container_width=True
    )


    # Model terbaik
    model_terbaik = hasil.loc[
        hasil["R² Score"].idxmax()
    ]


    st.success(
        f"Model dengan nilai R² Score tertinggi: "
        f"{model_terbaik['Model']}"
    )


    # Grafik perbandingan
    st.subheader(
        "Perbandingan R² Score"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        hasil["Model"],
        hasil["R² Score"]
    )

    ax.set_ylabel(
        "R² Score"
    )

    ax.set_title(
        "Perbandingan Performa Model"
    )

    plt.xticks(
        rotation=15
    )

    st.pyplot(fig)


# ============================================================
# MENU 3 — FEATURE IMPORTANCE
# ============================================================

elif menu == "Feature Importance":

    st.header(
        "Feature Importance Random Forest"
    )

    importance = pd.DataFrame({

        "Fitur": fitur,

        "Importance":
        model_rf.feature_importances_

    })


    importance = importance.sort_values(
        "Importance",
        ascending=False
    )


    st.dataframe(
        importance,
        use_container_width=True
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        importance["Fitur"],
        importance["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Variabel"
    )

    ax.set_title(
        "Feature Importance Random Forest"
    )

    ax.invert_yaxis()

    st.pyplot(fig)


# ============================================================
# MENU 4 — PREDIKSI 2025–2026
# ============================================================

elif menu == "Prediksi 2025–2026":

    st.header(
        "Prediksi Jumlah Balita Stunting "
        "Kabupaten Indramayu"
    )


    st.info(
        """
        Masukkan nilai prediktor untuk tahun 2025
        dan 2026. Sistem akan menghasilkan prediksi
        menggunakan Random Forest dan Linear Regression.
        """
    )


    # Nilai input 2025
    st.subheader(
        "Input Data Tahun 2025"
    )

    col1, col2 = st.columns(2)


    with col1:

        kemiskinan_2025 = st.number_input(
            "Persentase Penduduk Miskin 2025",
            min_value=0.0,
            value=float(
                df["persentase_penduduk_miskin"].iloc[-1]
            )
        )


        garis_2025 = st.number_input(
            "Garis Kemiskinan 2025",
            min_value=0.0,
            value=float(
                df["garis_kemiskinan"].iloc[-1]
            )
        )


    with col2:

        sanitasi_2025 = st.number_input(
            "Persentase Sanitasi Layak 2025",
            min_value=0.0,
            value=float(
                df["persentase_sanitasi_layak"].iloc[-1]
            )
        )


        gizi_2025 = st.number_input(
            "Jumlah Tenaga Gizi 2025",
            min_value=0.0,
            value=float(
                df["jumlah_tenaga_gizi"].iloc[-1]
            )
        )


    # Input 2026
    st.subheader(
        "Input Data Tahun 2026"
    )

    col1, col2 = st.columns(2)


    with col1:

        kemiskinan_2026 = st.number_input(
            "Persentase Penduduk Miskin 2026",
            min_value=0.0,
            value=float(
                df["persentase_penduduk_miskin"].iloc[-1]
            )
        )


        garis_2026 = st.number_input(
            "Garis Kemiskinan 2026",
            min_value=0.0,
            value=float(
                df["garis_kemiskinan"].iloc[-1]
            )
        )


    with col2:

        sanitasi_2026 = st.number_input(
            "Persentase Sanitasi Layak 2026",
            min_value=0.0,
            value=float(
                df["persentase_sanitasi_layak"].iloc[-1]
            )
        )


        gizi_2026 = st.number_input(
            "Jumlah Tenaga Gizi 2026",
            min_value=0.0,
            value=float(
                df["jumlah_tenaga_gizi"].iloc[-1]
            )
        )


    # ========================================================
    # MEMBUAT DATA PREDIKSI
    # ========================================================

    data_future = pd.DataFrame({

        "tahun": [
            2025,
            2026
        ],

        "persentase_penduduk_miskin": [
            kemiskinan_2025,
            kemiskinan_2026
        ],

        "garis_kemiskinan": [
            garis_2025,
            garis_2026
        ],

        "persentase_sanitasi_layak": [
            sanitasi_2025,
            sanitasi_2026
        ],

        "jumlah_tenaga_gizi": [
            gizi_2025,
            gizi_2026
        ]

    })


    X_future = data_future[fitur]


    # ========================================================
    # PREDIKSI RANDOM FOREST
    # ========================================================

    prediksi_rf = model_rf.predict(
        X_future
    )


    # ========================================================
    # PREDIKSI LINEAR REGRESSION
    # ========================================================

    prediksi_lr = model_lr.predict(
        X_future
    )


    # ========================================================
    # HASIL PREDIKSI
    # ========================================================

    hasil_prediksi = pd.DataFrame({

        "Tahun": [
            2025,
            2026
        ],

        "Prediksi Random Forest": [
            prediksi_rf[0],
            prediksi_rf[1]
        ],

        "Prediksi Linear Regression": [
            prediksi_lr[0],
            prediksi_lr[1]
        ]

    })


    st.subheader(
        "Hasil Prediksi"
    )


    st.dataframe(

        hasil_prediksi.style.format({

            "Prediksi Random Forest":
            "{:,.2f}",

            "Prediksi Linear Regression":
            "{:,.2f}"

        }),

        use_container_width=True

    )


    # ========================================================
    # GRAFIK PREDIKSI
    # ========================================================

    st.subheader(
        "Grafik Perbandingan Prediksi 2025–2026"
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    ax.plot(

        hasil_prediksi["Tahun"],

        hasil_prediksi[
            "Prediksi Random Forest"
        ],

        marker="o",

        label="Random Forest"

    )


    ax.plot(

        hasil_prediksi["Tahun"],

        hasil_prediksi[
            "Prediksi Linear Regression"
        ],

        marker="s",

        label="Linear Regression"

    )


    ax.set_xlabel(
        "Tahun"
    )

    ax.set_ylabel(
        "Jumlah Balita Stunting"
    )

    ax.set_title(
        "Perbandingan Prediksi Jumlah Balita Stunting"
    )

    ax.legend()

    ax.grid(True)


    st.pyplot(fig)


    # ========================================================
    # DOWNLOAD HASIL
    # ========================================================

    csv = hasil_prediksi.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        label="Download Hasil Prediksi",

        data=csv,

        file_name=
        "hasil_prediksi_stunting_indramayu_2025_2026.csv",

        mime="text/csv"

    )
