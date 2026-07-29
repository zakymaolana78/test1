import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Prediksi Jumlah Balita Stunting Kabupaten Indramayu",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# JUDUL APLIKASI
# =========================================================

st.title(
    "Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu"
)

st.markdown(
    "Menggunakan algoritma "
    "**Random Forest Regressor**"
)

st.markdown(
    """
    Aplikasi ini menggunakan data Kabupaten Indramayu
    tahun **2018–2024** untuk membangun model Random Forest
    dan melakukan prediksi jumlah kasus balita stunting
    tahun **2025–2026**.
    """
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_indramayu_2018_2024.csv"
    )

    # Membersihkan spasi pada nama kolom
    df.columns = (
        df.columns
        .str.strip()
    )

    # Mengurutkan data berdasarkan tahun
    df = df.sort_values(
        "tahun"
    ).reset_index(drop=True)

    return df


# =========================================================
# LOAD MODEL RANDOM FOREST
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "random_forest_model.pkl"
    )

    return model


# =========================================================
# LOAD DATA DAN MODEL
# =========================================================

try:

    df = load_data()

    rf = load_model()

except Exception as e:

    st.error(
        "Terjadi kesalahan saat memuat dataset "
        "atau model Random Forest."
    )

    st.exception(e)

    st.stop()


# =========================================================
# FITUR DAN TARGET
# =========================================================

# Variabel input (X)
fitur = [

    "persentase_penduduk_miskin",

    "garis_kemiskinan",

    "persentase_sanitasi_layak",

    "jumlah_nakes_gizi"

]


# Variabel target (Y)
target = (
    "jumlah_kasus_balita_stunting"
)


# =========================================================
# VALIDASI KOLOM DATASET
# =========================================================

kolom_wajib = [

    "kode_kabupaten_kota",

    "nama_kabupaten_kota",

    "tahun",

    "jumlah_kasus_balita_stunting",

    "persentase_penduduk_miskin",

    "garis_kemiskinan",

    "persentase_sanitasi_layak",

    "jumlah_nakes_gizi"

]


kolom_hilang = [

    kolom

    for kolom in kolom_wajib

    if kolom not in df.columns

]


if kolom_hilang:

    st.error(
        "Kolom berikut tidak ditemukan "
        "dalam dataset:"
    )

    st.write(
        kolom_hilang
    )

    st.write(
        "Kolom yang tersedia:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Informasi Penelitian"
)

st.sidebar.write(
    "Wilayah: Kabupaten Indramayu"
)

st.sidebar.write(
    "Kode Wilayah: 3212"
)

st.sidebar.write(
    "Periode Data: 2018–2024"
)

st.sidebar.write(
    "Periode Prediksi: 2025–2026"
)

st.sidebar.write(
    "Algoritma: Random Forest Regressor"
)


# =========================================================
# DATASET AKTUAL
# =========================================================

st.subheader(
    "Data Aktual Kabupaten Indramayu "
    "Tahun 2018–2024"
)


st.dataframe(

    df[
        kolom_wajib
    ],

    use_container_width=True

)


# =========================================================
# PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df[
    fitur
]


df_actual = df.copy()


df_actual[
    "Prediksi Random Forest"
] = rf.predict(
    X_actual
)


# =========================================================
# HASIL PREDIKSI DATA AKTUAL
# =========================================================

st.subheader(
    "Hasil Prediksi Random Forest "
    "Tahun 2018–2024"
)


st.dataframe(

    df_actual[

        [

            "tahun",

            "jumlah_kasus_balita_stunting",

            "Prediksi Random Forest"

        ]

    ],

    use_container_width=True

)


# =========================================================
# PREDIKSI TAHUN 2025–2026
# =========================================================

st.subheader(
    "Prediksi Jumlah Balita Stunting "
    "Tahun 2025–2026"
)


# Tahun yang akan diprediksi
tahun_future = [

    2025,

    2026

]


df_future = pd.DataFrame({

    "tahun":
    tahun_future

})


# =========================================================
# PROYEKSI VARIABEL INPUT 2025–2026
# =========================================================

for kolom in fitur:

    # Membuat tren linear berdasarkan
    # data historis tahun 2018–2024

    koefisien = np.polyfit(

        df["tahun"],

        df[kolom],

        1

    )


    fungsi_prediksi = np.poly1d(

        koefisien

    )


    df_future[

        kolom

    ] = fungsi_prediksi(

        df_future[
            "tahun"
        ]

    )


# =========================================================
# PREDIKSI RANDOM FOREST 2025–2026
# =========================================================

X_future = df_future[
    fitur
]


df_future[

    "Prediksi Jumlah Balita Stunting"

] = rf.predict(

    X_future

)


# Memastikan hasil tidak negatif
df_future[

    "Prediksi Jumlah Balita Stunting"

] = np.maximum(

    0,

    df_future[

        "Prediksi Jumlah Balita Stunting"

    ]

)


# Membulatkan hasil prediksi
df_future[

    "Prediksi Jumlah Balita Stunting"

] = (

    df_future[

        "Prediksi Jumlah Balita Stunting"

    ]

    .round(0)

)


# Nama wilayah
df_future[

    "Nama Kabupaten/Kota"

] = (

    "KABUPATEN INDRAMAYU"

)


# =========================================================
# TABEL HASIL PREDIKSI 2025–2026
# =========================================================

st.dataframe(

    df_future[

        [

            "Nama Kabupaten/Kota",

            "tahun",

            "Prediksi Jumlah Balita Stunting"

        ]

    ],

    use_container_width=True

)


# =========================================================
# VISUALISASI PREDIKSI 2025–2026
# =========================================================

st.subheader(

    "Visualisasi Prediksi Jumlah Balita Stunting "

    "Tahun 2025–2026"

)


fig, ax = plt.subplots(

    figsize=(10, 5)

)


ax.bar(

    df_future[

        "tahun"

    ].astype(str),

    df_future[

        "Prediksi Jumlah Balita Stunting"

    ]

)


ax.set_xlabel(

    "Tahun"

)


ax.set_ylabel(

    "Jumlah Kasus Balita Stunting"

)


ax.set_title(

    "Prediksi Jumlah Kasus Balita Stunting "

    "Kabupaten Indramayu Tahun 2025–2026"

)


ax.grid(

    axis="y",

    alpha=0.3

)


st.pyplot(

    fig

)


plt.close(

    fig

)


# =========================================================
# VISUALISASI TREN 2018–2026
# =========================================================

st.subheader(

    "Grafik Tren Jumlah Kasus Balita Stunting "

    "Kabupaten Indramayu Tahun 2018–2026"

)


# Data aktual
df_tren_aktual = df[

    [

        "tahun",

        "jumlah_kasus_balita_stunting"

    ]

].copy()


# Data prediksi
df_tren_prediksi = df_future[

    [

        "tahun",

        "Prediksi Jumlah Balita Stunting"

    ]

].copy()


# Membuat grafik
fig2, ax2 = plt.subplots(

    figsize=(10, 5)

)


# Grafik data aktual
ax2.plot(

    df_tren_aktual[

        "tahun"

    ],

    df_tren_aktual[

        "jumlah_kasus_balita_stunting"

    ],

    marker="o",

    label="Data Aktual"

)


# Grafik data prediksi
ax2.plot(

    df_tren_prediksi[

        "tahun"

    ],

    df_tren_prediksi[

        "Prediksi Jumlah Balita Stunting"

    ],

    marker="s",

    linestyle="--",

    label="Prediksi Random Forest"

)


ax2.set_title(

    "Tren Jumlah Kasus Balita Stunting "

    "Kabupaten Indramayu Tahun 2018–2026"

)


ax2.set_xlabel(

    "Tahun"

)


ax2.set_ylabel(

    "Jumlah Kasus Balita Stunting"

)


ax2.legend()


ax2.grid(

    True,

    alpha=0.3

)


st.pyplot(

    fig2

)


plt.close(

    fig2

)


# =========================================================
# EVALUASI MODEL
# =========================================================

st.subheader(

    "Evaluasi Model Random Forest"

)


st.caption(

    "Evaluasi model dilakukan menggunakan data "

    "Kabupaten Indramayu tahun 2018–2024. "

    "Prediksi tahun 2025–2026 digunakan untuk "

    "estimasi jumlah kasus balita stunting "

    "pada periode mendatang."

)


# Nilai aktual
y_actual = df[

    target

]


# Nilai prediksi
y_pred = df_actual[

    "Prediksi Random Forest"

]


# Menghitung R² Score
r2 = r2_score(

    y_actual,

    y_pred

)


# Menghitung MAE
mae = mean_absolute_error(

    y_actual,

    y_pred

)


# Menghitung MSE
mse = mean_squared_error(

    y_actual,

    y_pred

)


# Menghitung RMSE
rmse = np.sqrt(

    mse

)


# Membuat tabel evaluasi
eval_df = pd.DataFrame({

    "Model": [

        "Random Forest Regressor"

    ],

    "R² Score": [

        r2

    ],

    "MAE": [

        mae

    ],

    "MSE": [

        mse

    ],

    "RMSE": [

        rmse

    ]

})


st.dataframe(

    eval_df.style.format({

        "R² Score":

        "{:.4f}",

        "MAE":

        "{:,.2f}",

        "MSE":

        "{:,.2f}",

        "RMSE":

        "{:,.2f}"

    }),

    use_container_width=True

)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(

    "Feature Importance Random Forest"

)


feature_importance = pd.DataFrame({

    "Fitur":

    fitur,

    "Importance":

    rf.feature_importances_

})


feature_importance = (

    feature_importance

    .sort_values(

        "Importance",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


# =========================================================
# TABEL FEATURE IMPORTANCE
# =========================================================

st.dataframe(

    feature_importance,

    use_container_width=True

)


# =========================================================
# GRAFIK FEATURE IMPORTANCE
# =========================================================

fig3, ax3 = plt.subplots(

    figsize=(10, 5)

)


ax3.barh(

    feature_importance[

        "Fitur"

    ],

    feature_importance[

        "Importance"

    ]

)


ax3.set_xlabel(

    "Nilai Importance"

)


ax3.set_ylabel(

    "Variabel"

)


ax3.set_title(

    "Feature Importance "

    "Random Forest Regressor"

)


ax3.invert_yaxis()


ax3.grid(

    axis="x",

    alpha=0.3

)


st.pyplot(

    fig3

)


plt.close(

    fig3

)


# =========================================================
# KESIMPULAN PREDIKSI
# =========================================================

st.subheader(

    "Kesimpulan Prediksi"

)


# Tahun dengan prediksi tertinggi
tahun_tertinggi = int(

    df_future.loc[

        df_future[

            "Prediksi Jumlah Balita Stunting"

        ].idxmax(),

        "tahun"

    ]

)


# Nilai prediksi tertinggi
nilai_tertinggi = (

    df_future[

        "Prediksi Jumlah Balita Stunting"

    ].max()

)


st.info(

    f"Berdasarkan hasil prediksi menggunakan "

    f"Random Forest Regressor, estimasi jumlah "

    f"kasus balita stunting tertinggi pada periode "

    f"2025–2026 diperkirakan terjadi pada tahun "

    f"{tahun_tertinggi} dengan jumlah sekitar "

    f"{nilai_tertinggi:,.0f} kasus."

)
