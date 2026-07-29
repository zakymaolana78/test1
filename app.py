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
    page_title="Prediksi Stunting Kabupaten Indramayu",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# JUDUL
# =========================================================

st.title(
    "Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu"
)

st.markdown(
    "Menggunakan algoritma: "
    "**Random Forest Regressor**"
)

st.markdown(
    "Data penelitian tahun **2018–2024** "
    "digunakan untuk membangun model dan "
    "melakukan prediksi jumlah balita stunting "
    "tahun **2025–2026**."
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_indramayu_2018_2024.csv"
    )

    df = df[
        df["nama_kabupaten_kota"]
        .astype(str)
        .str.contains(
            "Indramayu",
            case=False,
            na=False
        )
    ].copy()

    return df.sort_values(
        "tahun"
    ).reset_index(drop=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "random_forest_model.pkl"
    )


# =========================================================
# LOAD DATA DAN MODEL
# =========================================================

df = load_data()
rf = load_model()


# =========================================================
# FITUR DAN TARGET
# =========================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

target = "jumlah_balita_stunting"


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
    "Data: 2018–2024"
)

st.sidebar.write(
    "Prediksi: 2025–2026"
)

st.sidebar.write(
    "Model: Random Forest Regressor"
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
        [
            "kode_kabupaten_kota",
            "nama_kabupaten_kota",
            "tahun",
            "jumlah_balita_stunting",
            "persentase_penduduk_miskin",
            "garis_kemiskinan",
            "persentase_sanitasi_layak",
            "jumlah_tenaga_gizi"
        ]
    ],
    use_container_width=True
)


# =========================================================
# PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df[fitur]

df_actual = df.copy()

df_actual[
    "Prediksi Random Forest"
] = rf.predict(
    X_actual
)


# =========================================================
# HASIL PREDIKSI 2018–2024
# =========================================================

st.subheader(
    "Hasil Prediksi Random Forest "
    "Tahun 2018–2024"
)

st.dataframe(
    df_actual[
        [
            "tahun",
            "jumlah_balita_stunting",
            "Prediksi Random Forest"
        ]
    ],
    use_container_width=True
)


# =========================================================
# PREDIKSI 2025–2026
# =========================================================

st.subheader(
    "Prediksi Jumlah Balita Stunting "
    "Tahun 2025–2026"
)

# Membuat data tahun 2025–2026
tahun_future = [2025, 2026]

df_future = pd.DataFrame({
    "tahun": tahun_future
})


# =========================================================
# PROYEKSI FITUR 2025–2026
# =========================================================

for kolom in fitur:

    koefisien = np.polyfit(
        df["tahun"],
        df[kolom],
        1
    )

    fungsi_prediksi = np.poly1d(
        koefisien
    )

    df_future[kolom] = (
        fungsi_prediksi(
            df_future["tahun"]
        )
    )


# =========================================================
# PREDIKSI RANDOM FOREST
# =========================================================

X_future = df_future[fitur]

df_future[
    "Prediksi Jumlah Balita Stunting"
] = rf.predict(
    X_future
)

df_future[
    "Nama Kabupaten/Kota"
] = "Indramayu"


# =========================================================
# TAMPILKAN HASIL PREDIKSI
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
    df_future["tahun"].astype(str),
    df_future[
        "Prediksi Jumlah Balita Stunting"
    ]
)

ax.set_xlabel(
    "Tahun"
)

ax.set_ylabel(
    "Jumlah Balita Stunting"
)

ax.set_title(
    "Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu Tahun 2025–2026"
)

st.pyplot(fig)


# =========================================================
# VISUALISASI TREN 2018–2026
# =========================================================

st.subheader(
    "Grafik Tren Jumlah Balita Stunting "
    "Kabupaten Indramayu Tahun 2018–2026"
)

df_tren_aktual = df[
    [
        "tahun",
        "jumlah_balita_stunting"
    ]
].copy()

df_tren_aktual[
    "Jenis"
] = "Data Aktual"

df_tren_prediksi = df_future[
    [
        "tahun",
        "Prediksi Jumlah Balita Stunting"
    ]
].copy()

df_tren_prediksi = (
    df_tren_prediksi.rename(
        columns={
            "Prediksi Jumlah Balita Stunting":
            "jumlah_balita_stunting"
        }
    )
)

df_tren_prediksi[
    "Jenis"
] = "Data Prediksi"

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.plot(
    df_tren_aktual["tahun"],
    df_tren_aktual[
        "jumlah_balita_stunting"
    ],
    marker="o",
    label="Data Aktual"
)

ax2.plot(
    df_tren_prediksi["tahun"],
    df_tren_prediksi[
        "jumlah_balita_stunting"
    ],
    marker="s",
    linestyle="--",
    label="Prediksi Random Forest"
)

ax2.set_title(
    "Tren Jumlah Balita Stunting "
    "Kabupaten Indramayu Tahun 2018–2026"
)

ax2.set_xlabel(
    "Tahun"
)

ax2.set_ylabel(
    "Jumlah Balita Stunting"
)

ax2.legend()

ax2.grid(True)

st.pyplot(fig2)


# =========================================================
# EVALUASI MODEL
# =========================================================

st.subheader(
    "Evaluasi Model Random Forest"
)

st.caption(
    "Evaluasi model dilakukan berdasarkan "
    "data aktual tahun 2018–2024. "
    "Prediksi tahun 2025–2026 digunakan "
    "untuk melihat estimasi jumlah balita stunting "
    "pada periode mendatang."
)


y_actual = df[
    target
]

y_pred = df_actual[
    "Prediksi Random Forest"
]


r2 = r2_score(
    y_actual,
    y_pred
)

mae = mean_absolute_error(
    y_actual,
    y_pred
)

mse = mean_squared_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mse
)


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

    "Fitur": fitur,

    "Importance":
    rf.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


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

st.pyplot(fig3)


# =========================================================
# KESIMPULAN
# =========================================================

st.subheader(
    "Kesimpulan Prediksi"
)

tahun_tertinggi = df_future.loc[
    df_future[
        "Prediksi Jumlah Balita Stunting"
    ].idxmax(),
    "tahun"
]

nilai_tertinggi = df_future[
    "Prediksi Jumlah Balita Stunting"
].max()


st.info(
    f"Berdasarkan hasil prediksi menggunakan "
    f"Random Forest Regressor, estimasi jumlah "
    f"balita stunting tertinggi pada periode "
    f"2025–2026 diperkirakan terjadi pada tahun "
    f"{tahun_tertinggi} dengan jumlah sekitar "
    f"{nilai_tertinggi:,.0f} balita."
)
```
