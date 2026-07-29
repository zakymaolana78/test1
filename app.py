```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Balita Stunting Kabupaten Indramayu",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. JUDUL APLIKASI
# ============================================================

st.title("📊 Prediksi Jumlah Balita Stunting Kabupaten Indramayu")

st.markdown("""
Aplikasi ini digunakan untuk membandingkan kinerja algoritma 
**Random Forest Regressor** dan **Linear Regression** dalam 
memprediksi jumlah balita stunting di Kabupaten Indramayu.

Model dilatih menggunakan data tahun **2019–2024** dan digunakan 
untuk melakukan prediksi tahun **2025–2026**.
""")

st.info("""
📍 Wilayah Penelitian: Kabupaten Indramayu  
📅 Data Training: 2019–2024  
🔮 Tahun Prediksi: 2025–2026  
🎯 Target: Jumlah Balita Stunting
""")

# ============================================================
# 3. LOAD DATASET
# ============================================================

st.header("1. Data Penelitian")

try:
    df = pd.read_csv("dataset_indramayu_2019_2024.csv")

    st.success("Dataset berhasil dimuat.")

except FileNotFoundError:
    st.error("""
    File dataset_indramayu_2019_2024.csv tidak ditemukan.

    Pastikan file CSV berada dalam folder yang sama dengan app.py.
    """)
    st.stop()

# ============================================================
# 4. PEMERIKSAAN DATA
# ============================================================

st.subheader("Data Kabupaten Indramayu")

st.dataframe(
    df,
    use_container_width=True
)

# ============================================================
# 5. VALIDASI DATA
# ============================================================

kolom_wajib = [
    "kode_kabupaten_kota",
    "nama_kabupaten_kota",
    "tahun",
    "jumlah_balita_stunting",
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

kolom_hilang = [
    kolom for kolom in kolom_wajib
    if kolom not in df.columns
]

if kolom_hilang:
    st.error(
        f"Kolom berikut tidak ditemukan dalam dataset: {kolom_hilang}"
    )
    st.stop()

# ============================================================
# 6. FILTER DATA INDRAMAYU
# ============================================================

df_indramayu = df[
    df["nama_kabupaten_kota"]
    .astype(str)
    .str.lower()
    .str.contains("indramayu")
].copy()

if df_indramayu.empty:
    st.error(
        "Data Kabupaten Indramayu tidak ditemukan dalam dataset."
    )
    st.stop()

# Pastikan tahun 2019–2024
df_indramayu = df_indramayu[
    (df_indramayu["tahun"] >= 2019) &
    (df_indramayu["tahun"] <= 2024)
].copy()

# Urutkan berdasarkan tahun
df_indramayu = df_indramayu.sort_values(
    by="tahun"
).reset_index(drop=True)

st.subheader("Dataset Kabupaten Indramayu Tahun 2019–2024")

st.dataframe(
    df_indramayu,
    use_container_width=True
)

# ============================================================
# 7. FITUR DAN TARGET
# ============================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

target = "jumlah_balita_stunting"

X = df_indramayu[fitur].copy()
y = df_indramayu[target].copy()

# ============================================================
# 8. CEK DATA KOSONG
# ============================================================

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:

    st.warning(
        "Terdapat data kosong. Data kosong akan diisi menggunakan "
        "nilai median masing-masing kolom."
    )

    X = X.fillna(X.median())
    y = y.fillna(y.median())

# ============================================================
# 9. PEMBAGIAN DATA TRAINING DAN TESTING
# ============================================================

st.header("2. Pembagian Data Training dan Testing")

st.write("""
Data tahun 2019–2024 digunakan sebagai data penelitian.
Data dibagi menjadi data training dan testing dengan proporsi 80:20.
""")

# Karena data hanya sedikit, gunakan pembagian berdasarkan waktu
# 2019–2023 = Training
# 2024 = Testing

df_train = df_indramayu[
    df_indramayu["tahun"] <= 2023
].copy()

df_test = df_indramayu[
    df_indramayu["tahun"] == 2024
].copy()

X_train = df_train[fitur]
y_train = df_train[target]

X_test = df_test[fitur]
y_test = df_test[target]

st.write(
    f"Jumlah data training: **{len(df_train)} data**"
)

st.write(
    f"Jumlah data testing: **{len(df_test)} data**"
)

# ============================================================
# 10. PEMBUATAN MODEL
# ============================================================

st.header("3. Pemodelan")

# Linear Regression
model_lr = LinearRegression()

# Random Forest
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# ============================================================
# 11. TRAINING MODEL
# ============================================================

model_lr.fit(
    X_train,
    y_train
)

model_rf.fit(
    X_train,
    y_train
)

st.success(
    "Model Linear Regression dan Random Forest berhasil dilatih."
)

# ============================================================
# 12. PREDIKSI DATA TESTING
# ============================================================

pred_lr = model_lr.predict(X_test)

pred_rf = model_rf.predict(X_test)

# ============================================================
# 13. FUNGSI EVALUASI
# ============================================================

def hitung_rmse(y_actual, y_prediksi):

    mse = mean_squared_error(
        y_actual,
        y_prediksi
    )

    rmse = np.sqrt(mse)

    return rmse


# Evaluasi Linear Regression
r2_lr = r2_score(
    y_test,
    pred_lr
)

mae_lr = mean_absolute_error(
    y_test,
    pred_lr
)

mse_lr = mean_squared_error(
    y_test,
    pred_lr
)

rmse_lr = hitung_rmse(
    y_test,
    pred_lr
)


# Evaluasi Random Forest
r2_rf = r2_score(
    y_test,
    pred_rf
)

mae_rf = mean_absolute_error(
    y_test,
    pred_rf
)

mse_rf = mean_squared_error(
    y_test,
    pred_rf
)

rmse_rf = hitung_rmse(
    y_test,
    pred_rf
)

# ============================================================
# 14. TABEL PERBANDINGAN MODEL
# ============================================================

st.header("4. Perbandingan Kinerja Model")

hasil_evaluasi = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest Regressor"
    ],

    "R² Score": [
        r2_lr,
        r2_rf
    ],

    "MAE": [
        mae_lr,
        mae_rf
    ],

    "MSE": [
        mse_lr,
        mse_rf
    ],

    "RMSE": [
        rmse_lr,
        rmse_rf
    ]

})

st.dataframe(
    hasil_evaluasi.style.format({
        "R² Score": "{:.4f}",
        "MAE": "{:,.2f}",
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}"
    }),
    use_container_width=True
)

# ============================================================
# 15. MENENTUKAN MODEL TERBAIK
# ============================================================

st.header("5. Model Terbaik")

# R2 lebih tinggi lebih baik
# MAE, MSE, RMSE lebih rendah lebih baik

if r2_rf > r2_lr:

    model_terbaik = "Random Forest Regressor"

    st.success(
        "🏆 Berdasarkan nilai R² Score, "
        "Random Forest Regressor memiliki performa terbaik."
    )

else:

    model_terbaik = "Linear Regression"

    st.success(
        "🏆 Berdasarkan nilai R² Score, "
        "Linear Regression memiliki performa terbaik."
    )

st.write(
    f"Model terbaik berdasarkan R² Score: **{model_terbaik}**"
)

# ============================================================
# 16. FEATURE IMPORTANCE RANDOM FOREST
# ============================================================

st.header("6. Feature Importance Random Forest")

feature_importance = pd.DataFrame({

    "Fitur": fitur,

    "Importance": model_rf.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(
    feature_importance,
    use_container_width=True
)

# Grafik feature importance
fig1, ax1 = plt.subplots(
    figsize=(10, 5)
)

ax1.bar(
    feature_importance["Fitur"],
    feature_importance["Importance"]
)

ax1.set_title(
    "Feature Importance Random Forest"
)

ax1.set_xlabel(
    "Variabel"
)

ax1.set_ylabel(
    "Tingkat Kepentingan"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig1)

# ============================================================
# 17. PREDIKSI 2025–2026
# ============================================================

st.header("7. Prediksi Jumlah Balita Stunting Tahun 2025–2026")

st.warning("""
Prediksi tahun 2025–2026 membutuhkan nilai variabel input 
persentase penduduk miskin, garis kemiskinan, sanitasi layak, 
dan jumlah tenaga gizi untuk tahun tersebut.
""")

st.write("""
Pada bagian ini, nilai fitur tahun 2025–2026 diproyeksikan 
berdasarkan tren data tahun 2019–2024.
""")

# ============================================================
# 18. PROYEKSI FITUR 2025–2026
# ============================================================

future_years = [2025, 2026]

future_data = []

for fitur_nama in fitur:

    data_tahun = df_indramayu[
        ["tahun", fitur_nama]
    ].copy()

    model_trend = LinearRegression()

    model_trend.fit(
        data_tahun[["tahun"]],
        data_tahun[fitur_nama]
    )

    prediksi_fitur = model_trend.predict(
        np.array(future_years).reshape(-1, 1)
    )

    for i, tahun in enumerate(future_years):

        if len(future_data) <= i:

            future_data.append({
                "tahun": tahun
            })

        future_data[i][fitur_nama] = prediksi_fitur[i]


df_future = pd.DataFrame(
    future_data
)

# ============================================================
# 19. PREDIKSI DENGAN 2 MODEL
# ============================================================

df_future[
    "Prediksi Linear Regression"
] = model_lr.predict(
    df_future[fitur]
)

df_future[
    "Prediksi Random Forest"
] = model_rf.predict(
    df_future[fitur]
)

# ============================================================
# 20. TAMPILKAN HASIL PREDIKSI
# ============================================================

st.subheader(
    "Hasil Prediksi Tahun 2025–2026"
)

st.dataframe(
    df_future[
        [
            "tahun",
            "Prediksi Linear Regression",
            "Prediksi Random Forest"
        ]
    ].style.format({
        "Prediksi Linear Regression": "{:,.2f}",
        "Prediksi Random Forest": "{:,.2f}"
    }),
    use_container_width=True
)

# ============================================================
# 21. GRAFIK PREDIKSI 2025–2026
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.plot(
    df_future["tahun"],
    df_future["Prediksi Linear Regression"],
    marker="o",
    label="Linear Regression"
)

ax2.plot(
    df_future["tahun"],
    df_future["Prediksi Random Forest"],
    marker="s",
    label="Random Forest"
)

ax2.set_title(
    "Perbandingan Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu Tahun 2025–2026"
)

ax2.set_xlabel(
    "Tahun"
)

ax2.set_ylabel(
    "Jumlah Balita Stunting"
)

ax2.legend()

ax2.grid(
    True
)

st.pyplot(fig2)

# ============================================================
# 22. KESIMPULAN APLIKASI
# ============================================================

st.header("8. Kesimpulan")

st.markdown(f"""
Berdasarkan hasil evaluasi model, diperoleh perbandingan antara 
algoritma **Linear Regression** dan **Random Forest Regressor** 
dalam memprediksi jumlah balita stunting di Kabupaten Indramayu.

Model dengan performa terbaik berdasarkan nilai R² Score adalah:

### 🏆 {model_terbaik}

Model tersebut selanjutnya dapat digunakan sebagai model utama 
dalam proses prediksi jumlah balita stunting tahun 2025–2026.
""")

# ============================================================
# 23. CATATAN METODOLOGI
# ============================================================

st.header("9. Catatan Penelitian")

st.markdown("""
**Alur penelitian pada aplikasi:**

Data 2019–2024  
↓  
Filter Kabupaten Indramayu  
↓  
Pemisahan Fitur (X) dan Target (Y)  
↓  
Data Training 2019–2023  
↓  
Data Testing 2024  
↓  
Training Linear Regression  
↓  
Training Random Forest Regressor  
↓  
Evaluasi Model  
↓  
Perbandingan Model  
↓  
Feature Importance Random Forest  
↓  
Proyeksi Variabel Input 2025–2026  
↓  
Prediksi Jumlah Balita Stunting 2025–2026
""")
```
