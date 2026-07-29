import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_DATASET = "dataset_indramayu_2018_2024.csv"

df = pd.read_csv(FILE_DATASET)

print("Dataset berhasil dibaca.")
print("Jumlah data awal:", len(df))
print("\nKolom dataset:")
print(df.columns.tolist())


# ============================================================
# 2. FILTER KABUPATEN INDRAMAYU
# ============================================================

df["nama_kabupaten_kota"] = (
    df["nama_kabupaten_kota"]
    .astype(str)
    .str.strip()
)

df = df[
    df["nama_kabupaten_kota"]
    .str.contains("Indramayu", case=False, na=False)
].copy()


# ============================================================
# 3. FILTER TAHUN 2018–2024
# ============================================================

df["tahun"] = pd.to_numeric(
    df["tahun"],
    errors="coerce"
)

df = df[
    (df["tahun"] >= 2018) &
    (df["tahun"] <= 2024)
].copy()


# ============================================================
# 4. URUTKAN DATA
# ============================================================

df = df.sort_values("tahun").reset_index(drop=True)

print("\nDataset setelah filter:")
print(df)


# ============================================================
# 5. MENENTUKAN FITUR DAN TARGET
# ============================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

target = "jumlah_balita_stunting"


# ============================================================
# 6. KONVERSI DATA NUMERIK
# ============================================================

for kolom in fitur + [target]:
    df[kolom] = pd.to_numeric(
        df[kolom],
        errors="coerce"
    )


# ============================================================
# 7. MENANGANI MISSING VALUE
# ============================================================

df = df.dropna(
    subset=fitur + [target]
).reset_index(drop=True)


print("\nJumlah data setelah preprocessing:", len(df))


# ============================================================
# 8. MENYIAPKAN X DAN Y
# ============================================================

X = df[fitur]
y = df[target]


# ============================================================
# 9. MEMBAGI DATA TRAINING DAN TESTING
# ============================================================

# Karena data sangat sedikit, test_size 0.2
# digunakan untuk memisahkan sebagian data sebagai pengujian.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nJumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))


# ============================================================
# 10. MEMBUAT MODEL RANDOM FOREST
# ============================================================

model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# ============================================================
# 11. MEMBUAT MODEL LINEAR REGRESSION
# ============================================================

model_lr = LinearRegression()


# ============================================================
# 12. TRAINING MODEL
# ============================================================

model_rf.fit(
    X_train,
    y_train
)

model_lr.fit(
    X_train,
    y_train
)


print("\nTraining model selesai.")


# ============================================================
# 13. PREDIKSI DATA TESTING
# ============================================================

y_pred_rf = model_rf.predict(X_test)

y_pred_lr = model_lr.predict(X_test)


# ============================================================
# 14. FUNGSI EVALUASI
# ============================================================

def evaluasi_model(
    nama_model,
    y_true,
    y_pred
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    return {
        "Model": nama_model,
        "R2 Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


# ============================================================
# 15. EVALUASI KEDUA MODEL
# ============================================================

hasil_rf = evaluasi_model(
    "Random Forest Regressor",
    y_test,
    y_pred_rf
)

hasil_lr = evaluasi_model(
    "Linear Regression",
    y_test,
    y_pred_lr
)


hasil_evaluasi = pd.DataFrame([
    hasil_rf,
    hasil_lr
])


# ============================================================
# 16. MENAMPILKAN HASIL EVALUASI
# ============================================================

print("\n======================================")
print("HASIL EVALUASI MODEL")
print("======================================")

print(
    hasil_evaluasi.to_string(
        index=False
    )
)


# ============================================================
# 17. MENENTUKAN MODEL TERBAIK
# ============================================================

model_terbaik = hasil_evaluasi.loc[
    hasil_evaluasi["R2 Score"].idxmax()
]

print("\n======================================")
print("MODEL TERBAIK")
print("======================================")

print(
    "Model:",
    model_terbaik["Model"]
)

print(
    "R2 Score:",
    model_terbaik["R2 Score"]
)


# ============================================================
# 18. FEATURE IMPORTANCE RANDOM FOREST
# ============================================================

feature_importance = pd.DataFrame({
    "Fitur": fitur,
    "Importance": model_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


print("\n======================================")
print("FEATURE IMPORTANCE RANDOM FOREST")
print("======================================")

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 19. SIMPAN MODEL RANDOM FOREST
# ============================================================

joblib.dump(
    model_rf,
    "model_random_forest.pkl"
)


# ============================================================
# 20. SIMPAN MODEL LINEAR REGRESSION
# ============================================================

joblib.dump(
    model_lr,
    "model_linear_regression.pkl"
)


# ============================================================
# 21. SIMPAN HASIL EVALUASI
# ============================================================

hasil_evaluasi.to_csv(
    "hasil_evaluasi_model.csv",
    index=False
)


# ============================================================
# 22. SIMPAN FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "feature_importance_random_forest.csv",
    index=False
)


print("\n======================================")
print("MODEL BERHASIL DISIMPAN")
print("======================================")

print("model_random_forest.pkl")
print("model_linear_regression.pkl")
print("hasil_evaluasi_model.csv")
print("feature_importance_random_forest.csv")
