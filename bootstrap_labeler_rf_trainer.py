# bootstrap_labeler_rf_trainer.py

"""
Bootstrap Labeler + Random Forest Trainer
Klasifikasi Kesehatan Kebun — LPP Agro Nusantara
================================================
Alur:
  1. Baca GeoTIFF VI dari Google Drive (hasil GEE export)
  2. Generate pseudo-label per piksel via rule-based threshold
  3. Sampling piksel + balancing kelas
  4. Train Random Forest (sklearn)
  5. Evaluasi: classification report + confusion matrix
  6. Simpan model.pkl siap pakai di agent mingguan

Instalasi:
  pip install rasterio numpy pandas scikit-learn matplotlib seaborn joblib
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    # Path ke GeoTIFF hasil GEE export (5 band: NDVI,GNDVI,SAVI,RVI,EVI)
    "vi_tiff_path": "./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif",

    # Folder output model + artefak
    "output_dir": "./model_output",

    # Sampling — jumlah piksel per kelas untuk training
    # Lebih banyak = lebih akurat, tapi lebih lambat
    "samples_per_class": 5000,

    # Random seed untuk reprodusibilitas
    "random_seed": 42,

    # Random Forest hyperparameter
    "rf_n_estimators": 200,
    "rf_max_depth": None,     # None = grow full trees
    "rf_min_samples_leaf": 5,
    "rf_n_jobs": -1,          # pakai semua CPU core

    # Cross-validation folds
    "cv_folds": 5,
}

# ----------------------------------------------------------------
# THRESHOLD RULE-BASED UNTUK PSEUDO-LABELING
# Disesuaikan untuk kelapa sawit (Elaeis guineensis) Jawa Timur
# Sumber referensi: Jaya et al. 2020, Wahyu et al. 2022
# ----------------------------------------------------------------
# Format: (min_ndvi, min_gndvi, min_savi, min_rvi)
# Kelas ditentukan dari kombinasi semua indeks (voting mayoritas)
THRESHOLDS = {
    # Kelas 1: Sehat — vegetasi vigor tinggi, kanopi rapat
    "Sehat": {
        "NDVI_min":  0.65, "NDVI_max":  1.00,
        "GNDVI_min": 0.50, "GNDVI_max": 1.00,
        "SAVI_min":  0.55, "SAVI_max":  1.00,
        "RVI_min":   3.80, "RVI_max": 999.0,
    },
    # Kelas 2: Stres Ringan — penurunan klorofil awal
    "Stres_Ringan": {
        "NDVI_min":  0.45, "NDVI_max":  0.65,
        "GNDVI_min": 0.35, "GNDVI_max": 0.50,
        "SAVI_min":  0.38, "SAVI_max":  0.55,
        "RVI_min":   2.50, "RVI_max":   3.80,
    },
    # Kelas 3: Stres Berat — defisiensi nutrisi / kekeringan signifikan
    "Stres_Berat": {
        "NDVI_min":  0.25, "NDVI_max":  0.45,
        "GNDVI_min": 0.20, "GNDVI_max": 0.35,
        "SAVI_min":  0.18, "SAVI_max":  0.38,
        "RVI_min":   1.50, "RVI_max":   2.50,
    },
    # Kelas 4: Kritis — potensi kematian / serangan hama berat
    "Kritis": {
        "NDVI_min": -1.00, "NDVI_max":  0.25,
        "GNDVI_min":-1.00, "GNDVI_max": 0.20,
        "SAVI_min": -1.00, "SAVI_max":  0.18,
        "RVI_min":   0.00, "RVI_max":   1.50,
    },
}

CLASS_NAMES   = ["Sehat", "Stres_Ringan", "Stres_Berat", "Kritis"]
CLASS_COLORS  = ["#2ecc40", "#ffdc00", "#ff851b", "#e74c3c"]
CLASS_LABELS  = {1: "Sehat", 2: "Stres_Ringan", 3: "Stres_Berat", 4: "Kritis"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ================================================================
# 1. BACA GEOTIFF
# ================================================================
# def load_vi_tiff(tiff_path: str) -> tuple[np.ndarray, dict]:
#     """
#     Baca GeoTIFF 5-band hasil GEE.
#     Urutan band: NDVI(1), GNDVI(2), SAVI(3), RVI(4), EVI(5)

#     Returns:
#         vi_stack : np.ndarray shape (5, H, W)
#         meta     : dict metadata rasterio (CRS, transform, dll)
#     """
#     path = Path(tiff_path)
#     if not path.exists():
#         raise FileNotFoundError(
#             f"File tidak ditemukan: {path}\n"
#             f"Pastikan GEE export sudah selesai dan file ada di path tersebut."
#         )

#     with rasterio.open(path) as src:
#         log.info(f"Membaca: {path.name}")
#         log.info(f"  Dimensi : {src.width} x {src.height} piksel")
#         log.info(f"  CRS     : {src.crs}")
#         log.info(f"  Bands   : {src.count}")
#         log.info(f"  Resolusi: {src.res[0]:.1f} m")

#         vi_stack = src.read(
#             out_dtype="float32",
#             resampling=Resampling.nearest
#         )
#         meta = src.meta.copy()

#     # Replace nodata dengan NaN
#     vi_stack = np.where(vi_stack <= -9999, np.nan, vi_stack)
#     vi_stack = np.where(vi_stack > 10, np.nan, vi_stack)  # VI di luar range

#     log.info(f"  Valid pixels: {np.sum(~np.isnan(vi_stack[0]))}/{vi_stack[0].size}")
#     return vi_stack, meta

def load_vi_tiff(tiff_path: str) -> tuple[np.ndarray, dict]:
    """
    Baca GeoTIFF 12-band Sentinel-2 raw DN.
    Hitung VI: NDVI, GNDVI, SAVI, RVI, EVI
    Returns:
        vi_stack : np.ndarray shape (5, H, W) → NDVI,GNDVI,SAVI,RVI,EVI
        meta     : dict metadata rasterio
    """
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with rasterio.open(path) as src:
        log.info(f"Membaca: {path.name}")
        log.info(f"  Dimensi : {src.width} x {src.height} piksel")
        log.info(f"  CRS     : {src.crs}")
        log.info(f"  Bands   : {src.count}")
        log.info(f"  Deskripsi: {src.descriptions}")

        # Baca band yang dibutuhkan (index 0-based)
        # B2=band2, B3=band3, B4=band4, B8=band8
        B2  = src.read(2,  out_dtype="float32")   # Blue
        B3  = src.read(3,  out_dtype="float32")   # Green
        B4  = src.read(4,  out_dtype="float32")   # Red
        B8  = src.read(8,  out_dtype="float32")   # NIR
        meta = src.meta.copy()

    # Konversi DN → reflectance (0–1)
    B2  = B2  / 10000.0
    B3  = B3  / 10000.0
    B4  = B4  / 10000.0
    B8  = B8  / 10000.0

    # Hindari pembagian nol
    eps = 1e-10

    # Hitung VI
    NDVI  = (B8 - B4) / (B8 + B4 + eps)

    GNDVI = (B8 - B3) / (B8 + B3 + eps)

    L     = 0.5
    SAVI  = ((B8 - B4) / (B8 + B4 + L + eps)) * (1 + L)

    RVI   = B8 / (B4 + eps)

    EVI   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)

    # Stack jadi (5, H, W)
    vi_stack = np.stack([NDVI, GNDVI, SAVI, RVI, EVI], axis=0)

    # Filter nilai di luar range wajar
    vi_stack = np.where(np.abs(vi_stack) > 10, np.nan, vi_stack)

    # Log statistik
    vi_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    for i, name in enumerate(vi_names):
        valid = vi_stack[i][np.isfinite(vi_stack[i])]
        log.info(f"  {name}: min={valid.min():.3f}, max={valid.max():.3f}, "
                 f"mean={valid.mean():.3f}")

    log.info(f"  Valid pixels: {np.sum(np.isfinite(vi_stack[0]))}/{vi_stack[0].size}")
    return vi_stack, meta

# ================================================================
# 2. PSEUDO-LABELING — RULE-BASED VOTING
# ================================================================
def generate_pseudo_labels(vi_stack: np.ndarray) -> np.ndarray:
    """
    Generate label kelas per piksel menggunakan voting mayoritas
    dari 4 indeks (NDVI, GNDVI, SAVI, RVI).

    Setiap indeks memberikan 1 vote untuk kelas tertentu.
    Kelas dengan vote terbanyak menang.
    Jika semua sama (tie) → ambil dari NDVI (indeks paling reliabel).

    Returns:
        label_map : np.ndarray shape (H, W), nilai 1–4, NaN untuk invalid
    """
    ndvi  = vi_stack[0]
    gndvi = vi_stack[1]
    savi  = vi_stack[2]
    rvi   = vi_stack[3]
    # EVI (band 4) dipakai sebagai fitur training tapi tidak untuk labeling

    H, W = ndvi.shape
    votes = np.zeros((4, H, W), dtype=np.uint8)  # votes[kelas-1]

    log.info("Generating pseudo-labels via rule-based voting...")

    for i, (cls_name, thr) in enumerate(THRESHOLDS.items()):
        cls_idx = i  # 0=Sehat, 1=Stres_Ringan, 2=Stres_Berat, 3=Kritis

        v_ndvi  = ((ndvi  >= thr["NDVI_min"])  & (ndvi  < thr["NDVI_max"]))
        v_gndvi = ((gndvi >= thr["GNDVI_min"]) & (gndvi < thr["GNDVI_max"]))
        v_savi  = ((savi  >= thr["SAVI_min"])  & (savi  < thr["SAVI_max"]))
        v_rvi   = ((rvi   >= thr["RVI_min"])   & (rvi   < thr["RVI_max"]))

        votes[cls_idx] = (v_ndvi.astype(np.uint8) +
                          v_gndvi.astype(np.uint8) +
                          v_savi.astype(np.uint8) +
                          v_rvi.astype(np.uint8))

    # Ambil kelas dengan vote tertinggi (argmax → +1 untuk kelas 1-based)
    label_map = np.argmax(votes, axis=0).astype(np.float32) + 1

    # Piksel dengan semua VI = NaN → label NaN
    invalid_mask = np.isnan(ndvi) | np.isnan(gndvi) | np.isnan(savi) | np.isnan(rvi)
    label_map[invalid_mask] = np.nan

    # Distribusi kelas
    for i, cls in enumerate(CLASS_NAMES):
        count = np.sum(label_map == (i + 1))
        pct   = 100 * count / np.sum(~np.isnan(label_map))
        log.info(f"  Kelas {i+1} — {cls:15s}: {count:8,d} piksel ({pct:.1f}%)")

    return label_map


# ================================================================
# 3. SAMPLING & BALANCING DATASET
# ================================================================
def sample_pixels(vi_stack: np.ndarray,
                  label_map: np.ndarray,
                  samples_per_class: int,
                  random_seed: int) -> pd.DataFrame:
    """
    Sampling piksel per kelas + gabungkan jadi DataFrame.
    Stratified sampling: ambil N piksel per kelas untuk balancing.
    """
    rng = np.random.default_rng(random_seed)
    records = []

    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]

    for cls_id in range(1, 5):
        # Temukan koordinat piksel untuk kelas ini
        rows, cols = np.where(label_map == cls_id)

        # Filter piksel dengan VI valid (semua band tidak NaN)
        valid_mask = np.all(
            [~np.isnan(vi_stack[b][rows, cols]) for b in range(5)],
            axis=0
        )
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        n_available = len(rows)
        n_sample    = min(samples_per_class, n_available)

        if n_sample == 0:
            log.warning(f"Kelas {cls_id} tidak ada piksel valid!")
            continue

        # Random sampling
        idx = rng.choice(n_available, size=n_sample, replace=False)
        sampled_rows = rows[idx]
        sampled_cols = cols[idx]

        for b, fname in enumerate(feature_names):
            vals = vi_stack[b][sampled_rows, sampled_cols]
            if b == 0:
                df_part = pd.DataFrame({fname: vals})
            else:
                df_part[fname] = vals

        df_part["label"]    = cls_id
        df_part["label_str"] = CLASS_LABELS[cls_id]
        records.append(df_part)

        log.info(f"  Sampled kelas {cls_id} — {CLASS_LABELS[cls_id]}: "
                 f"{n_sample:,} dari {n_available:,} piksel")

    df = pd.concat(records, ignore_index=True).dropna()
    log.info(f"Total dataset: {len(df):,} baris, {df['label'].nunique()} kelas")
    return df


# ================================================================
# 4. TRAIN RANDOM FOREST
# ================================================================
def train_random_forest(df: pd.DataFrame, config: dict) -> tuple:
    """
    Train Random Forest classifier.
    Returns: model, X_test, y_test, feature_names
    """
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    X = df[feature_names].values
    y = df["label"].values.astype(int)

    log.info(f"\nTraining Random Forest ({config['rf_n_estimators']} trees)...")

    rf = RandomForestClassifier(
        n_estimators    = config["rf_n_estimators"],
        max_depth       = config["rf_max_depth"],
        min_samples_leaf= config["rf_min_samples_leaf"],
        class_weight    = "balanced",
        random_state    = config["random_seed"],
        n_jobs          = config["rf_n_jobs"],
        oob_score       = True,
    )

    # Cross-validation
    log.info(f"Running {config['cv_folds']}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True,
                         random_state=config["random_seed"])
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train final model pada seluruh data
    rf.fit(X, y)
    log.info(f"  OOB Score  : {rf.oob_score_:.4f}")

    return rf, feature_names, cv_scores


# ================================================================
# 5. EVALUASI & VISUALISASI
# ================================================================
def evaluate_and_plot(rf, df: pd.DataFrame,
                      feature_names: list, output_dir: Path):
    """
    Buat classification report + confusion matrix + feature importance plot.
    """
    X = df[feature_names].values
    y = df["label"].values.astype(int)
    y_pred = rf.predict(X)

    # Classification report
    report = classification_report(
        y, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    log.info("\nClassification Report (training data):")
    log.info(classification_report(y, y_pred, target_names=CLASS_NAMES))

    # Simpan report ke JSON
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- Plot 1: Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — RF Klasifikasi Kesehatan Kebun", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: confusion_matrix.png")

    # ---- Plot 2: Feature Importance ----
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#378ADD" if v < importances.max() * 0.8 else "#1D9E75"
              for v in importances.values]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Kontribusi Setiap Indeks terhadap Klasifikasi", fontsize=12)
    ax.axvline(importances.mean(), color="red", linestyle="--",
               linewidth=1, label=f"Mean = {importances.mean():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: feature_importance.png")

    # ---- Plot 3: Distribusi VI per kelas ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    selected_vi = ["NDVI", "GNDVI", "SAVI", "RVI"]

    for ax, vi_name in zip(axes, selected_vi):
        for cls_id, cls_name, color in zip(
            range(1, 5), CLASS_NAMES, CLASS_COLORS
        ):
            vals = df[df["label"] == cls_id][vi_name].values
            ax.hist(vals, bins=60, alpha=0.6, label=cls_name,
                    color=color, density=True)
        ax.set_title(vi_name, fontsize=11)
        ax.set_xlabel("Nilai Indeks")
        ax.set_ylabel("Densitas")
        ax.legend(fontsize=8)

    fig.suptitle("Distribusi VI per Kelas Kesehatan Kebun", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "vi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: vi_distribution.png")

    return report


# ================================================================
# 6. SIMPAN MODEL & METADATA
# ================================================================
def save_model(rf, feature_names: list, cv_scores,
               config: dict, output_dir: Path):
    """
    Simpan model.pkl + metadata JSON.
    """
    model_path = output_dir / "model.pkl"
    joblib.dump(rf, model_path, compress=3)
    log.info(f"Model saved: {model_path}")

    metadata = {
        "model_type"       : "RandomForestClassifier",
        "n_estimators"     : config["rf_n_estimators"],
        "oob_score"        : float(rf.oob_score_),
        "cv_accuracy_mean" : float(cv_scores.mean()),
        "cv_accuracy_std"  : float(cv_scores.std()),
        "feature_names"    : feature_names,
        "class_labels"     : CLASS_LABELS,
        "thresholds_used"  : {
            k: {tk: float(tv) for tk, tv in v.items()}
            for k, v in THRESHOLDS.items()
        },
        "trained_on"       : pd.Timestamp.now().isoformat(),
        "vi_tiff_source"   : config["vi_tiff_path"],
        "samples_per_class": config["samples_per_class"],
    }

    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info(f"Metadata saved: {meta_path}")

    return model_path


# ================================================================
# 7. MAIN
# ================================================================
def main():
    log.info("=" * 60)
    log.info("Bootstrap Labeler + RF Trainer — LPP Agro Nusantara")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Baca GeoTIFF
    vi_stack, meta = load_vi_tiff(CONFIG["vi_tiff_path"])

    # Step 2: Generate pseudo-label
    label_map = generate_pseudo_labels(vi_stack)

    # Simpan label map sebagai GeoTIFF (opsional — untuk validasi visual di QGIS)
    label_meta = meta.copy()
    label_meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})
    label_out = output_dir / "pseudo_label_map.tif"
    with rasterio.open(label_out, "w", **label_meta) as dst:
        dst.write(label_map[np.newaxis, :, :].astype(np.float32))
    log.info(f"Label map saved: {label_out}")

    # Step 3: Sampling
    df = sample_pixels(
        vi_stack, label_map,
        CONFIG["samples_per_class"],
        CONFIG["random_seed"]
    )
    df.to_csv(output_dir / "training_dataset.csv", index=False)
    log.info(f"Dataset saved: training_dataset.csv")

    # Step 4: Train
    rf, feature_names, cv_scores = train_random_forest(df, CONFIG)

    # Step 5: Evaluasi
    report = evaluate_and_plot(rf, df, feature_names, output_dir)

    # Step 6: Simpan model
    model_path = save_model(rf, feature_names, cv_scores, CONFIG, output_dir)

    log.info("\n" + "=" * 60)
    log.info("SELESAI — ringkasan output:")
    log.info(f"  model.pkl              → {model_path}")
    log.info(f"  pseudo_label_map.tif   → buka di QGIS untuk validasi")
    log.info(f"  confusion_matrix.png   → evaluasi akurasi per kelas")
    log.info(f"  feature_importance.png → kontribusi tiap VI")
    log.info(f"  vi_distribution.png    → sebaran nilai per kelas")
    log.info(f"  training_dataset.csv   → dataset mentah untuk audit")
    log.info(f"\nModel siap digunakan di agent pipeline mingguan.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
