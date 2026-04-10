"""
Bootstrap Labeler + Random Forest Trainer — v2
Klasifikasi Kesehatan Kebun | LPP Agro Nusantara
=================================================
PERUBAHAN dari v1:
  - Kelas dikurangi dari 4 → 3 (Sehat / Sedang / Stres)
  - Threshold & CLASS_MAP diimport dari health_classes.py
  - Semua hardcode kelas dihapus, pakai loop CLASS_MAP
  - Plot distribusi VI menyesuaikan 3 kelas + warna baru

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

# ── import definisi kelas terpusat ──────────────────────────────
from health_classes import CLASS_MAP, CLASS_LABELS, CLASS_NAMES, THRESHOLDS

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    "vi_tiff_path"     : "./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif",
    "output_dir"       : "./model_output",
    "samples_per_class": 5000,
    "random_seed"      : 42,
    "rf_n_estimators"  : 200,
    "rf_max_depth"     : None,
    "rf_min_samples_leaf": 5,
    "rf_n_jobs"        : -1,
    "cv_folds"         : 5,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ================================================================
# 1. BACA GEOTIFF
# ================================================================
def load_vi_tiff(tiff_path: str) -> tuple[np.ndarray, dict]:
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan GEE export sudah selesai."
        )

    with rasterio.open(path) as src:
        log.info(f"Membaca: {path.name}")
        log.info(f"  Dimensi : {src.width} x {src.height} piksel")
        log.info(f"  CRS     : {src.crs}")
        log.info(f"  Bands   : {src.count}")

        vi_stack = src.read(out_dtype="float32",
                            resampling=Resampling.nearest)
        meta = src.meta.copy()

    vi_stack = np.where(vi_stack <= -9999, np.nan, vi_stack)
    vi_stack = np.where(vi_stack > 10, np.nan, vi_stack)

    log.info(f"  Valid pixels: {np.sum(~np.isnan(vi_stack[0]))}/{vi_stack[0].size}")
    return vi_stack, meta


# ================================================================
# 2. PSEUDO-LABELING — voting dari THRESHOLDS di health_classes.py
# ================================================================
def generate_pseudo_labels(vi_stack: np.ndarray) -> np.ndarray:
    """
    Generate label per piksel via voting mayoritas.
    Kelas dan threshold dibaca dari health_classes.THRESHOLDS.
    """
    ndvi  = vi_stack[0]
    gndvi = vi_stack[1]
    savi  = vi_stack[2]
    rvi   = vi_stack[3]

    H, W   = ndvi.shape
    n_cls  = len(THRESHOLDS)
    votes  = np.zeros((n_cls, H, W), dtype=np.uint8)

    log.info("Generating pseudo-labels (3 kelas: Sehat / Sedang / Stres)...")

    for i, (cls_name, thr) in enumerate(THRESHOLDS.items()):
        v_ndvi  = (ndvi  >= thr["NDVI_min"])  & (ndvi  < thr["NDVI_max"])
        v_gndvi = (gndvi >= thr["GNDVI_min"]) & (gndvi < thr["GNDVI_max"])
        v_savi  = (savi  >= thr["SAVI_min"])  & (savi  < thr["SAVI_max"])
        v_rvi   = (rvi   >= thr["RVI_min"])   & (rvi   < thr["RVI_max"])

        votes[i] = (v_ndvi.astype(np.uint8) + v_gndvi.astype(np.uint8) +
                    v_savi.astype(np.uint8) + v_rvi.astype(np.uint8))

    label_map = np.argmax(votes, axis=0).astype(np.float32) + 1

    invalid = np.isnan(ndvi) | np.isnan(gndvi) | np.isnan(savi) | np.isnan(rvi)
    label_map[invalid] = np.nan

    total_valid = np.sum(~np.isnan(label_map))
    for cls_id, cls_info in CLASS_MAP.items():
        count = np.sum(label_map == cls_id)
        pct   = 100 * count / total_valid if total_valid > 0 else 0
        log.info(f"  Kelas {cls_id} — {cls_info['name']:10s}: "
                 f"{count:8,.0f} piksel ({pct:.1f}%)")

    return label_map


# ================================================================
# 3. SAMPLING & BALANCING
# ================================================================
def sample_pixels(vi_stack: np.ndarray, label_map: np.ndarray,
                  samples_per_class: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    records = []

    for cls_id, cls_info in CLASS_MAP.items():
        rows, cols = np.where(label_map == cls_id)

        # Filter piksel dengan semua VI valid
        valid_mask = np.all(
            [~np.isnan(vi_stack[b][rows, cols]) for b in range(5)],
            axis=0
        )
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        n_available = len(rows)
        n_sample    = min(samples_per_class, n_available)

        if n_sample == 0:
            log.warning(f"Kelas {cls_id} ({cls_info['name']}) tidak ada piksel valid!")
            continue

        idx = rng.choice(n_available, size=n_sample, replace=False)
        s_rows = rows[idx]
        s_cols = cols[idx]

        df_part = pd.DataFrame(
            {fname: vi_stack[b][s_rows, s_cols]
             for b, fname in enumerate(feature_names)}
        )
        df_part["label"]     = cls_id
        df_part["label_str"] = cls_info["name"]
        records.append(df_part)

        log.info(f"  Sampled {cls_info['name']:10s}: "
                 f"{n_sample:,} dari {n_available:,} piksel")

    df = pd.concat(records, ignore_index=True).dropna()
    log.info(f"Total dataset: {len(df):,} baris, {df['label'].nunique()} kelas")
    return df


# ================================================================
# 4. TRAIN RANDOM FOREST
# ================================================================
def train_random_forest(df: pd.DataFrame, config: dict) -> tuple:
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    X = df[feature_names].values
    y = df["label"].values.astype(int)

    log.info(f"\nTraining Random Forest ({config['rf_n_estimators']} trees, "
             f"{len(CLASS_MAP)} kelas)...")

    rf = RandomForestClassifier(
        n_estimators     = config["rf_n_estimators"],
        max_depth        = config["rf_max_depth"],
        min_samples_leaf = config["rf_min_samples_leaf"],
        class_weight     = "balanced",
        random_state     = config["random_seed"],
        n_jobs           = config["rf_n_jobs"],
        oob_score        = True,
    )

    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True,
                         random_state=config["random_seed"])
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    rf.fit(X, y)
    log.info(f"  OOB Score  : {rf.oob_score_:.4f}")

    return rf, feature_names, cv_scores


# ================================================================
# 5. EVALUASI & VISUALISASI
# ================================================================
def evaluate_and_plot(rf, df: pd.DataFrame,
                      feature_names: list, output_dir: Path):
    X = df[feature_names].values
    y = df["label"].values.astype(int)
    y_pred = rf.predict(X)

    # Classification report
    report = classification_report(
        y, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    log.info("\nClassification Report:")
    log.info(classification_report(y, y_pred, target_names=CLASS_NAMES))

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- Plot 1: Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — RF 3 Kelas Kesehatan Kebun", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: Feature Importance ----
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = [cls_info["hex"] if i == len(importances) - 1 else "#85B7EB"
                  for i, _ in enumerate(importances)]
    importances.plot(kind="barh", ax=ax, color=bar_colors)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Kontribusi Setiap Indeks terhadap Klasifikasi", fontsize=12)
    ax.axvline(importances.mean(), color="red", linestyle="--",
               linewidth=1, label=f"Mean = {importances.mean():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: Distribusi VI per kelas (3 kelas) ----
    selected_vi = ["NDVI", "GNDVI", "SAVI", "RVI"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, vi_name in zip(axes, selected_vi):
        for cls_id, cls_info in CLASS_MAP.items():
            vals = df[df["label"] == cls_id][vi_name].values
            ax.hist(vals, bins=60, alpha=0.65,
                    label=cls_info["name"],
                    color=cls_info["hex"], density=True)

        # Tambahkan garis threshold NDVI
        if vi_name == "NDVI":
            ax.axvline(0.5, color="green",  linestyle="--", linewidth=1.2,
                       label="Threshold 0.5 (Sehat)")
            ax.axvline(0.2, color="red",    linestyle="--", linewidth=1.2,
                       label="Threshold 0.2 (Stres)")

        ax.set_title(vi_name, fontsize=11)
        ax.set_xlabel("Nilai Indeks")
        ax.set_ylabel("Densitas")
        ax.legend(fontsize=8)

    fig.suptitle("Distribusi VI per Kelas Kesehatan (3 Kelas)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "vi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot evaluasi tersimpan.")

    return report


# ================================================================
# 6. SIMPAN MODEL & METADATA
# ================================================================
def save_model(rf, feature_names: list, cv_scores,
               config: dict, output_dir: Path) -> Path:
    model_path = output_dir / "model.pkl"
    joblib.dump(rf, model_path, compress=3)
    log.info(f"Model saved: {model_path}")

    metadata = {
        "model_type"        : "RandomForestClassifier",
        "n_classes"         : len(CLASS_MAP),
        "class_labels"      : CLASS_LABELS,
        "class_definitions" : {
            "Sehat" : "NDVI > 0.5",
            "Sedang": "NDVI 0.2–0.5",
            "Stres" : "NDVI < 0.2",
        },
        "n_estimators"      : config["rf_n_estimators"],
        "oob_score"         : float(rf.oob_score_),
        "cv_accuracy_mean"  : float(cv_scores.mean()),
        "cv_accuracy_std"   : float(cv_scores.std()),
        "feature_names"     : feature_names,
        "thresholds_used"   : {
            k: {tk: float(tv) for tk, tv in v.items()}
            for k, v in THRESHOLDS.items()
        },
        "trained_on"        : pd.Timestamp.now().isoformat(),
        "vi_tiff_source"    : config["vi_tiff_path"],
        "samples_per_class" : config["samples_per_class"],
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
    log.info("Bootstrap Labeler + RF Trainer v2 — 3 Kelas")
    log.info(f"Kelas: {', '.join(CLASS_NAMES)}")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vi_stack, meta  = load_vi_tiff(CONFIG["vi_tiff_path"])
    label_map       = generate_pseudo_labels(vi_stack)

    # Simpan label map untuk validasi di QGIS
    label_meta = meta.copy()
    label_meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_dir / "pseudo_label_map.tif", "w",
                       **label_meta) as dst:
        dst.write(label_map[np.newaxis].astype(np.float32))
    log.info("pseudo_label_map.tif saved — buka di QGIS untuk validasi")

    df = sample_pixels(vi_stack, label_map,
                       CONFIG["samples_per_class"], CONFIG["random_seed"])
    df.to_csv(output_dir / "training_dataset.csv", index=False)

    rf, feature_names, cv_scores = train_random_forest(df, CONFIG)
    report  = evaluate_and_plot(rf, df, feature_names, output_dir)
    model_path = save_model(rf, feature_names, cv_scores, CONFIG, output_dir)

    log.info("\n" + "=" * 60)
    log.info("SELESAI")
    log.info(f"  OOB Accuracy : {rf.oob_score_:.4f}")
    log.info(f"  CV Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    log.info(f"  Model        : {model_path}")
    log.info("Lanjutkan dengan: python weekly_agent_pipeline.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
