"""
Bootstrap Labeler + RF Trainer v3 — Fixed Band Detection
LPP Agro Nusantara | Madiun, Jawa Timur
=========================================================
FIX dari v2:
  - Auto-detect urutan band dari TIFF (description / band name)
  - Jika VI belum ada → hitung langsung dari band spektral (B4, B8, B3, B11)
  - Fallback: coba semua kemungkinan posisi band NDVI
  - Diagnostic print: tampilkan semua band + nilai statistik sebelum labeling
  - Guard: jika hanya 1 kelas terdeteksi → hentikan training + tampilkan saran

Jalankan diagnostic dulu sebelum training:
  python bootstrap_labeler_rf_trainer_v3.py --diagnose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from health_classes import CLASS_MAP, CLASS_LABELS, CLASS_NAMES, THRESHOLDS

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    # Ganti path ini ke file TIFF kamu
    "vi_tiff_path"      : "./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif",
    "output_dir"        : "./model_output",
    "samples_per_class" : 5000,
    "random_seed"       : 42,
    "rf_n_estimators"   : 200,
    "rf_max_depth"      : None,
    "rf_min_samples_leaf": 5,
    "rf_n_jobs"         : -1,
    "cv_folds"          : 5,
}

# Nama band yang mungkin ada di GeoTIFF GEE export
# Key = nama yang mungkin muncul di band description (case-insensitive)
BAND_ALIASES = {
    "NDVI" : ["ndvi"],
    "GNDVI": ["gndvi"],
    "SAVI" : ["savi"],
    "RVI"  : ["rvi"],
    "EVI"  : ["evi"],
    "B2"   : ["b2", "blue", "band2", "band 2"],
    "B3"   : ["b3", "green", "band3", "band 3"],
    "B4"   : ["b4", "red", "band4", "band 4"],
    "B5"   : ["b5", "rededge", "red_edge", "band5"],
    "B8"   : ["b8", "nir", "band8", "band 8", "nir_broad"],
    "B8A"  : ["b8a", "nir_narrow", "band8a"],
    "B11"  : ["b11", "swir1", "band11"],
    "B12"  : ["b12", "swir2", "band12"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ================================================================
# DIAGNOSTIC — tampilkan info semua band
# ================================================================
def diagnose_tiff(tiff_path: str):
    """
    Tampilkan informasi lengkap semua band di TIFF.
    Jalankan ini dulu sebelum training untuk tahu posisi band.
    """
    path = Path(tiff_path)
    print("\n" + "=" * 65)
    print(f"DIAGNOSTIC: {path.name}")
    print("=" * 65)

    with rasterio.open(path) as src:
        print(f"Dimensi    : {src.width} x {src.height} piksel")
        print(f"CRS        : {src.crs}")
        print(f"Jumlah band: {src.count}")
        print(f"Dtype      : {src.dtypes[0]}")
        print(f"NoData     : {src.nodata}")
        print(f"Bounds     : {src.bounds}")
        print()
        print(f"{'Band':>5} {'Description':25} {'Min':>10} {'Max':>10} "
              f"{'Mean':>10} {'Valid%':>7}")
        print("-" * 70)

        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or f"Band_{i}"
            # Baca sample 1000x1000 tengah (cepat)
            w = min(src.width,  1000)
            h = min(src.height, 1000)
            col_off = (src.width  - w) // 2
            row_off = (src.height - h) // 2
            window  = rasterio.windows.Window(col_off, row_off, w, h)
            data    = src.read(i, window=window).astype(float)

            nodata = src.nodata if src.nodata is not None else -9999
            valid  = data[
                np.isfinite(data) &
                (data > nodata + 1) &
                (data < 1e6)
            ]
            if valid.size > 0:
                mn, mx, me = valid.min(), valid.max(), valid.mean()
                pct_valid  = 100 * valid.size / data.size
            else:
                mn = mx = me = float("nan")
                pct_valid = 0.0

            print(f"  {i:>3} {desc:25} {mn:>10.4f} {mx:>10.4f} "
                  f"{me:>10.4f} {pct_valid:>6.1f}%")

    print("=" * 65)
    print("\nTips:")
    print("  - Band dengan nilai -1 s/d 1  → kemungkinan VI (NDVI, dll)")
    print("  - Band dengan nilai 0 s/d 0.5 → kemungkinan reflectance spektral")
    print("  - Band dengan nilai 0 s/d 1e4 → kemungkinan DN (belum dibagi 10000)")
    print("\nSetelah tahu posisi band, set BAND_POSITIONS di CONFIG atau")
    print("biarkan auto-detect berjalan saat training.\n")


# ================================================================
# AUTO-DETECT POSISI BAND
# ================================================================
def detect_band_positions(src: rasterio.DatasetReader) -> dict:
    """
    Coba deteksi posisi band dari descriptions.
    Return dict: {"NDVI": 1, "GNDVI": 2, ...} (1-indexed)
    """
    positions = {}
    descriptions = [
        (src.descriptions[i] or f"band_{i+1}").lower()
        for i in range(src.count)
    ]

    log.info("Auto-detecting band positions...")
    for target, aliases in BAND_ALIASES.items():
        for band_idx, desc in enumerate(descriptions):
            if any(alias in desc for alias in aliases):
                positions[target] = band_idx + 1  # 1-indexed
                log.info(f"  Found {target:6s} → Band {band_idx + 1} ({desc})")
                break

    return positions


def detect_band_by_range(src: rasterio.DatasetReader) -> dict:
    """
    Fallback: deteksi band berdasarkan range nilai.
    VI biasanya -1..1, reflectance 0..0.5, DN 0..10000.
    """
    positions = {}
    log.info("Fallback: detecting bands by value range...")

    # Baca sampel kecil per band
    w = min(src.width, 500)
    h = min(src.height, 500)
    col_off = (src.width - w) // 2
    row_off = (src.height - h) // 2
    window = rasterio.windows.Window(col_off, row_off, w, h)

    band_stats = []
    for i in range(1, src.count + 1):
        data   = src.read(i, window=window).astype(float)
        nodata = src.nodata if src.nodata is not None else -9999
        valid  = data[np.isfinite(data) & (data > nodata + 1) & (data < 1e6)]
        if valid.size > 0:
            band_stats.append({
                "idx" : i,
                "min" : float(valid.min()),
                "max" : float(valid.max()),
                "mean": float(valid.mean()),
                "std" : float(valid.std()),
            })
        else:
            band_stats.append({"idx": i, "min": 0, "max": 0, "mean": 0, "std": 0})

    # Kandidat VI: nilai mean antara -0.2 dan 0.9, range kecil
    vi_candidates = [
        b for b in band_stats
        if -0.5 <= b["mean"] <= 0.95 and (b["max"] - b["min"]) < 3.0
    ]

    # Sort berdasarkan mean menurun (NDVI biasanya tertinggi di vegetasi)
    vi_candidates.sort(key=lambda x: x["mean"], reverse=True)

    vi_keys = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    for i, key in enumerate(vi_keys):
        if i < len(vi_candidates):
            positions[key] = vi_candidates[i]["idx"]
            log.info(f"  Assigned {key:6s} → Band {vi_candidates[i]['idx']} "
                     f"(mean={vi_candidates[i]['mean']:.3f})")

    # Kandidat spektral: nilai mean 0..0.5 (reflectance) atau band sisanya
    spec_candidates = [
        b for b in band_stats
        if b["idx"] not in [v for v in positions.values()]
        and 0.0 <= b["mean"] <= 0.6
    ]
    spec_keys = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]
    for i, key in enumerate(spec_keys):
        if i < len(spec_candidates):
            positions[key] = spec_candidates[i]["idx"]

    return positions


# ================================================================
# HITUNG VI DARI BAND SPEKTRAL (jika VI tidak ada di TIFF)
# ================================================================
def compute_vi_from_spectral(src: rasterio.DatasetReader,
                              band_pos: dict) -> dict[str, np.ndarray]:
    """
    Hitung NDVI, GNDVI, SAVI, RVI, EVI dari band spektral.
    Dipanggil jika VI tidak ditemukan di TIFF.
    """
    log.info("Menghitung VI dari band spektral...")

    def read_band(key):
        if key not in band_pos:
            raise KeyError(f"Band {key} tidak ditemukan. Jalankan --diagnose.")
        data = src.read(band_pos[key], out_dtype="float32")
        nodata = src.nodata if src.nodata is not None else -9999
        data = np.where((data <= nodata + 1) | (~np.isfinite(data)), np.nan, data)

        # Auto-scale DN → reflectance jika nilai > 2
        if np.nanmean(data[np.isfinite(data)][:1000]) > 2:
            log.info(f"  {key}: nilai besar terdeteksi, dibagi 10000 (DN→reflectance)")
            data = data / 10000.0
        return data

    nir  = read_band("B8")
    red  = read_band("B4")
    green = read_band("B3")
    blue  = read_band("B2") if "B2" in band_pos else None

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)

    # GNDVI
    gndvi = (nir - green) / (nir + green + 1e-10)
    gndvi = np.clip(gndvi, -1, 1)

    # SAVI (L=0.5)
    L = 0.5
    savi = ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)
    savi = np.clip(savi, -1, 1)

    # RVI
    rvi = nir / (red + 1e-10)
    rvi = np.clip(rvi, 0, 30)

    # EVI
    if blue is not None:
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
    else:
        evi = 2.5 * (nir - red) / (nir + 6 * red + 1 + 1e-10)
    evi = np.clip(evi, -1, 1)

    vi_dict = {"NDVI": ndvi, "GNDVI": gndvi, "SAVI": savi, "RVI": rvi, "EVI": evi}

    log.info("Statistik VI yang dihitung (sample pusat 500x500):")
    for name, arr in vi_dict.items():
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            log.info(f"  {name:6s}: min={valid.min():.3f} max={valid.max():.3f} "
                     f"mean={valid.mean():.3f}")

    return vi_dict


# ================================================================
# LOAD VI STACK — auto-detect atau hitung dari spektral
# ================================================================
def load_vi_stack(tiff_path: str) -> tuple[np.ndarray, dict]:
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with rasterio.open(path) as src:
        log.info(f"Membaca: {path.name}")
        log.info(f"  Dimensi : {src.width} x {src.height}")
        log.info(f"  CRS     : {src.crs}")
        log.info(f"  Bands   : {src.count}")
        meta = src.meta.copy()

        # Step 1: coba detect dari description
        band_pos = detect_band_positions(src)

        # Step 2: jika NDVI tidak ketemu, fallback by range
        if "NDVI" not in band_pos:
            log.warning("NDVI tidak ditemukan dari description. "
                        "Coba fallback deteksi by value range...")
            band_pos = detect_band_by_range(src)

        # Step 3: jika masih tidak ada NDVI, hitung dari spektral
        if "NDVI" not in band_pos:
            log.warning("VI tidak ditemukan. Menghitung dari band spektral...")
            vi_dict = compute_vi_from_spectral(src, band_pos)
        else:
            # Baca VI langsung dari band yang terdeteksi
            log.info("Membaca VI dari band yang terdeteksi...")
            vi_dict = {}
            for vi_name in ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]:
                if vi_name in band_pos:
                    data = src.read(band_pos[vi_name], out_dtype="float32")
                    nodata = src.nodata if src.nodata is not None else -9999
                    data = np.where(
                        (data <= nodata + 1) | (~np.isfinite(data)),
                        np.nan, data
                    )
                    # Auto-scale jika perlu
                    valid_sample = data[np.isfinite(data)]
                    if valid_sample.size > 0 and np.abs(valid_sample).mean() > 2:
                        data = data / 10000.0
                    vi_dict[vi_name] = data
                    log.info(f"  {vi_name:6s} dari band {band_pos[vi_name]}: "
                             f"mean={np.nanmean(data):.3f}")

            # Jika ada VI yang hilang, ganti dengan nol
            for vi_name in ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]:
                if vi_name not in vi_dict:
                    log.warning(f"  {vi_name} tidak tersedia, diisi 0")
                    shape = list(vi_dict.values())[0].shape
                    vi_dict[vi_name] = np.zeros(shape, dtype=np.float32)

    # Stack ke (5, H, W): NDVI, GNDVI, SAVI, RVI, EVI
    vi_stack = np.stack([
        vi_dict["NDVI"],
        vi_dict["GNDVI"],
        vi_dict["SAVI"],
        vi_dict["RVI"],
        vi_dict["EVI"],
    ], axis=0)

    # Validasi distribusi NDVI
    ndvi_valid = vi_stack[0][np.isfinite(vi_stack[0])]
    log.info(f"\nValidasi NDVI:")
    log.info(f"  Min={ndvi_valid.min():.3f} Max={ndvi_valid.max():.3f} "
             f"Mean={ndvi_valid.mean():.3f}")
    log.info(f"  Sehat  (>0.5): {100*np.mean(ndvi_valid > 0.5):.1f}%")
    log.info(f"  Sedang (0.2–0.5): {100*np.mean((ndvi_valid >= 0.2) & (ndvi_valid <= 0.5)):.1f}%")
    log.info(f"  Stres  (<0.2): {100*np.mean(ndvi_valid < 0.2):.1f}%")

    return vi_stack, meta


# ================================================================
# PSEUDO-LABELING
# ================================================================
def generate_pseudo_labels(vi_stack: np.ndarray) -> np.ndarray:
    ndvi  = vi_stack[0]
    gndvi = vi_stack[1]
    savi  = vi_stack[2]
    rvi   = vi_stack[3]

    H, W  = ndvi.shape
    n_cls = len(THRESHOLDS)
    votes = np.zeros((n_cls, H, W), dtype=np.uint8)

    log.info("\nGenerating pseudo-labels...")

    for i, (cls_name, thr) in enumerate(THRESHOLDS.items()):
        v_ndvi  = (ndvi  >= thr["NDVI_min"])  & (ndvi  < thr["NDVI_max"])
        v_gndvi = (gndvi >= thr["GNDVI_min"]) & (gndvi < thr["GNDVI_max"])
        v_savi  = (savi  >= thr["SAVI_min"])  & (savi  < thr["SAVI_max"])
        v_rvi   = (rvi   >= thr["RVI_min"])   & (rvi   < thr["RVI_max"])
        votes[i] = (v_ndvi.astype(np.uint8) + v_gndvi.astype(np.uint8) +
                    v_savi.astype(np.uint8) + v_rvi.astype(np.uint8))

    label_map = np.argmax(votes, axis=0).astype(np.float32) + 1
    invalid = (
        np.isnan(ndvi) | np.isnan(gndvi) |
        np.isnan(savi) | np.isnan(rvi)
    )
    label_map[invalid] = np.nan

    total_valid = float(np.sum(~np.isnan(label_map)))
    n_classes_found = 0
    for cls_id, cls_info in CLASS_MAP.items():
        count = np.sum(label_map == cls_id)
        pct   = 100 * count / total_valid if total_valid > 0 else 0
        log.info(f"  Kelas {cls_id} — {cls_info['name']:10s}: "
                 f"{count:10,.0f} piksel ({pct:.1f}%)")
        if count > 0:
            n_classes_found += 1

    # Guard: semua piksel masuk satu kelas → hentikan
    if n_classes_found < 2:
        log.error(
            "\n[ERROR] Hanya 1 kelas terdeteksi! Kemungkinan penyebab:\n"
            "  1. Band yang dibaca bukan VI — jalankan dulu:\n"
            "     python bootstrap_labeler_rf_trainer_v3.py --diagnose\n"
            "  2. VI sudah ada tapi nilainya belum di-scale (masih DN ×10000)\n"
            "  3. AOI mencakup area non-vegetasi (kota, laut, sawah kosong)\n"
            "\nSolusi cepat: jalankan dengan --diagnose untuk lihat band info."
        )
        sys.exit(1)

    return label_map


# ================================================================
# SAMPLING
# ================================================================
def sample_pixels(vi_stack, label_map, samples_per_class, random_seed):
    rng = np.random.default_rng(random_seed)
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    records = []

    for cls_id, cls_info in CLASS_MAP.items():
        rows, cols = np.where(label_map == cls_id)
        valid_mask = np.all(
            [~np.isnan(vi_stack[b][rows, cols]) for b in range(5)], axis=0
        )
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        n_available = len(rows)
        n_sample    = min(samples_per_class, n_available)

        if n_sample == 0:
            log.warning(f"Kelas {cls_id} ({cls_info['name']}): 0 piksel valid!")
            continue

        idx    = rng.choice(n_available, size=n_sample, replace=False)
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

    if not records:
        log.error("Tidak ada piksel yang bisa di-sample!")
        sys.exit(1)

    df = pd.concat(records, ignore_index=True).dropna()
    log.info(f"Total dataset: {len(df):,} baris, {df['label'].nunique()} kelas")
    return df


# ================================================================
# TRAIN RF
# ================================================================
def train_random_forest(df, config):
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]
    X = df[feature_names].values
    y = df["label"].values.astype(int)

    n_cls_actual = len(np.unique(y))
    log.info(f"\nTraining RF: {config['rf_n_estimators']} trees, "
             f"{n_cls_actual} kelas aktual")

    rf = RandomForestClassifier(
        n_estimators     = config["rf_n_estimators"],
        max_depth        = config["rf_max_depth"],
        min_samples_leaf = config["rf_min_samples_leaf"],
        class_weight     = "balanced",
        random_state     = config["random_seed"],
        n_jobs           = config["rf_n_jobs"],
        oob_score        = True,
    )

    if n_cls_actual >= 2:
        cv = StratifiedKFold(n_splits=min(config["cv_folds"], n_cls_actual * 2),
                             shuffle=True, random_state=config["random_seed"])
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        log.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    else:
        cv_scores = np.array([1.0])

    rf.fit(X, y)
    log.info(f"  OOB Score  : {rf.oob_score_:.4f}")
    return rf, feature_names, cv_scores


# ================================================================
# EVALUASI
# ================================================================
def evaluate_and_plot(rf, df, feature_names, output_dir):
    X      = df[feature_names].values
    y      = df["label"].values.astype(int)
    y_pred = rf.predict(X)

    present_labels = sorted(np.unique(y))
    present_names  = [CLASS_LABELS[l] for l in present_labels]

    report = classification_report(
        y, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True
    )
    log.info("\nClassification Report:")
    log.info(classification_report(y, y_pred, labels=present_labels,
                                   target_names=present_names))

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    ConfusionMatrixDisplay(cm, display_labels=present_names).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title("Confusion Matrix — RF Kesehatan Kebun", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color="#85B7EB")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Kontribusi Setiap Indeks terhadap Klasifikasi")
    ax.axvline(importances.mean(), color="red", linestyle="--",
               linewidth=1, label=f"Mean = {importances.mean():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Distribusi VI
    selected_vi = ["NDVI", "GNDVI", "SAVI", "RVI"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, vi_name in zip(axes.flatten(), selected_vi):
        for cls_id, cls_info in CLASS_MAP.items():
            vals = df[df["label"] == cls_id][vi_name].values
            if len(vals) > 0:
                ax.hist(vals, bins=60, alpha=0.65,
                        label=cls_info["name"], color=cls_info["hex"], density=True)
        if vi_name == "NDVI":
            ax.axvline(0.5, color="green", linestyle="--", lw=1.2, label="0.5")
            ax.axvline(0.2, color="red",   linestyle="--", lw=1.2, label="0.2")
        ax.set_title(vi_name, fontsize=11)
        ax.set_xlabel("Nilai Indeks")
        ax.set_ylabel("Densitas")
        ax.legend(fontsize=8)
    fig.suptitle("Distribusi VI per Kelas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "vi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Plot evaluasi tersimpan.")
    return report


# ================================================================
# SIMPAN MODEL
# ================================================================
def save_model(rf, feature_names, cv_scores, config, output_dir):
    model_path = output_dir / "model.pkl"
    joblib.dump(rf, model_path, compress=3)

    metadata = {
        "model_type"       : "RandomForestClassifier",
        "n_classes"        : len(CLASS_MAP),
        "class_labels"     : CLASS_LABELS,
        "class_definitions": {
            "Sehat" : "NDVI > 0.5",
            "Sedang": "NDVI 0.2–0.5",
            "Stres" : "NDVI < 0.2",
        },
        "n_estimators"     : config["rf_n_estimators"],
        "oob_score"        : float(rf.oob_score_),
        "cv_accuracy_mean" : float(cv_scores.mean()),
        "cv_accuracy_std"  : float(cv_scores.std()),
        "feature_names"    : feature_names,
        "trained_on"       : pd.Timestamp.now().isoformat(),
        "vi_tiff_source"   : config["vi_tiff_path"],
        "samples_per_class": config["samples_per_class"],
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info(f"Model + metadata saved: {model_path}")
    return model_path


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="Tampilkan info semua band tanpa training")
    parser.add_argument("--tiff", type=str, default=None,
                        help="Override path TIFF")
    args = parser.parse_args()

    tiff_path = args.tiff or CONFIG["vi_tiff_path"]

    if args.diagnose:
        diagnose_tiff(tiff_path)
        return

    log.info("=" * 60)
    log.info("Bootstrap Labeler + RF Trainer v3")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vi_stack, meta  = load_vi_stack(tiff_path)
    label_map       = generate_pseudo_labels(vi_stack)

    label_meta = meta.copy()
    label_meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_dir / "pseudo_label_map.tif", "w",
                       **label_meta) as dst:
        dst.write(label_map[np.newaxis].astype(np.float32))

    df          = sample_pixels(vi_stack, label_map,
                                CONFIG["samples_per_class"], CONFIG["random_seed"])
    df.to_csv(output_dir / "training_dataset.csv", index=False)

    rf, feature_names, cv_scores = train_random_forest(df, CONFIG)
    evaluate_and_plot(rf, df, feature_names, output_dir)
    model_path  = save_model(rf, feature_names, cv_scores, CONFIG, output_dir)

    log.info("\n" + "=" * 60)
    log.info("SELESAI")
    log.info(f"  OOB  : {rf.oob_score_:.4f}")
    log.info(f"  CV   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    log.info(f"  Model: {model_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
