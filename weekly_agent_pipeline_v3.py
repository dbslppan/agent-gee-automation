"""
Agent Pipeline Mingguan — v3
LPP Agro Nusantara | Madiun, Jawa Timur
========================================
FIX dari v2:
  - predict_health_class: auto-detect band VI dari TIFF
    (tidak hardcode reshape ke 5 band)
  - Reuse load_vi_stack logic dari bootstrap v3:
    → baca dari description → fallback range → hitung dari spektral
  - Chunk processing menyesuaikan jumlah band aktual
  - Log band yang dipakai sebelum inferensi
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling

from health_classes import CLASS_MAP, CLASS_LABELS, COLOR_LOOKUP

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    "drive_watch_dir"   : "./GEE_LPP_MADIUN",
    "model_path"        : "./model_output/model.pkl",
    "output_dir"        : "./agent_output",
    "state_file"        : "./agent_state.json",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "ISI_TOKEN_BOT_KAMU"),
    "telegram_chat_id"  : os.getenv("TELEGRAM_CHAT_ID",   "ISI_CHAT_ID_GRUP"),
    "viewer_url"        : "https://lpp-agro-kebun.streamlit.app",
    "chunk_size"        : 2048,
}

# Nama alias band — sinkron dengan bootstrap v3
BAND_ALIASES = {
    "NDVI" : ["ndvi"],
    "GNDVI": ["gndvi"],
    "SAVI" : ["savi"],
    "RVI"  : ["rvi"],
    "EVI"  : ["evi"],
    "B2"   : ["b2", "blue",  "band2"],
    "B3"   : ["b3", "green", "band3"],
    "B4"   : ["b4", "red",   "band4"],
    "B8"   : ["b8", "nir",   "band8", "nir_broad"],
    "B8A"  : ["b8a", "nir_narrow"],
    "B11"  : ["b11", "swir1"],
    "B12"  : ["b12", "swir2"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ================================================================
# STATE
# ================================================================
def load_state(state_file: str) -> dict:
    p = Path(state_file)
    return json.load(open(p)) if p.exists() else {"processed_files": [], "last_run": None}


def save_state(state: dict, state_file: str):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ================================================================
# STEP 1: POLL DRIVE
# ================================================================
def find_new_tiff(watch_dir: str, state: dict):
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        log.error(f"Watch directory tidak ditemukan: {watch_path}")
        return None

    tiff_files = sorted(
        list(watch_path.glob("*_VI_*.tif")) + list(watch_path.glob("*VI*.tif")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not tiff_files:
        log.warning("Tidak ada TIFF ditemukan.")
        return None

    processed = set(state.get("processed_files", []))
    for p in tiff_files:
        if str(p) not in processed:
            log.info(f"File baru: {p.name}")
            return p

    log.info("Semua file sudah diproses.")
    return None


# ================================================================
# BAND DETECTION — identik dengan bootstrap v3
# ================================================================
def _detect_by_description(src: rasterio.DatasetReader) -> dict:
    positions = {}
    descs = [(src.descriptions[i] or f"band_{i+1}").lower()
             for i in range(src.count)]
    for target, aliases in BAND_ALIASES.items():
        for band_idx, desc in enumerate(descs):
            if any(alias in desc for alias in aliases):
                positions[target] = band_idx + 1
                break
    return positions


def _detect_by_range(src: rasterio.DatasetReader) -> dict:
    positions = {}
    w = min(src.width, 500)
    h = min(src.height, 500)
    col_off = (src.width - w) // 2
    row_off = (src.height - h) // 2
    window  = rasterio.windows.Window(col_off, row_off, w, h)

    stats = []
    for i in range(1, src.count + 1):
        data  = src.read(i, window=window).astype(float)
        nodata = src.nodata if src.nodata is not None else -9999
        valid = data[np.isfinite(data) & (data > nodata + 1) & (data < 1e6)]
        stats.append({
            "idx" : i,
            "mean": float(valid.mean()) if valid.size > 0 else 0,
            "rng" : float(valid.max() - valid.min()) if valid.size > 0 else 0,
        })

    vi_cands = sorted(
        [b for b in stats if -0.5 <= b["mean"] <= 0.95 and b["rng"] < 3.0],
        key=lambda x: x["mean"], reverse=True
    )
    for i, key in enumerate(["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]):
        if i < len(vi_cands):
            positions[key] = vi_cands[i]["idx"]

    spec_used = set(positions.values())
    spec_cands = [b for b in stats
                  if b["idx"] not in spec_used and 0 <= b["mean"] <= 0.6]
    for i, key in enumerate(["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]):
        if i < len(spec_cands):
            positions[key] = spec_cands[i]["idx"]

    return positions


def _read_band_safe(src, band_idx: int) -> np.ndarray:
    data   = src.read(band_idx, out_dtype="float32")
    nodata = src.nodata if src.nodata is not None else -9999
    data   = np.where((data <= nodata + 1) | (~np.isfinite(data)), np.nan, data)
    # Auto-scale DN → reflectance
    valid  = data[np.isfinite(data)]
    if valid.size > 0 and np.abs(valid).mean() > 2:
        data = data / 10000.0
    return data


def _compute_vi_from_spectral(src, band_pos: dict) -> dict:
    """Hitung VI dari band spektral jika VI belum ada."""
    def rb(key):
        return _read_band_safe(src, band_pos[key])

    nir   = rb("B8")
    red   = rb("B4")
    green = rb("B3")
    blue  = rb("B2") if "B2" in band_pos else None

    ndvi  = np.clip((nir - red)   / (nir + red   + 1e-10), -1, 1)
    gndvi = np.clip((nir - green) / (nir + green + 1e-10), -1, 1)
    L     = 0.5
    savi  = np.clip(((nir - red) / (nir + red + L + 1e-10)) * (1 + L), -1, 1)
    rvi   = np.clip(nir / (red + 1e-10), 0, 30)
    if blue is not None:
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
    else:
        evi = 2.5 * (nir - red) / (nir + 6 * red + 1 + 1e-10)
    evi = np.clip(evi, -1, 1)

    return {"NDVI": ndvi, "GNDVI": gndvi, "SAVI": savi, "RVI": rvi, "EVI": evi}


# ================================================================
# STEP 2: INFERENSI RF — chunk-based, band-aware
# ================================================================
def predict_health_class(tiff_path: Path, model_path: str,
                          output_dir: Path) -> tuple:
    log.info(f"Loading model: {model_path}")
    rf = joblib.load(model_path)

    # Baca metadata model untuk tahu feature order
    meta_path = Path(model_path).parent / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model_meta    = json.load(f)
        feature_names = model_meta.get("feature_names",
                                       ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"])
    else:
        feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]

    log.info(f"Feature order: {feature_names}")

    pred_path = output_dir / f"pred_{tiff_path.stem}.tif"

    with rasterio.open(tiff_path) as src:
        raster_meta = src.meta.copy()
        H, W        = src.height, src.width
        log.info(f"Raster: {W}x{H} px, {src.count} bands")

        # ---- Deteksi posisi band ----
        band_pos = _detect_by_description(src)
        if "NDVI" not in band_pos:
            log.warning("NDVI tidak ditemukan dari description. Fallback by range...")
            band_pos = _detect_by_range(src)

        vi_mode = "direct"  # baca VI langsung dari band
        if "NDVI" not in band_pos:
            log.warning("VI tidak ada di TIFF. Akan hitung dari spektral.")
            vi_mode = "compute"

        log.info(f"Mode: {vi_mode} | Band positions: "
                 f"{ {k: v for k, v in band_pos.items() if k in feature_names} }")

        # ---- Jika compute mode: baca semua band spektral sekaligus ----
        if vi_mode == "compute":
            vi_arrays = _compute_vi_from_spectral(src, band_pos)
            # Stack sesuai feature_names
            vi_stack = np.stack(
                [vi_arrays.get(fn, np.zeros((H, W), dtype=np.float32))
                 for fn in feature_names],
                axis=0
            )  # (n_features, H, W)

            # Predict sekaligus (tidak perlu chunk jika RAM cukup)
            # Untuk raster besar, chunk tetap dipakai
            pred_full = _predict_chunked_from_stack(rf, vi_stack, H, W,
                                                    CONFIG["chunk_size"])
        else:
            # ---- Direct mode: baca per chunk, ambil band yang relevan ----
            pred_full = _predict_chunked_direct(
                src, rf, band_pos, feature_names, H, W, CONFIG["chunk_size"]
            )

    # Simpan prediction
    pred_meta = raster_meta.copy()
    pred_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(pred_path, "w", **pred_meta) as dst:
        dst.write(pred_full[np.newaxis])

    log.info(f"Prediction saved: {pred_path}")

    # Distribusi kelas
    total = int(np.sum(pred_full > 0))
    for cls_id, cls_info in CLASS_MAP.items():
        count = int(np.sum(pred_full == cls_id))
        pct   = 100 * count / total if total > 0 else 0
        log.info(f"  {cls_info['name']:10s}: {count:8,} px ({pct:.1f}%)")

    return pred_path, raster_meta


def _predict_chunked_direct(src, rf, band_pos: dict, feature_names: list,
                             H: int, W: int, chunk_size: int) -> np.ndarray:
    """
    Chunk-based inference: baca hanya band yang diperlukan per chunk.
    Tidak hardcode jumlah band.
    """
    pred_full    = np.zeros((H, W), dtype=np.uint8)
    n_features   = len(feature_names)
    total_chunks = (
        ((H + chunk_size - 1) // chunk_size) *
        ((W + chunk_size - 1) // chunk_size)
    )
    chunk_count  = 0

    for row_off in range(0, H, chunk_size):
        row_end = min(row_off + chunk_size, H)
        for col_off in range(0, W, chunk_size):
            col_end     = min(col_off + chunk_size, W)
            chunk_count += 1
            window      = rasterio.windows.Window(
                col_off, row_off, col_end - col_off, row_end - row_off
            )
            ch = row_end - row_off
            cw = col_end - col_off

            # Baca hanya band VI yang dibutuhkan (bukan semua 12 band)
            bands_needed = []
            for fn in feature_names:
                if fn in band_pos:
                    data = src.read(band_pos[fn], window=window,
                                    out_dtype="float32")
                    nodata = src.nodata if src.nodata is not None else -9999
                    data   = np.where(
                        (data <= nodata + 1) | (~np.isfinite(data)), np.nan, data
                    )
                    valid_vals = data[np.isfinite(data)]
                    if valid_vals.size > 0 and np.abs(valid_vals).mean() > 2:
                        data = data / 10000.0
                    bands_needed.append(data.ravel())
                else:
                    bands_needed.append(np.zeros(ch * cw, dtype=np.float32))

            # pixels shape: (n_pixels, n_features)
            pixels = np.stack(bands_needed, axis=1)
            valid  = np.all(np.isfinite(pixels), axis=1)

            pred_chunk = np.zeros(pixels.shape[0], dtype=np.uint8)
            if valid.sum() > 0:
                pred_chunk[valid] = rf.predict(pixels[valid]).astype(np.uint8)

            pred_full[row_off:row_end, col_off:col_end] = \
                pred_chunk.reshape(ch, cw)

            if chunk_count % 10 == 0 or chunk_count == total_chunks:
                log.info(f"  Progress: {100*chunk_count/total_chunks:.0f}% "
                         f"({chunk_count}/{total_chunks})")

    return pred_full


def _predict_chunked_from_stack(rf, vi_stack: np.ndarray,
                                 H: int, W: int, chunk_size: int) -> np.ndarray:
    """Predict dari pre-computed vi_stack (n_features, H, W)."""
    pred_full    = np.zeros((H, W), dtype=np.uint8)
    total_chunks = (
        ((H + chunk_size - 1) // chunk_size) *
        ((W + chunk_size - 1) // chunk_size)
    )
    chunk_count  = 0

    for row_off in range(0, H, chunk_size):
        row_end = min(row_off + chunk_size, H)
        for col_off in range(0, W, chunk_size):
            col_end     = min(col_off + chunk_size, W)
            chunk_count += 1

            chunk  = vi_stack[:, row_off:row_end, col_off:col_end]
            ch, cw = chunk.shape[1], chunk.shape[2]
            pixels = chunk.reshape(chunk.shape[0], -1).T  # (n_px, n_feat)
            valid  = np.all(np.isfinite(pixels), axis=1)

            pred_chunk = np.zeros(pixels.shape[0], dtype=np.uint8)
            if valid.sum() > 0:
                pred_chunk[valid] = rf.predict(pixels[valid]).astype(np.uint8)

            pred_full[row_off:row_end, col_off:col_end] = \
                pred_chunk.reshape(ch, cw)

            if chunk_count % 10 == 0 or chunk_count == total_chunks:
                log.info(f"  Progress: {100*chunk_count/total_chunks:.0f}% "
                         f"({chunk_count}/{total_chunks})")

    return pred_full


# ================================================================
# STEP 3: COLORIZE → COG
# ================================================================
def colorize_prediction(pred_path: Path, output_dir: Path,
                         date_tag: str) -> tuple:
    log.info("Colorizing → COG...")

    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        meta = src.meta.copy()

    H, W = pred.shape
    rgba = np.zeros((4, H, W), dtype=np.uint8)
    for cls_id, color_rgba in COLOR_LOOKUP.items():
        mask = pred == cls_id
        rgba[0][mask], rgba[1][mask] = color_rgba[0], color_rgba[1]
        rgba[2][mask], rgba[3][mask] = color_rgba[2], color_rgba[3]

    rgb_meta = meta.copy()
    rgb_meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    rgb_path = output_dir / f"health_map_rgb_{date_tag}.tif"
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(rgba[:3])

    cog_path = output_dir / f"health_map_COG_{date_tag}.tif"
    cog_meta = rgb_meta.copy()
    cog_meta.update({
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "DEFLATE", "predictor": 2, "interleave": "pixel",
    })
    with rasterio.open(cog_path, "w", **cog_meta) as dst:
        dst.write(rgba[:3])
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    log.info(f"RGB: {rgb_path}")
    log.info(f"COG: {cog_path}")
    return rgb_path, cog_path


# ================================================================
# STEP 4: STATISTIK + PIE CHART
# ================================================================
def generate_stats(pred_path: Path, date_tag: str, output_dir: Path) -> dict:
    with rasterio.open(pred_path) as src:
        pred  = src.read(1)
        res_m = abs(src.res[0])

    pixel_area_ha = (res_m ** 2) / 10000
    total_valid   = int(np.sum(pred > 0))

    stats = {
        "date": date_tag, "resolution_m": res_m,
        "total_valid_ha": 0.0, "classes": {},
        "threshold_def": {
            "Sehat": "NDVI > 0.5", "Sedang": "NDVI 0.2–0.5", "Stres": "NDVI < 0.2"
        },
    }

    for cls_id, cls_info in CLASS_MAP.items():
        count   = int(np.sum(pred == cls_id))
        area_ha = round(count * pixel_area_ha, 2)
        pct     = round(100 * count / total_valid, 2) if total_valid > 0 else 0.0
        stats["classes"][cls_info["name"]] = {
            "class_id": cls_id, "pixel_count": count,
            "area_ha": area_ha, "percentage": pct,
            "color_hex": cls_info["hex"],
        }
        stats["total_valid_ha"] += area_ha
        log.info(f"  {cls_info['name']:10s}: {area_ha:8.1f} ha ({pct:.1f}%)")

    stats["total_valid_ha"] = round(stats["total_valid_ha"], 2)

    with open(output_dir / f"stats_{date_tag}.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Pie chart
    labels, sizes, colors = [], [], []
    for cls_name, d in stats["classes"].items():
        if d["pixel_count"] > 0:
            thr = stats["threshold_def"].get(cls_name, "")
            labels.append(f"{cls_name} ({thr})\n{d['area_ha']:.0f} ha — {d['percentage']:.1f}%")
            sizes.append(d["area_ha"])
            colors.append(d["color_hex"])

    if sizes:
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.legend(wedges, labels, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.set_title(
            f"Kesehatan Kebun — {date_tag}\nTotal: {stats['total_valid_ha']:.0f} ha",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig(output_dir / f"pie_chart_{date_tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    return stats


# ================================================================
# STEP 5: NOTIFIKASI TELEGRAM
# ================================================================
def send_telegram_notification(stats: dict, date_tag: str, config: dict):
    bot_token = config["telegram_bot_token"]
    chat_id   = config["telegram_chat_id"]

    if "ISI_TOKEN" in bot_token or "ISI_CHAT" in chat_id:
        log.warning("Telegram credentials belum diset. Skip.")
        return

    icons = {"Sehat": "🟢", "Sedang": "🟡", "Stres": "🔴"}
    kelas_lines = ""
    for cls_id, cls_info in CLASS_MAP.items():
        name = cls_info["name"]
        d    = stats["classes"].get(name, {})
        thr  = stats["threshold_def"].get(name, "")
        icon = icons.get(name, "⚪")
        kelas_lines += (
            f"{icon} *{name}* \\({thr}\\)\n"
            f"   {d.get('area_ha', 0):,.0f} ha \\- {d.get('percentage', 0):.1f}%\n"
        )

    d_fmt  = f"{date_tag[:4]}\\-{date_tag[4:6]}\\-{date_tag[6:]}"
    message = (
        f"🌿 *Update Peta Kesehatan Kebun*\n"
        f"📅 {d_fmt} \\| 📍 Madiun, Jawa Timur\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{kelas_lines}"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total: {stats['total_valid_ha']:,.0f} ha\n"
        f"🗺️ [Lihat peta]({config['viewer_url']})\n"
        f"_Diproses otomatis oleh SENTINEL\\-KEBUN_"
    )

    url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"
    }, timeout=15)
    log.info(f"Telegram text: {resp.status_code}")

    chart_path = Path(config["output_dir"]) / f"pie_chart_{date_tag}.png"
    if chart_path.exists():
        url2 = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(chart_path, "rb") as img:
            resp2 = requests.post(url2, data={"chat_id": chat_id},
                                  files={"photo": img}, timeout=30)
        log.info(f"Telegram photo: {resp2.status_code}")


# ================================================================
# MAIN
# ================================================================
def run_pipeline():
    t0 = time.time()
    log.info("=" * 60)
    log.info("SENTINEL-KEBUN Agent v3 — Pipeline Mingguan")
    log.info(f"Kelas: {', '.join(c['name'] for c in CLASS_MAP.values())}")
    log.info(f"Run : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state     = load_state(CONFIG["state_file"])
    tiff_path = find_new_tiff(CONFIG["drive_watch_dir"], state)

    if tiff_path is None:
        log.info("Tidak ada file baru. Pipeline selesai.")
        return

    date_tag = datetime.now().strftime("%Y%m%d")

    log.info("\n[Step 2] Inferensi Random Forest...")
    pred_path, raster_meta = predict_health_class(
        tiff_path, CONFIG["model_path"], output_dir
    )

    log.info("\n[Step 3] Colorize → COG...")
    rgb_path, cog_path = colorize_prediction(pred_path, output_dir, date_tag)

    log.info("\n[Step 4] Statistik...")
    stats = generate_stats(pred_path, date_tag, output_dir)

    log.info("\n[Step 5] Notifikasi Telegram...")
    send_telegram_notification(stats, date_tag, CONFIG)

    state.setdefault("processed_files", []).append(str(tiff_path))
    state["last_run"]      = datetime.now().isoformat()
    state["last_cog_path"] = str(cog_path)
    save_state(state, CONFIG["state_file"])

    log.info("\n" + "=" * 60)
    log.info(f"Selesai dalam {time.time()-t0:.1f} detik")
    log.info(f"  COG   : health_map_COG_{date_tag}.tif")
    log.info(f"  Stats : stats_{date_tag}.json")
    log.info(f"  Chart : pie_chart_{date_tag}.png")
    log.info("Lanjut: python clip_tile_agent.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
