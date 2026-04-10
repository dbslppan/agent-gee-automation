"""
Agent Pipeline Mingguan — v2
LPP Agro Nusantara | Madiun, Jawa Timur
========================================
PERUBAHAN dari v1:
  - CLASS_MAP & COLOR_LOOKUP diimport dari health_classes.py (3 kelas)
  - Colorize pakai COLOR_LOOKUP dari health_classes (tidak hardcode)
  - Statistik & notifikasi Telegram menyesuaikan 3 kelas
  - Pesan Telegram menampilkan threshold NDVI per kelas
  - Semua hardcode warna/kelas dihapus

Instalasi:
  pip install rasterio numpy pandas scikit-learn joblib \
              matplotlib pillow requests rio-cogeo
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling

# ── import definisi kelas terpusat ──────────────────────────────
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
    "nodata_value"      : -9999.0,
    "chunk_size"        : 2048,
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
        watch_path.glob("*_VI_*.tif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not tiff_files:
        log.warning("Tidak ada file GeoTIFF VI ditemukan.")
        return None

    processed = set(state.get("processed_files", []))
    for tiff_path in tiff_files:
        if str(tiff_path) not in processed:
            log.info(f"File baru ditemukan: {tiff_path.name}")
            return tiff_path

    log.info("Semua file sudah diproses. Tidak ada yang baru.")
    return None


# ================================================================
# STEP 2: INFERENSI RF (chunked)
# ================================================================
def predict_health_class(tiff_path: Path, model_path: str,
                          output_dir: Path) -> tuple[Path, dict]:
    log.info(f"Loading model: {model_path}")
    rf = joblib.load(model_path)

    pred_path = output_dir / f"pred_{tiff_path.stem}.tif"

    with rasterio.open(tiff_path) as src:
        meta  = src.meta.copy()
        H, W  = src.height, src.width
        log.info(f"Raster: {W}x{H} px — proses per chunk {CONFIG['chunk_size']}px")

        pred_full   = np.zeros((H, W), dtype=np.uint8)
        chunk_size  = CONFIG["chunk_size"]
        total_chunks = (
            ((H + chunk_size - 1) // chunk_size) *
            ((W + chunk_size - 1) // chunk_size)
        )
        chunk_count = 0

        for row_off in range(0, H, chunk_size):
            row_end = min(row_off + chunk_size, H)
            for col_off in range(0, W, chunk_size):
                col_end = min(col_off + chunk_size, W)
                chunk_count += 1

                window = rasterio.windows.Window(
                    col_off, row_off,
                    col_end - col_off, row_end - row_off
                )
                chunk = src.read(window=window, out_dtype="float32")
                ch, cw = chunk.shape[1], chunk.shape[2]

                pixels = chunk.reshape(5, -1).T
                valid  = np.all(np.isfinite(pixels) & (pixels > -9999), axis=1)

                pred_chunk = np.zeros(pixels.shape[0], dtype=np.uint8)
                if valid.sum() > 0:
                    pred_chunk[valid] = rf.predict(pixels[valid]).astype(np.uint8)

                pred_full[row_off:row_end, col_off:col_end] = \
                    pred_chunk.reshape(ch, cw)

                if chunk_count % 10 == 0:
                    log.info(f"  {100*chunk_count/total_chunks:.0f}% "
                             f"({chunk_count}/{total_chunks})")

    pred_meta = meta.copy()
    pred_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(pred_path, "w", **pred_meta) as dst:
        dst.write(pred_full[np.newaxis])

    log.info(f"Prediction raster: {pred_path}")
    return pred_path, meta


# ================================================================
# STEP 3: COLORIZE → RGB + COG
# ================================================================
def colorize_prediction(pred_path: Path, output_dir: Path,
                         date_tag: str) -> tuple[Path, Path]:
    log.info("Colorizing → COG...")

    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        meta = src.meta.copy()

    H, W = pred.shape
    rgba = np.zeros((4, H, W), dtype=np.uint8)

    # Gunakan COLOR_LOOKUP dari health_classes.py (3 kelas + nodata)
    for cls_id, color_rgba in COLOR_LOOKUP.items():
        mask = pred == cls_id
        rgba[0][mask] = color_rgba[0]  # R
        rgba[1][mask] = color_rgba[1]  # G
        rgba[2][mask] = color_rgba[2]  # B
        rgba[3][mask] = color_rgba[3]  # A

    # RGB GeoTIFF (tanpa alpha — untuk kompatibilitas luas)
    rgb_meta = meta.copy()
    rgb_meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    rgb_path = output_dir / f"health_map_rgb_{date_tag}.tif"
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(rgba[:3])

    # COG
    cog_path = output_dir / f"health_map_COG_{date_tag}.tif"
    _write_cog(rgba[:3], rgb_meta, cog_path)

    return rgb_path, cog_path


def _write_cog(data: np.ndarray, meta: dict, out_path: Path):
    cog_meta = meta.copy()
    cog_meta.update({
        "driver"    : "GTiff",
        "tiled"     : True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress"  : "DEFLATE",
        "predictor" : 2,
        "interleave": "pixel",
    })
    with rasterio.open(out_path, "w", **cog_meta) as dst:
        dst.write(data)
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    log.info(f"COG saved: {out_path}")


# ================================================================
# STEP 4: STATISTIK
# ================================================================
def generate_stats(pred_path: Path, date_tag: str,
                   output_dir: Path) -> dict:
    with rasterio.open(pred_path) as src:
        pred  = src.read(1)
        res_m = abs(src.res[0])

    pixel_area_ha = (res_m ** 2) / 10000
    total_valid   = int(np.sum(pred > 0))

    stats = {
        "date"          : date_tag,
        "resolution_m"  : res_m,
        "total_valid_ha": 0.0,
        "classes"       : {},
        "threshold_def" : {
            "Sehat" : "NDVI > 0.5",
            "Sedang": "NDVI 0.2–0.5",
            "Stres" : "NDVI < 0.2",
        },
    }

    for cls_id, cls_info in CLASS_MAP.items():
        count   = int(np.sum(pred == cls_id))
        area_ha = round(count * pixel_area_ha, 2)
        pct     = round(100 * count / total_valid, 2) if total_valid > 0 else 0.0

        stats["classes"][cls_info["name"]] = {
            "class_id"   : cls_id,
            "pixel_count": count,
            "area_ha"    : area_ha,
            "percentage" : pct,
            "color_hex"  : cls_info["hex"],
        }
        stats["total_valid_ha"] += area_ha
        log.info(f"  {cls_info['name']:10s}: {area_ha:8.1f} ha  ({pct:.1f}%)")

    stats["total_valid_ha"] = round(stats["total_valid_ha"], 2)

    # Simpan JSON
    stats_path = output_dir / f"stats_{date_tag}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    _plot_pie_chart(stats, date_tag, output_dir)
    return stats


def _plot_pie_chart(stats: dict, date_tag: str, output_dir: Path):
    labels, sizes, colors = [], [], []
    for cls_name, data in stats["classes"].items():
        if data["pixel_count"] > 0:
            thr = stats["threshold_def"].get(cls_name, "")
            labels.append(
                f"{cls_name} ({thr})\n"
                f"{data['area_ha']:.0f} ha — {data['percentage']:.1f}%"
            )
            sizes.append(data["area_ha"])
            colors.append(data["color_hex"])

    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, _ = ax.pie(
        sizes, labels=None, colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Distribusi Kesehatan Kebun — {date_tag}\n"
        f"Total: {stats['total_valid_ha']:.0f} ha",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    chart_path = output_dir / f"pie_chart_{date_tag}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Pie chart: {chart_path}")


# ================================================================
# STEP 5: NOTIFIKASI TELEGRAM
# ================================================================
def send_telegram_notification(stats: dict, date_tag: str, config: dict):
    bot_token = config["telegram_bot_token"]
    chat_id   = config["telegram_chat_id"]

    if "ISI_TOKEN" in bot_token or "ISI_CHAT" in chat_id:
        log.warning("Telegram credentials belum diset. Skip.")
        return

    cls_data = stats["classes"]

    # Bangun baris per kelas secara dinamis dari CLASS_MAP
    kelas_lines = ""
    icons = {"Sehat": "🟢", "Sedang": "🟡", "Stres": "🔴"}
    for cls_id, cls_info in CLASS_MAP.items():
        name = cls_info["name"]
        d    = cls_data.get(name, {})
        thr  = stats["threshold_def"].get(name, "")
        icon = icons.get(name, "⚪")
        kelas_lines += (
            f"{icon} *{name}* `({thr})`\n"
            f"   {d.get('area_ha', 0):,.0f} ha — {d.get('percentage', 0):.1f}%\n"
        )

    message = (
        f"🌿 *Update Peta Kesehatan Kebun*\n"
        f"📅 Tanggal: {date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}\n"
        f"📍 Area: Madiun, Jawa Timur\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{kelas_lines}"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total area: {stats['total_valid_ha']:,.0f} ha\n"
        f"🗺️ Lihat peta: {config['viewer_url']}\n"
        f"_Diproses otomatis oleh SENTINEL\\-KEBUN Agent_"
    )

    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url_msg, json={
        "chat_id"   : chat_id,
        "text"      : message,
        "parse_mode": "MarkdownV2",
    }, timeout=15)

    if resp.status_code == 200:
        log.info("Telegram message sent OK")
    else:
        log.error(f"Telegram error: {resp.status_code} — {resp.text}")

    # Kirim pie chart
    chart_path = Path(config["output_dir"]) / f"pie_chart_{date_tag}.png"
    if chart_path.exists():
        url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(chart_path, "rb") as img:
            resp2 = requests.post(url_photo,
                                  data={"chat_id": chat_id},
                                  files={"photo": img}, timeout=30)
        if resp2.status_code == 200:
            log.info("Telegram photo sent OK")
        else:
            log.error(f"Telegram photo error: {resp2.status_code}")


# ================================================================
# MAIN
# ================================================================
def run_pipeline():
    t_start = time.time()
    log.info("=" * 60)
    log.info("SENTINEL-KEBUN Agent v2 — Pipeline Mingguan (3 Kelas)")
    log.info(f"Kelas: {', '.join(c['name'] for c in CLASS_MAP.values())}")
    log.info(f"Run : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(CONFIG["state_file"])

    tiff_path = find_new_tiff(CONFIG["drive_watch_dir"], state)
    if tiff_path is None:
        log.info("Tidak ada file baru. Pipeline selesai.")
        return

    date_tag = datetime.now().strftime("%Y%m%d")

    log.info("\n[Step 2] Inferensi Random Forest...")
    pred_path, raster_meta = predict_health_class(
        tiff_path, CONFIG["model_path"], output_dir
    )

    log.info("\n[Step 3] Colorize → Cloud Optimized GeoTIFF...")
    rgb_path, cog_path = colorize_prediction(pred_path, output_dir, date_tag)

    log.info("\n[Step 4] Statistik per kelas...")
    stats = generate_stats(pred_path, date_tag, output_dir)

    log.info("\n[Step 5] Notifikasi Telegram...")
    send_telegram_notification(stats, date_tag, CONFIG)

    state.setdefault("processed_files", []).append(str(tiff_path))
    state["last_run"]      = datetime.now().isoformat()
    state["last_cog_path"] = str(cog_path)
    save_state(state, CONFIG["state_file"])

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info(f"Pipeline selesai dalam {elapsed:.1f} detik")
    log.info(f"  health_map_COG_{date_tag}.tif → upload ke hosting")
    log.info(f"  stats_{date_tag}.json          → dibaca viewer")
    log.info(f"  pie_chart_{date_tag}.png        → sudah ke Telegram")
    log.info("\nLanjutkan dengan: python clip_tile_agent.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
