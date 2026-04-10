# weekly_agent_pipeline.py

"""
Agent Pipeline Mingguan — Klasifikasi Kesehatan Kebun
LPP Agro Nusantara | Madiun, Jawa Timur
=====================================================
Alur otomatis setiap 5-7 hari:
  1. Poll folder GEE -> deteksi GeoTIFF VI baru
  2. Load model.pkl -> inferensi RF per piksel
  3. Colorize peta kelas -> Cloud Optimized GeoTIFF (COG)
  4. Generate ringkasan statistik per kelas (JSON + PNG)
  5. Kirim notifikasi Telegram ke grup manajemen

Jalankan manual:
  python weekly_agent_pipeline.py

Untuk cron (setiap Senin 06:00 WIB):
  0 23 * * 0 /usr/bin/python3 /path/to/weekly_agent_pipeline.py
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

# ----------------------------------------------------------------
# IMPORT DARI health_classes.py — sumber tunggal definisi kelas
# ----------------------------------------------------------------
from health_classes import CLASS_MAP, CLASS_LABELS

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    # === PATH ===
    "drive_watch_dir"  : "./GEE_LPP_MADIUN",
    "model_path"       : "./model_output/model.pkl",
    "output_dir"       : "./agent_output",
    "state_file"       : "./agent_state.json",

    # === IDENTITAS KEBUN ===
    "kebun_nama"  : "LPP Agro Nusantara",
    "kebun_area"  : "Madiun, Jawa Timur",
    "kebun_agent" : "SENTINEL-KEBUN Agent",

    # === TELEGRAM ===
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "8666397523:AAGqFgF7jnCV3Ih7tKcirlbt6hgbqQe7V44"),
    "telegram_chat_id"  : os.getenv("TELEGRAM_CHAT_ID",  "-1003840524953"),

    # === STREAMLIT VIEWER ===
    "viewer_url": "https://lppagro-kebun.streamlit.app",

    # === PROCESSING ===
    "chunk_size": 2048,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ================================================================
# UTILITAS STATE — deduplication
# ================================================================
def load_state(state_file: str) -> dict:
    p = Path(state_file)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"processed_files": [], "last_run": None}


def save_state(state: dict, state_file: str):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ================================================================
# STEP 1: POLL DRIVE — temukan GeoTIFF VI baru
# ================================================================
def find_new_tiff(watch_dir: str, state: dict) -> Path | None:
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
        log.warning("Tidak ada file GeoTIFF VI ditemukan di watch directory.")
        return None

    processed = set(state.get("processed_files", []))
    for tiff_path in tiff_files:
        if str(tiff_path) not in processed:
            log.info(f"File baru ditemukan: {tiff_path.name}")
            return tiff_path

    log.info("Semua file sudah diproses sebelumnya. Tidak ada yang baru.")
    return None


# ================================================================
# STEP 2: INFERENSI RF — predict kelas per piksel
# ================================================================
def predict_health_class(tiff_path: Path, model_path: str,
                          output_dir: Path) -> tuple[Path, dict]:
    """
    Load GeoTIFF 12-band Sentinel-2 -> hitung VI -> predict kelas RF.
    Diproses per chunk untuk menghindari OOM pada file besar.
    """
    log.info(f"Loading model: {model_path}")
    rf   = joblib.load(model_path)
    pred_path = output_dir / f"pred_{tiff_path.stem}.tif"

    with rasterio.open(tiff_path) as src:
        meta = src.meta.copy()
        H, W = src.height, src.width
        log.info(f"Memproses raster {W}x{H} piksel...")

        pred_full  = np.zeros((H, W), dtype=np.uint8)
        chunk_size = CONFIG["chunk_size"]
        total_chunks = (((H + chunk_size - 1) // chunk_size) *
                        ((W + chunk_size - 1) // chunk_size))
        chunk_count = 0

        for row_off in range(0, H, chunk_size):
            row_end = min(row_off + chunk_size, H)
            for col_off in range(0, W, chunk_size):
                col_end = min(col_off + chunk_size, W)
                chunk_count += 1

                window = rasterio.windows.Window(
                    col_off, row_off,
                    col_end - col_off,
                    row_end - row_off
                )

                # Baca hanya band yang dibutuhkan (1-based: B2,B3,B4,B8)
                raw = src.read([2, 3, 4, 8], window=window,
                               out_dtype="float32")  # (4, ch, cw)
                ch, cw = raw.shape[1], raw.shape[2]

                # Konversi DN -> reflectance
                B2 = raw[0] / 10000.0
                B3 = raw[1] / 10000.0
                B4 = raw[2] / 10000.0
                B8 = raw[3] / 10000.0
                eps = 1e-10

                # Hitung VI
                NDVI  = (B8 - B4) / (B8 + B4 + eps)
                GNDVI = (B8 - B3) / (B8 + B3 + eps)
                L     = 0.5
                SAVI  = ((B8 - B4) / (B8 + B4 + L + eps)) * (1 + L)
                RVI   = B8 / (B4 + eps)
                EVI   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)

                # Stack -> reshape ke (n_pixels, 5)
                vi_chunk = np.stack([NDVI, GNDVI, SAVI, RVI, EVI], axis=0)
                pixels   = vi_chunk.reshape(5, -1).T  # (n, 5)

                # Mask piksel nodata (semua band raw = 0)
                raw_pixels = raw.reshape(4, -1).T
                valid = (
                    np.all(np.isfinite(pixels) & (np.abs(pixels) < 10), axis=1)
                    & np.any(raw_pixels > 0, axis=1)
                )

                pred_chunk = np.zeros(pixels.shape[0], dtype=np.uint8)
                if valid.sum() > 0:
                    pred_chunk[valid] = rf.predict(pixels[valid]).astype(np.uint8)

                pred_full[row_off:row_end, col_off:col_end] = \
                    pred_chunk.reshape(ch, cw)

                if chunk_count % 10 == 0:
                    pct = 100 * chunk_count / total_chunks
                    log.info(f"  Progress: {pct:.0f}% ({chunk_count}/{total_chunks})")

    # Simpan prediction GeoTIFF
    pred_meta = meta.copy()
    pred_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(pred_path, "w", **pred_meta) as dst:
        dst.write(pred_full[np.newaxis, :, :])

    log.info(f"Prediction raster saved: {pred_path}")
    return pred_path, meta


# ================================================================
# STEP 3: COLORIZE -> RGB GeoTIFF + COG
# ================================================================
def colorize_prediction(pred_path: Path, output_dir: Path,
                         date_tag: str) -> tuple[Path, Path]:
    """Konversi prediction raster (1-band uint8) -> RGB GeoTIFF + COG."""
    log.info("Colorizing prediction raster...")

    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        meta = src.meta.copy()

    H, W = pred.shape
    rgb  = np.zeros((3, H, W), dtype=np.uint8)

    for cls_id, cls_info in CLASS_MAP.items():
        mask = pred == cls_id
        r, g, b = cls_info["color"]
        rgb[0][mask] = r
        rgb[1][mask] = g
        rgb[2][mask] = b

    # Nodata (0) -> hitam
    rgb[:, pred == 0] = 0

    # Simpan RGB GeoTIFF
    rgb_meta = meta.copy()
    rgb_meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    rgb_path = output_dir / f"health_map_rgb_{date_tag}.tif"
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(rgb)
    log.info(f"RGB GeoTIFF saved: {rgb_path}")

    # COG
    cog_path = output_dir / f"health_map_COG_{date_tag}.tif"
    _write_cog(rgb, rgb_meta, cog_path)

    return rgb_path, cog_path


def _write_cog(data: np.ndarray, meta: dict, out_path: Path):
    """Tulis Cloud Optimized GeoTIFF dengan internal tiling + overviews."""
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
# STEP 4: STATISTIK RINGKASAN
# ================================================================
def generate_stats(pred_path: Path, date_tag: str, output_dir: Path) -> dict:
    """Hitung luas per kelas (ha) + persentase + buat pie chart."""
    with rasterio.open(pred_path) as src:
        pred   = src.read(1)
        res    = src.res
        crs    = src.crs
        bounds = src.bounds

    # Hitung luas piksel — handle CRS geografis (derajat) vs projected (meter)
    if crs.is_geographic:
        lat_center     = (bounds.top + bounds.bottom) / 2
        pixel_height_m = res[0] * 111320.0
        pixel_width_m  = res[1] * 111320.0 * np.cos(np.radians(lat_center))
        pixel_area_ha  = (pixel_height_m * pixel_width_m) / 10000
        res_display_m  = (pixel_height_m + pixel_width_m) / 2
    else:
        pixel_area_ha = (res[0] * res[1]) / 10000
        res_display_m = res[0]

    stats = {
        "date"          : date_tag,
        "resolution_m"  : round(res_display_m, 1),
        "crs"           : str(crs),
        "classes"       : {},
        "total_valid_ha": 0.0,
    }

    total_valid = np.sum(pred > 0)

    for cls_id, cls_info in CLASS_MAP.items():
        count   = int(np.sum(pred == cls_id))
        area_ha = count * pixel_area_ha
        pct     = 100 * count / total_valid if total_valid > 0 else 0.0

        stats["classes"][cls_info["name"]] = {
            "class_id"   : cls_id,
            "pixel_count": count,
            "area_ha"    : round(area_ha, 2),
            "percentage" : round(pct, 2),
            "color_hex"  : cls_info["hex"],
        }
        stats["total_valid_ha"] += area_ha
        log.info(f"  {cls_info['name']:10s}: {area_ha:8.1f} ha  ({pct:.1f}%)")

    stats["total_valid_ha"] = round(stats["total_valid_ha"], 2)

    stats_path = output_dir / f"stats_{date_tag}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    _plot_pie_chart(stats, date_tag, output_dir)
    return stats


def _plot_pie_chart(stats: dict, date_tag: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    labels, sizes, colors = [], [], []
    for cls_name, data in stats["classes"].items():
        if data["pixel_count"] > 0 and data["area_ha"] > 0:
            labels.append(f"{cls_name}\n{data['area_ha']:.0f} ha "
                          f"({data['percentage']:.1f}%)")
            sizes.append(data["area_ha"])
            colors.append(data["color_hex"])

    if not labels:
        ax.text(0.5, 0.5, "Tidak ada data valid",
                ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(output_dir / f"pie_chart_{date_tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    wedges, _ = ax.pie(
        sizes, labels=None, colors=colors,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Distribusi Kesehatan Kebun\n"
        f"{CONFIG['kebun_area']} - {date_tag}\n"
        f"Total: {stats['total_valid_ha']:.0f} ha",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    chart_path = output_dir / f"pie_chart_{date_tag}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Pie chart saved: {chart_path}")


# ================================================================
# STEP 5: NOTIFIKASI TELEGRAM
# ================================================================
def send_telegram_notification(stats: dict, cog_path: Path,
                                date_tag: str, config: dict):
    """Kirim pesan + pie chart ke grup Telegram manajemen."""
    bot_token = config["telegram_bot_token"]
    chat_id   = config["telegram_chat_id"]

    if "ISI_TOKEN" in bot_token or "ISI_CHAT" in chat_id:
        log.warning("Telegram credentials belum diset. Skip notifikasi.")
        return

    cls_data = stats["classes"]

    # Bangun baris pesan dinamis dari CLASS_MAP
    # (otomatis menyesuaikan jumlah dan nama kelas dari health_classes.py)
    icon_map = {
        "Sehat" : "Sehat",
        "Sedang": "Sedang",
        "Stres" : "Stres",
    }
    emoji_map = {1: "🟢", 2: "🟡", 3: "🔴"}

    kelas_lines = ""
    for cls_id, cls_info in CLASS_MAP.items():
        cls_name = cls_info["name"]
        data     = cls_data.get(cls_name, {})
        emoji    = emoji_map.get(cls_id, "⚪")
        kelas_lines += (
            f"{emoji} {cls_name:<12}: {data.get('area_ha', 0):.0f} ha "
            f"({data.get('percentage', 0):.1f}%)\n"
        )

    message = (
        f"🌿 *Update Peta Kesehatan Kebun*\n"
        f"📅 Periode: {date_tag}\n"
        f"📍 Area: {config['kebun_area']}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{kelas_lines}"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total area   : {stats['total_valid_ha']:.0f} ha\n"
        f"🗺 Lihat peta   : {config['viewer_url']}\n"
        f"_Diproses otomatis oleh {config['kebun_agent']}_"
    )

    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url_msg, json={
        "chat_id"   : chat_id,
        "text"      : message,
        "parse_mode": "Markdown",
    }, timeout=15)

    if resp.status_code == 200:
        log.info("Telegram text message sent OK")
    else:
        log.error(f"Telegram error: {resp.status_code} - {resp.text}")

    # Kirim pie chart
    chart_path = Path(config["output_dir"]) / f"pie_chart_{date_tag}.png"
    if chart_path.exists():
        url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(chart_path, "rb") as img:
            resp2 = requests.post(url_photo, data={"chat_id": chat_id},
                                  files={"photo": img}, timeout=30)
        if resp2.status_code == 200:
            log.info("Telegram chart photo sent OK")
        else:
            log.error(f"Telegram photo error: {resp2.status_code}")


# ================================================================
# MAIN PIPELINE
# ================================================================
def run_pipeline():
    t_start = time.time()
    log.info("=" * 60)
    log.info(f"SENTINEL-KEBUN Agent - Pipeline Mingguan")
    log.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Kelas aktif: {[v['name'] for v in CLASS_MAP.values()]}")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(CONFIG["state_file"])

    # Step 1: Cari file baru
    tiff_path = find_new_tiff(CONFIG["drive_watch_dir"], state)
    if tiff_path is None:
        log.info("Tidak ada file baru. Pipeline selesai.")
        return

    date_tag = datetime.now().strftime("%Y%m%d")

    # Step 2: Inferensi RF
    log.info("\n[Step 2] Inferensi Random Forest...")
    pred_path, raster_meta = predict_health_class(
        tiff_path, CONFIG["model_path"], output_dir
    )

    # Step 3: Colorize -> COG
    log.info("\n[Step 3] Colorize -> Cloud Optimized GeoTIFF...")
    rgb_path, cog_path = colorize_prediction(pred_path, output_dir, date_tag)

    # Step 4: Statistik
    log.info("\n[Step 4] Hitung statistik luas per kelas...")
    stats = generate_stats(pred_path, date_tag, output_dir)

    # Step 5: Notifikasi Telegram
    log.info("\n[Step 5] Kirim notifikasi Telegram...")
    send_telegram_notification(stats, cog_path, date_tag, CONFIG)

    # Update state
    state.setdefault("processed_files", []).append(str(tiff_path))
    state["last_run"]      = datetime.now().isoformat()
    state["last_cog_path"] = str(cog_path)
    save_state(state, CONFIG["state_file"])

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info(f"Pipeline selesai dalam {elapsed:.1f} detik")
    log.info(f"Output tersimpan di: {output_dir}/")
    log.info(f"  health_map_COG_{date_tag}.tif  -> upload ke hosting/Streamlit")
    log.info(f"  pie_chart_{date_tag}.png        -> sudah dikirim ke Telegram")
    log.info(f"  stats_{date_tag}.json           -> dibaca oleh Streamlit viewer")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()