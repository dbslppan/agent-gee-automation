# weekly_agent_pipeline.py

"""
Agent Pipeline Mingguan — Klasifikasi Kesehatan Kebun
LPP Agro Nusantara | Madiun, Jawa Timur
=====================================================
Alur otomatis setiap 5–7 hari:
  1. Poll Google Drive → deteksi GeoTIFF VI baru dari GEE
  2. Load model.pkl → inferensi RF per piksel
  3. Colorize peta kelas → Cloud Optimized GeoTIFF (COG)
  4. Generate ringkasan statistik per kelas (JSON + PNG)
  5. Upload hasil ke folder output / hosting
  6. Kirim notifikasi Telegram ke grup manajemen

Instalasi:
  pip install rasterio numpy pandas scikit-learn joblib \
              matplotlib pillow google-api-python-client \
              google-auth rio-cogeo requests

Jalankan manual:
  python weekly_agent_pipeline.py

Untuk cron (setiap Senin 06:00 WIB = 23:00 UTC Minggu):
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
import matplotlib.patches as mpatches
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

# ----------------------------------------------------------------
# KONFIGURASI — sesuaikan semua nilai di sini
# ----------------------------------------------------------------
CONFIG = {
    # === PATH ===
    # Folder lokal dimana GEE export di-sync dari Google Drive
    "drive_watch_dir"  : "./GEE_LPP_MADIUN",
    # Model RF hasil bootstrap labeler
    "model_path"       : "./model_output/model.pkl",
    # Folder output peta + statistik
    "output_dir"       : "./agent_output",
    # State file untuk deduplication
    "state_file"       : "./agent_state.json",

    # === TELEGRAM ===
    # Dapatkan BOT_TOKEN dari @BotFather di Telegram
    # Dapatkan CHAT_ID dari @userinfobot setelah bot join grup
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "ISI_TOKEN_BOT_KAMU"),
    "telegram_chat_id"  : os.getenv("TELEGRAM_CHAT_ID",  "ISI_CHAT_ID_GRUP"),

    # === STREAMLIT VIEWER ===
    # URL viewer yang akan dikirim di notifikasi
    "viewer_url": "https://lpp-agro-kebun.streamlit.app",

    # === PROCESSING ===
    "nodata_value"     : -9999.0,
    "chunk_size"       : 2048,   # piksel per chunk (memory management)
}

# Definisi kelas — harus sinkron dengan bootstrap_labeler
CLASS_MAP = {
    1: {"name": "Sehat",        "color": (46,  204, 64),  "hex": "#2ecc40"},
    2: {"name": "Stres Ringan", "color": (255, 220,  0),  "hex": "#ffdc00"},
    3: {"name": "Stres Berat",  "color": (255, 133, 27),  "hex": "#ff851b"},
    4: {"name": "Kritis",       "color": (231, 76,  60),  "hex": "#e74c3c"},
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
    """
    Scan folder lokal untuk GeoTIFF VI terbaru yang belum diproses.
    Urutkan berdasarkan tanggal modifikasi, ambil yang paling baru.
    """
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        log.error(f"Watch directory tidak ditemukan: {watch_path}")
        return None

    # Cari semua file VI (bukan Bands)
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
    Load GeoTIFF VI → predict kelas dengan RF → simpan sebagai GeoTIFF.

    Diproses per chunk untuk menghindari OOM pada file besar.
    Returns: (pred_tiff_path, raster_meta)
    """
    log.info(f"Loading model: {model_path}")
    rf = joblib.load(model_path)
    feature_names = ["NDVI", "GNDVI", "SAVI", "RVI", "EVI"]

    pred_path = output_dir / f"pred_{tiff_path.stem}.tif"

    with rasterio.open(tiff_path) as src:
        meta = src.meta.copy()
        H, W = src.height, src.width
        log.info(f"Memproses raster {W}x{H} piksel...")

        # Siapkan output array
        pred_full = np.zeros((H, W), dtype=np.uint8)

        chunk_size = CONFIG["chunk_size"]
        total_chunks = ((H + chunk_size - 1) // chunk_size) * \
                       ((W + chunk_size - 1) // chunk_size)
        chunk_count = 0

        for row_off in range(0, H, chunk_size):
            row_end = min(row_off + chunk_size, H)
            for col_off in range(0, W, chunk_size):
                col_end = min(col_off + chunk_size, W)
                chunk_count += 1

                # Baca chunk 5 band
                window = rasterio.windows.Window(
                    col_off, row_off,
                    col_end - col_off,
                    row_end - row_off
                )
                chunk = src.read(window=window, out_dtype="float32")  # (5, ch, cw)
                ch, cw = chunk.shape[1], chunk.shape[2]

                # Reshape ke (n_pixels, n_features)
                pixels = chunk.reshape(5, -1).T  # (n, 5)

                # Mask NaN/nodata
                valid = np.all(
                    np.isfinite(pixels) & (pixels > -9999),
                    axis=1
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
# STEP 3: COLORIZE → RGB GeoTIFF + COG
# ================================================================
def colorize_prediction(pred_path: Path, output_dir: Path,
                         date_tag: str) -> tuple[Path, Path]:
    """
    Konversi prediction raster (1-band uint8) →
      - RGB GeoTIFF (untuk tampilan langsung)
      - Cloud Optimized GeoTIFF / COG (untuk web viewer tile)
    """
    log.info("Colorizing prediction raster...")

    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        meta = src.meta.copy()

    H, W = pred.shape
    rgb = np.zeros((3, H, W), dtype=np.uint8)

    for cls_id, cls_info in CLASS_MAP.items():
        mask = pred == cls_id
        r, g, b = cls_info["color"]
        rgb[0][mask] = r
        rgb[1][mask] = g
        rgb[2][mask] = b

    # Piksel nodata (0) → hitam transparan
    nodata_mask = pred == 0
    rgb[:, nodata_mask] = 0

    # Simpan RGB GeoTIFF
    rgb_meta = meta.copy()
    rgb_meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    rgb_path = output_dir / f"health_map_rgb_{date_tag}.tif"
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(rgb)
    log.info(f"RGB GeoTIFF saved: {rgb_path}")

    # Buat COG menggunakan rasterio (manual tiling + overviews)
    # (Alternatif: pakai `rio cogeo create` CLI jika rio-cogeo terinstall)
    cog_path = output_dir / f"health_map_COG_{date_tag}.tif"
    _write_cog(rgb, rgb_meta, cog_path)

    return rgb_path, cog_path


def _write_cog(data: np.ndarray, meta: dict, out_path: Path):
    """Tulis Cloud Optimized GeoTIFF dengan internal tiling + overviews."""
    cog_meta = meta.copy()
    cog_meta.update({
        "driver"  : "GTiff",
        "tiled"   : True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
        "predictor": 2,
        "interleave": "pixel",
    })

    with rasterio.open(out_path, "w", **cog_meta) as dst:
        dst.write(data)
        # Build overview levels: 2x, 4x, 8x, 16x, 32x
        overview_levels = [2, 4, 8, 16, 32]
        dst.build_overviews(overview_levels, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    log.info(f"COG saved: {out_path}")


# ================================================================
# STEP 4: STATISTIK RINGKASAN
# ================================================================
def generate_stats(pred_path: Path, date_tag: str,
                   output_dir: Path) -> dict:
    """
    Hitung luas per kelas (ha) + persentase + buat pie chart.
    """
    with rasterio.open(pred_path) as src:
        pred  = src.read(1)
        res_m = abs(src.res[0])  # resolusi dalam meter

    pixel_area_ha = (res_m ** 2) / 10000  # m² → ha

    stats = {
        "date"       : date_tag,
        "resolution_m": res_m,
        "classes"    : {},
        "total_valid_ha": 0.0,
    }

    total_valid = np.sum(pred > 0)

    for cls_id, cls_info in CLASS_MAP.items():
        count = int(np.sum(pred == cls_id))
        area_ha = count * pixel_area_ha
        pct = 100 * count / total_valid if total_valid > 0 else 0.0

        stats["classes"][cls_info["name"]] = {
            "class_id"   : cls_id,
            "pixel_count": count,
            "area_ha"    : round(area_ha, 2),
            "percentage" : round(pct, 2),
            "color_hex"  : cls_info["hex"],
        }
        stats["total_valid_ha"] += area_ha
        log.info(f"  {cls_info['name']:15s}: {area_ha:8.1f} ha  ({pct:.1f}%)")

    stats["total_valid_ha"] = round(stats["total_valid_ha"], 2)

    # Simpan JSON
    stats_path = output_dir / f"stats_{date_tag}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Buat pie chart
    _plot_pie_chart(stats, date_tag, output_dir)

    return stats


def _plot_pie_chart(stats: dict, date_tag: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    labels, sizes, colors = [], [], []
    for cls_name, data in stats["classes"].items():
        if data["pixel_count"] > 0:
            labels.append(f"{cls_name}\n{data['area_ha']:.0f} ha ({data['percentage']:.1f}%)")
            sizes.append(data["area_ha"])
            colors.append(data["color_hex"])

    wedges, texts = ax.pie(
        sizes, labels=None, colors=colors,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )

    ax.legend(
        wedges, labels,
        loc="center left", bbox_to_anchor=(1, 0.5),
        fontsize=9, framealpha=0.9
    )

    ax.set_title(
        f"Distribusi Kesehatan Kebun\n"
        f"AOI Madiun — {date_tag}\n"
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
    """
    Kirim pesan + pie chart ke grup Telegram manajemen.
    """
    bot_token = config["telegram_bot_token"]
    chat_id   = config["telegram_chat_id"]

    if "ISI_TOKEN" in bot_token or "ISI_CHAT" in chat_id:
        log.warning("Telegram credentials belum diset. Skip notifikasi.")
        return

    cls_data = stats["classes"]

    # Format pesan
    sehat    = cls_data.get("Sehat",        {})
    ringan   = cls_data.get("Stres Ringan", {})
    berat    = cls_data.get("Stres Berat",  {})
    kritis   = cls_data.get("Kritis",       {})

    message = (
        f"🌿 *Update Peta Kesehatan Kebun*\n"
        f"📅 Periode: {date_tag}\n"
        f"📍 Area: Madiun, Jawa Timur\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Sehat        : {sehat.get('area_ha', 0):.0f} ha "
        f"({sehat.get('percentage', 0):.1f}%)\n"
        f"🟡 Stres Ringan : {ringan.get('area_ha', 0):.0f} ha "
        f"({ringan.get('percentage', 0):.1f}%)\n"
        f"🟠 Stres Berat  : {berat.get('area_ha', 0):.0f} ha "
        f"({berat.get('percentage', 0):.1f}%)\n"
        f"🔴 Kritis       : {kritis.get('area_ha', 0):.0f} ha "
        f"({kritis.get('percentage', 0):.1f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total area   : {stats['total_valid_ha']:.0f} ha\n"
        f"🗺️ Lihat peta   : {config['viewer_url']}\n"
        f"_Diproses otomatis oleh SENTINEL-KEBUN Agent_"
    )

    # Kirim teks
    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url_msg, json={
        "chat_id"   : chat_id,
        "text"      : message,
        "parse_mode": "Markdown",
    }, timeout=15)

    if resp.status_code == 200:
        log.info("Telegram text message sent OK")
    else:
        log.error(f"Telegram error: {resp.status_code} — {resp.text}")

    # Kirim pie chart sebagai foto
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
    log.info("SENTINEL-KEBUN Agent — Pipeline Mingguan")
    log.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(CONFIG["state_file"])

    # ---- Step 1: Cari file baru ----
    tiff_path = find_new_tiff(CONFIG["drive_watch_dir"], state)
    if tiff_path is None:
        log.info("Tidak ada file baru. Pipeline selesai.")
        return

    date_tag = datetime.now().strftime("%Y%m%d")

    # ---- Step 2: Inferensi RF ----
    log.info("\n[Step 2] Inferensi Random Forest...")
    pred_path, raster_meta = predict_health_class(
        tiff_path, CONFIG["model_path"], output_dir
    )

    # ---- Step 3: Colorize → COG ----
    log.info("\n[Step 3] Colorize → Cloud Optimized GeoTIFF...")
    rgb_path, cog_path = colorize_prediction(pred_path, output_dir, date_tag)

    # ---- Step 4: Statistik ----
    log.info("\n[Step 4] Hitung statistik luas per kelas...")
    stats = generate_stats(pred_path, date_tag, output_dir)

    # ---- Step 5: Notifikasi Telegram ----
    log.info("\n[Step 5] Kirim notifikasi Telegram...")
    send_telegram_notification(stats, cog_path, date_tag, CONFIG)

    # ---- Update state ----
    state.setdefault("processed_files", []).append(str(tiff_path))
    state["last_run"]      = datetime.now().isoformat()
    state["last_cog_path"] = str(cog_path)
    save_state(state, CONFIG["state_file"])

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info(f"Pipeline selesai dalam {elapsed:.1f} detik")
    log.info(f"Output tersimpan di: {output_dir}/")
    log.info(f"  health_map_COG_{date_tag}.tif  → upload ke hosting/Streamlit")
    log.info(f"  pie_chart_{date_tag}.png        → sudah dikirim ke Telegram")
    log.info(f"  stats_{date_tag}.json           → dibaca oleh Streamlit viewer")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
