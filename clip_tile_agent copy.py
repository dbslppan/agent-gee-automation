"""
Clip & Tile Agent — LPP Agro Nusantara
=======================================
Fungsi:
  1. Match file TIFF (dari GEE export) dengan shapefile poligon
     berdasarkan kesamaan tanggal dalam nama file
  2. Clip raster prediction per poligon (lahan_id)
  3. Hitung statistik kelas per poligon (ha + %)
  4. Gabungkan dengan atribut tabel poligon
  5. Export ke GeoJSON (untuk Leaflet hover/click popup)
  6. Export ke PMTiles / MBTiles (untuk tile server interaktif)
  7. Simpan semua output ke Drive folder

Instalasi tambahan:
  pip install geopandas rasterio fiona shapely numpy pandas
  pip install mapbox-vector-tile tippecanoe  # untuk PMTiles (opsional)

Struktur nama file yang diharapkan:
  TIFF      : S2_VI_Madiun_VI_20260408.tif  (atau pred_*20260408*.tif)
  Shapefile : poligon_madiun_20260408.shp   (atau *madiun*20260408*.shp)
  Keduanya dicocokkan by: (1) nama area, (2) tanggal YYYYMMDD
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio as _rio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from shapely.geometry import mapping, shape

from health_classes import CLASS_MAP, CLASS_LABELS

import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    # Folder TIFF prediction (output weekly_agent_pipeline.py)
    "tiff_dir"      : "./agent_output",

    # Folder shapefile poligon lahan
    # Shapefile harus punya kolom: lahan_id (string/int), + atribut lain
    "shapefile_dir" : "./shapefiles",

    # Folder output clip + GeoJSON
    "output_dir"    : "./clip_output",

    # Google Drive folder untuk upload hasil
    "drive_folder"  : "GEE_LPP_MADIUN/clip_output",

    # Resolusi raster (meter) — harus sama dengan GEE export
    "pixel_size_m"  : 10,

    # Kolom ID lahan di shapefile
    "lahan_id_col"  : "LahanID",

    # Toleransi matching tanggal (hari) — file dalam rentang ini dianggap match
    "date_tolerance_days": 3,

    # Export PMTiles (butuh tippecanoe terinstall)
    "export_pmtiles": False,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("clip_agent.log"),
    ],
)
log = logging.getLogger(__name__)

# PIXEL_AREA_HA = (CONFIG["pixel_size_m"] ** 2) / 10000
def _compute_pixel_area_ha() -> float:
    """Hitung luas piksel dalam ha dari pred TIFF pertama yang ditemukan."""
    tiff_dir = Path(CONFIG["tiff_dir"])
    tiffs    = sorted(tiff_dir.glob("pred_*.tif"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not tiffs:
        return (CONFIG["pixel_size_m"] ** 2) / 10000  # fallback
    with _rio.open(tiffs[0]) as src:
        res    = src.res
        crs    = src.crs
        bounds = src.bounds
    if crs.is_geographic:
        lat_c  = (bounds.top + bounds.bottom) / 2
        ph     = res[0] * 111320.0
        pw     = res[1] * 111320.0 * np.cos(np.radians(lat_c))
        return (ph * pw) / 10000
    return (res[0] * res[1]) / 10000

PIXEL_AREA_HA = _compute_pixel_area_ha()

# ================================================================
# STEP 1: MATCH TIFF ↔ SHAPEFILE BERDASARKAN TANGGAL
# ================================================================
def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Ekstrak tanggal YYYYMMDD dari nama file.
    Mendukung format: *_20260408*, *_2026-04-08*, *_08042026*
    """
    patterns = [
        r"(\d{8})",           # YYYYMMDD
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{2})(\d{2})(\d{4})",    # DDMMYYYY
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            groups = m.groups()
            if len(groups) == 1:
                return groups[0]  # YYYYMMDD langsung
            elif len(groups) == 3:
                # Coba deteksi urutan: jika grup[0] > 31 → YYYY
                if int(groups[0]) > 31:
                    return f"{groups[0]}{groups[1]}{groups[2]}"
                else:
                    return f"{groups[2]}{groups[1]}{groups[0]}"
    return None


def extract_area_from_filename(filename: str) -> str:
    """Ekstrak nama area (misal 'madiun') dari nama file — lowercase."""
    name = Path(filename).stem.lower()
    # Hapus tanggal dan underscore/angka berlebih
    name = re.sub(r"\d{6,8}", "", name)
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name


def find_matching_pairs(tiff_dir: str, shp_dir: str) -> list[dict]:
    """
    Scan kedua folder dan cari pasangan TIFF + shapefile
    berdasarkan nama area dan tanggal yang sama.

    Returns list of:
      {"tiff": Path, "shp": Path, "date": str, "area": str}
    """
    tiff_dir = Path(tiff_dir)
    shp_dir  = Path(shp_dir)

    # Cari semua prediction TIFF
    tiff_files = list(tiff_dir.glob("pred_*.tif")) + \
                 list(tiff_dir.glob("*health_map*.tif"))

    # Cari semua shapefile
    shp_files  = list(shp_dir.glob("*.shp")) + \
                 list(shp_dir.glob("*.geojson"))

    if not tiff_files:
        log.warning(f"Tidak ada prediction TIFF di {tiff_dir}")
        return []
    if not shp_files:
        log.warning(f"Tidak ada shapefile di {shp_dir}")
        return []

    pairs = []
    for tiff_path in tiff_files:
        tiff_date = extract_date_from_filename(tiff_path.name)
        tiff_area = extract_area_from_filename(tiff_path.name)

        if not tiff_date:
            log.warning(f"Tidak bisa ekstrak tanggal dari: {tiff_path.name}")
            continue

        for shp_path in shp_files:
            shp_date = extract_date_from_filename(shp_path.name)
            shp_area = extract_area_from_filename(shp_path.name)

            if not shp_date:
                continue

            # Cek kecocokan tanggal (dalam toleransi N hari)
            date_diff = abs(int(tiff_date) - int(shp_date))
            area_match = (
                tiff_area in shp_area or
                shp_area in tiff_area or
                tiff_area.split()[0] == shp_area.split()[0]  # kata pertama sama
            )

            if date_diff <= CONFIG["date_tolerance_days"] and area_match:
                pairs.append({
                    "tiff" : tiff_path,
                    "shp"  : shp_path,
                    "date" : tiff_date,
                    "area" : tiff_area.strip(),
                })
                log.info(
                    f"MATCH: {tiff_path.name} ↔ {shp_path.name} "
                    f"[date: {tiff_date}, area: {tiff_area}]"
                )

    if not pairs:
        log.warning(
            "Tidak ada pasangan yang cocok ditemukan.\n"
            "Pastikan nama file mengandung tanggal dan nama area yang sama.\n"
            "Contoh:\n"
            "  TIFF  : pred_S2_VI_Madiun_20260408.tif\n"
            "  Shapefile: poligon_madiun_20260408.shp"
        )

    return pairs


# ================================================================
# STEP 2: CLIP RASTER PER POLYGON + HITUNG STATISTIK
# ================================================================
def clip_and_compute_stats(
    tiff_path: Path,
    gdf: gpd.GeoDataFrame,
    lahan_id_col: str,
    pixel_area_ha: float,
) -> gpd.GeoDataFrame:
    """
    Untuk setiap baris (polygon) di GeoDataFrame:
      1. Clip raster ke polygon tersebut
      2. Hitung jumlah piksel per kelas
      3. Konversi ke ha dan persentase
      4. Tambahkan kolom statistik ke GeoDataFrame

    Returns GeoDataFrame yang diperkaya dengan kolom statistik.
    """
    results = []

    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs

        # Reproject shapefile ke CRS raster jika berbeda
        if gdf.crs != raster_crs:
            log.info(f"Reproject shapefile {gdf.crs} → {raster_crs}")
            gdf_raster = gdf.to_crs(raster_crs)
        else:
            gdf_raster = gdf.copy()

        total_polygons = len(gdf)
        log.info(f"Memproses {total_polygons} poligon lahan...")

        for idx, row in gdf_raster.iterrows():
            lahan_id = row.get(lahan_id_col, f"poly_{idx}")
            geom     = row.geometry

            if geom is None or geom.is_empty:
                log.warning(f"  [{lahan_id}] Geometri kosong, skip.")
                continue

            try:
                # Clip raster ke polygon
                out_image, out_transform = rasterio.mask.mask(
                    src,
                    [mapping(geom)],
                    crop=True,
                    nodata=0,
                    filled=True
                )
                clipped = out_image[0]  # band 1 = prediction class

                # Hitung piksel per kelas
                total_valid = int(np.sum(clipped > 0))
                if total_valid == 0:
                    log.debug(f"  [{lahan_id}] Tidak ada piksel valid.")
                    continue

                stat_row = {
                    lahan_id_col     : lahan_id,
                    "total_piksel"   : total_valid,
                    "total_ha"       : round(total_valid * pixel_area_ha, 2),
                }

                # Per kelas
                dominant_cls    = 0
                dominant_count  = 0

                for cls_id, cls_info in CLASS_MAP.items():
                    count   = int(np.sum(clipped == cls_id))
                    area_ha = round(count * pixel_area_ha, 2)
                    pct     = round(100 * count / total_valid, 1) if total_valid else 0.0
                    cname   = cls_info["name"].lower().replace(" ", "_")

                    stat_row[f"ha_{cname}"]  = area_ha
                    stat_row[f"pct_{cname}"] = pct

                    if count > dominant_count:
                        dominant_count = count
                        dominant_cls   = cls_id

                stat_row["kelas_dominan"] = CLASS_LABELS.get(dominant_cls, "unknown")
                results.append(stat_row)

                if (idx + 1) % 50 == 0:
                    log.info(f"  Progress: {idx + 1}/{total_polygons}")

            except Exception as e:
                log.error(f"  [{lahan_id}] Error saat clip: {e}")
                continue

    if not results:
        log.warning("Tidak ada hasil clip yang valid.")
        return gdf

    # Gabungkan statistik ke GeoDataFrame original (WGS84 untuk GeoJSON)
    df_stats  = pd.DataFrame(results)
    gdf_orig  = gdf.to_crs("EPSG:4326") if gdf.crs.to_epsg() != 4326 else gdf.copy()
    gdf_final = gdf_orig.merge(df_stats, on=lahan_id_col, how="left")

    log.info(f"Statistik berhasil dihitung untuk {len(results)}/{total_polygons} poligon")
    return gdf_final


# ================================================================
# STEP 3: SIMPAN PER-POLYGON TIFF (opsional — untuk arsip)
# ================================================================
def export_clipped_tiffs(
    tiff_path: Path,
    gdf: gpd.GeoDataFrame,
    lahan_id_col: str,
    output_dir: Path,
    date_tag: str,
    area_name: str,
):
    """
    Simpan TIFF terpisah per polygon ke subfolder.
    Nama file: clip_{area}_{lahan_id}_{date}.tif
    """
    clip_dir = output_dir / "clipped_tiffs" / date_tag
    clip_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs
        gdf_raster = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        for idx, row in gdf_raster.iterrows():
            lahan_id = row.get(lahan_id_col, f"poly_{idx}")
            geom     = row.geometry

            if geom is None or geom.is_empty:
                continue

            try:
                out_image, out_transform = rasterio.mask.mask(
                    src, [mapping(geom)],
                    crop=True, nodata=0, filled=True
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver"   : "GTiff",
                    "height"   : out_image.shape[1],
                    "width"    : out_image.shape[2],
                    "transform": out_transform,
                    "nodata"   : 0,
                    "compress" : "DEFLATE",
                })

                out_path = clip_dir / f"clip_{area_name}_{lahan_id}_{date_tag}.tif"
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)

            except Exception as e:
                log.warning(f"  Skip clip TIFF [{lahan_id}]: {e}")

    log.info(f"Clipped TIFFs saved to: {clip_dir}")


# ================================================================
# STEP 4: EXPORT GEOJSON (untuk Leaflet hover/click)
# ================================================================
def export_geojson(gdf: gpd.GeoDataFrame, output_path: Path):
    """
    Simpan GeoDataFrame sebagai GeoJSON WGS84.
    Semua kolom numerik dibulatkan untuk efisiensi ukuran file.
    """
    gdf_out = gdf.copy()

    # Bulatkan semua float
    float_cols = gdf_out.select_dtypes(include="float").columns
    gdf_out[float_cols] = gdf_out[float_cols].round(2)

    # Simplifikasi geometri untuk web (toleransi ~10m)
    if gdf_out.crs and gdf_out.crs.to_epsg() == 4326:
        gdf_out["geometry"] = gdf_out["geometry"].simplify(
            0.0001, preserve_topology=True
        )

    gdf_out.to_file(output_path, driver="GeoJSON")
    size_kb = output_path.stat().st_size / 1024
    log.info(f"GeoJSON saved: {output_path} ({size_kb:.0f} KB)")
    return output_path


# ================================================================
# STEP 5: EXPORT PMTILES (opsional — butuh tippecanoe CLI)
# ================================================================
def export_pmtiles(geojson_path: Path, output_dir: Path, date_tag: str) -> Optional[Path]:
    """
    Konversi GeoJSON → PMTiles menggunakan tippecanoe.
    Hasilnya bisa di-serve langsung oleh MapLibre GL JS.

    Install tippecanoe:
      Linux  : sudo apt install tippecanoe
      macOS  : brew install tippecanoe
    """
    pmtiles_path = output_dir / f"health_{date_tag}.pmtiles"

    cmd = [
        "tippecanoe",
        "--output", str(pmtiles_path),
        "--layer", "health",
        "--minimum-zoom", "8",
        "--maximum-zoom", "16",
        "--drop-densest-as-needed",
        "--force",
        str(geojson_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            size_mb = pmtiles_path.stat().st_size / (1024 * 1024)
            log.info(f"PMTiles saved: {pmtiles_path} ({size_mb:.1f} MB)")
            return pmtiles_path
        else:
            log.error(f"tippecanoe error: {result.stderr}")
            return None
    except FileNotFoundError:
        log.warning("tippecanoe tidak ditemukan. Skip PMTiles export.")
        log.warning("Install: sudo apt install tippecanoe")
        return None
    except subprocess.TimeoutExpired:
        log.error("tippecanoe timeout (>120s).")
        return None


# ================================================================
# MAIN PIPELINE
# ================================================================
# def run_clip_agent():
#     log.info("=" * 60)
#     log.info("Clip & Tile Agent — LPP Agro Nusantara")
#     log.info("=" * 60)

#     output_dir = Path(CONFIG["output_dir"])
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # ---- Step 1: Temukan pasangan TIFF ↔ Shapefile ----
#     pairs = find_matching_pairs(CONFIG["tiff_dir"], CONFIG["shapefile_dir"])
#     if not pairs:
#         log.error("Tidak ada pasangan file yang cocok. Pipeline dihentikan.")
#         return

#     for pair in pairs:
#         tiff_path = pair["tiff"]
#         shp_path  = pair["shp"]
#         date_tag  = pair["date"]
#         area_name = pair["area"].replace(" ", "_")

#         log.info(f"\nMemproses: {area_name} | {date_tag}")
#         log.info(f"  TIFF : {tiff_path.name}")
#         log.info(f"  SHP  : {shp_path.name}")

#         # Load shapefile
#         try:
#             gdf = gpd.read_file(shp_path)
#         except Exception as e:
#             log.error(f"Gagal baca shapefile: {e}")
#             continue

#         # Validasi kolom lahan_id
#         lahan_col = CONFIG["lahan_id_col"]
#         if lahan_col not in gdf.columns:
#             log.warning(
#                 f"Kolom '{lahan_col}' tidak ditemukan. "
#                 f"Kolom tersedia: {list(gdf.columns)}\n"
#                 f"Menggunakan index sebagai lahan_id."
#             )
#             gdf[lahan_col] = gdf.index.astype(str)

#         log.info(f"  Jumlah poligon: {len(gdf)}")
#         log.info(f"  Kolom atribut : {list(gdf.columns)}")

#         # ---- Step 2: Clip + hitung statistik ----
#         gdf_enriched = clip_and_compute_stats(
#             tiff_path, gdf,
#             lahan_col,
#             PIXEL_AREA_HA,
#         )

#         # ---- Step 3: Export clipped TIFFs per polygon ----
#         export_clipped_tiffs(
#             tiff_path, gdf, lahan_col,
#             output_dir, date_tag, area_name
#         )

#         # ---- Step 4: Export GeoJSON ----
#         geojson_path = output_dir / f"health_{area_name}_{date_tag}.geojson"
#         export_geojson(gdf_enriched, geojson_path)

#         # ---- Step 5: Export PMTiles (opsional) ----
#         if CONFIG["export_pmtiles"]:
#             export_pmtiles(geojson_path, output_dir, date_tag)

#         # ---- Ringkasan per area ----
#         log.info(f"\nRingkasan {area_name} {date_tag}:")
#         for cls_id, cls_info in CLASS_MAP.items():
#             cname = cls_info["name"].lower().replace(" ", "_")
#             if f"ha_{cname}" in gdf_enriched.columns:
#                 total_ha = gdf_enriched[f"ha_{cname}"].sum()
#                 log.info(f"  {cls_info['name']:10s}: {total_ha:,.1f} ha total")

#     log.info("\n" + "=" * 60)
#     log.info("Clip Agent selesai.")
#     log.info(f"Output: {output_dir}/")
#     log.info("  health_*.geojson  → upload ke Drive / Streamlit")
#     log.info("  clipped_tiffs/    → arsip per poligon")
#     log.info("=" * 60)

def run_clip_agent():
    log.info("=" * 60)
    log.info("Clip & Tile Agent — LPP Agro Nusantara")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardcode pasangan file karena nama tidak mengandung tanggal
    from datetime import datetime
    date_tag  = datetime.now().strftime("%Y%m%d")
    area_name = "madiun"

    # Cari pred tiff terbaru di agent_output
    tiff_dir   = Path(CONFIG["tiff_dir"])
    tiff_files = sorted(tiff_dir.glob("pred_*.tif"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not tiff_files:
        log.error("Tidak ada prediction TIFF di agent_output/. "
                  "Jalankan weekly_agent_pipeline.py dulu.")
        return
    tiff_path = tiff_files[0]
    log.info(f"TIFF  : {tiff_path.name}")

    # Hardcode shapefile
    shp_path = Path(CONFIG["shapefile_dir"]) / "sk12.shp"
    if not shp_path.exists():
        log.error(f"Shapefile tidak ditemukan: {shp_path}")
        return
    log.info(f"SHP   : {shp_path.name}")

    # Load shapefile
    import geopandas as gpd
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        log.error(f"Gagal baca shapefile: {e}")
        return

    # Validasi kolom lahan_id
    lahan_col = CONFIG["lahan_id_col"]
    if lahan_col not in gdf.columns:
        log.warning(f"Kolom '{lahan_col}' tidak ditemukan.")
        log.warning(f"Kolom tersedia: {list(gdf.columns)}")
        log.warning("Menggunakan index sebagai lahan_id.")
        gdf[lahan_col] = gdf.index.astype(str)

    log.info(f"Jumlah poligon : {len(gdf)}")
    log.info(f"Kolom tersedia : {list(gdf.columns)}")

    # Clip + statistik
    gdf_enriched = clip_and_compute_stats(
        tiff_path, gdf, lahan_col, PIXEL_AREA_HA
    )

    # Export clipped TIFFs per polygon
    export_clipped_tiffs(
        tiff_path, gdf, lahan_col, output_dir, date_tag, area_name
    )

    # Export GeoJSON
    geojson_path = output_dir / f"health_{area_name}_{date_tag}.geojson"
    export_geojson(gdf_enriched, geojson_path)

    # Export PMTiles (opsional)
    if CONFIG["export_pmtiles"]:
        export_pmtiles(geojson_path, output_dir, date_tag)

    # Ringkasan
    log.info(f"\nRingkasan {area_name} {date_tag}:")
    for cls_id, cls_info in CLASS_MAP.items():
        cname = cls_info["name"].lower()
        if f"ha_{cname}" in gdf_enriched.columns:
            total_ha = gdf_enriched[f"ha_{cname}"].sum()
            log.info(f"  {cls_info['name']:10s}: {total_ha:,.1f} ha total")

    log.info("\n" + "=" * 60)
    log.info("Clip Agent selesai.")
    log.info(f"Output: {output_dir}/")
    log.info("=" * 60)


if __name__ == "__main__":
    run_clip_agent()
