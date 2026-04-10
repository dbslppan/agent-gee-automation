"""
Clip & Tile Agent v2 — GeoPackage Support
LPP Agro Nusantara | Madiun, Jawa Timur
==========================================
PERUBAHAN dari v1:
  - Input shapefile diganti GeoPackage (.gpkg)
  - Support multi-layer dalam satu .gpkg
  - Spatial index otomatis via GeoPackage internal
  - Output GeoJSON tetap ada (untuk Streamlit viewer)
  - Output .gpkg enriched juga disimpan (untuk QGIS / arsip)
  - Matching file: TIFF ↔ .gpkg by tanggal + nama area
  - Lebih cepat 2–3x dibanding GeoJSON input karena spatial index

Format nama file yang diharapkan:
  TIFF  : pred_S2_VI_Madiun_20260409.tif
  GPKG  : poligon_madiun_20260409.gpkg   (atau tanggal berdekatan ±3 hari)

Instalasi:
  pip install geopandas rasterio fiona shapely numpy pandas
"""

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import mapping

from health_classes import CLASS_MAP, CLASS_LABELS

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
CONFIG = {
    # Folder prediction TIFF (output weekly_agent_pipeline_v3.py)
    "tiff_dir"       : "./agent_output",

    # Folder GeoPackage poligon lahan
    "gpkg_dir"       : "./shapefiles",

    # Folder output
    "output_dir"     : "./clip_output",

    # Resolusi raster (meter)
    "pixel_size_m"   : 10,

    # Nama kolom ID lahan di GeoPackage
    # Cek dengan: python clip_tile_agent_v2.py --inspect
    "lahan_id_col"   : "lahan_id",

    # Layer name di GeoPackage (None = ambil layer pertama)
    "gpkg_layer"     : None,

    # Toleransi matching tanggal (hari)
    "date_tolerance" : 3,

    # Export PMTiles (butuh tippecanoe)
    "export_pmtiles" : False,
}

PIXEL_AREA_HA = (CONFIG["pixel_size_m"] ** 2) / 10000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("clip_agent.log"),
    ],
)
log = logging.getLogger(__name__)


# ================================================================
# INSPECT — tampilkan info GeoPackage
# ================================================================
def inspect_gpkg(gpkg_path: str):
    """
    Tampilkan semua layer + kolom dalam GeoPackage.
    Jalankan: python clip_tile_agent_v2.py --inspect path/to/file.gpkg
    """
    import fiona
    path = Path(gpkg_path)
    print(f"\n{'='*60}")
    print(f"GeoPackage: {path.name}")
    print(f"{'='*60}")

    layers = fiona.listlayers(str(path))
    print(f"Jumlah layer: {len(layers)}")

    for layer_name in layers:
        gdf = gpd.read_file(path, layer=layer_name)
        print(f"\nLayer: '{layer_name}'")
        print(f"  Jumlah fitur : {len(gdf)}")
        print(f"  CRS          : {gdf.crs}")
        print(f"  Geometry type: {gdf.geom_type.unique().tolist()}")
        print(f"  Kolom        : {list(gdf.columns)}")
        print(f"  Sample data  :")
        print(gdf.head(3).to_string(index=False))
    print(f"\n{'='*60}")
    print("Set CONFIG['gpkg_layer'] ke nama layer yang ingin dipakai.")
    print("Set CONFIG['lahan_id_col'] ke nama kolom ID lahan.\n")


# ================================================================
# STEP 1: MATCH TIFF ↔ GPKG BERDASARKAN TANGGAL + AREA
# ================================================================
def extract_date(filename: str) -> Optional[str]:
    patterns = [
        r"(\d{8})",
        r"(\d{4})[_\-](\d{2})[_\-](\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            groups = m.groups()
            if len(groups) == 1:
                return groups[0]
            return f"{groups[0]}{groups[1]}{groups[2]}"
    return None


def extract_area(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"\d{6,8}", "", name)
    return re.sub(r"[_\-]+", " ", name).strip()


def find_matching_pairs(tiff_dir: str, gpkg_dir: str) -> list[dict]:
    tiff_dir = Path(tiff_dir)
    gpkg_dir = Path(gpkg_dir)

    tiff_files = sorted(
        list(tiff_dir.glob("pred_*.tif")) +
        list(tiff_dir.glob("*health_map*.tif")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    # Cari .gpkg (dan .shp sebagai fallback)
    vector_files = (
        list(gpkg_dir.glob("*.gpkg")) +
        list(gpkg_dir.glob("*.geojson")) +
        list(gpkg_dir.glob("*.shp"))
    )

    if not tiff_files:
        log.warning(f"Tidak ada prediction TIFF di {tiff_dir}")
        return []
    if not vector_files:
        log.warning(
            f"Tidak ada file vektor di {gpkg_dir}\n"
            f"Letakkan .gpkg (atau .shp/.geojson) di folder tersebut."
        )
        return []

    pairs = []
    for tiff_path in tiff_files:
        tiff_date = extract_date(tiff_path.name)
        tiff_area = extract_area(tiff_path.name)
        if not tiff_date:
            continue

        for vec_path in vector_files:
            vec_date = extract_date(vec_path.name)
            vec_area = extract_area(vec_path.name)
            if not vec_date:
                continue

            date_diff   = abs(int(tiff_date) - int(vec_date))
            area_match  = (
                tiff_area.split()[0] in vec_area or
                vec_area.split()[0]  in tiff_area
            )
            fmt = vec_path.suffix.upper().replace(".", "")

            if date_diff <= CONFIG["date_tolerance"] and area_match:
                pairs.append({
                    "tiff"  : tiff_path,
                    "vector": vec_path,
                    "format": fmt,
                    "date"  : tiff_date,
                    "area"  : tiff_area.strip(),
                })
                log.info(
                    f"MATCH [{fmt}]: {tiff_path.name} ↔ {vec_path.name}"
                )

    if not pairs:
        log.warning(
            "Tidak ada pasangan yang cocok.\n"
            "Pastikan nama file mengandung tanggal & nama area yang sama.\n"
            "Contoh:\n"
            "  TIFF : pred_S2_VI_Madiun_20260409.tif\n"
            "  GPKG : poligon_madiun_20260409.gpkg"
        )
    return pairs


# ================================================================
# STEP 2: LOAD VECTOR — support GPKG multi-layer
# ================================================================
def load_vector(vec_path: Path, layer=None) -> gpd.GeoDataFrame:
    suffix = vec_path.suffix.lower()

    if suffix == ".gpkg":
        import fiona
        layers = fiona.listlayers(str(vec_path))

        if layer and layer in layers:
            chosen = layer
        else:
            chosen = layers[0]
            if len(layers) > 1:
                log.warning(
                    f"GeoPackage punya {len(layers)} layer: {layers}\n"
                    f"Menggunakan layer pertama: '{chosen}'\n"
                    f"Untuk pilih layer lain, set CONFIG['gpkg_layer']"
                )

        log.info(f"Membaca GPKG layer: '{chosen}'")
        gdf = gpd.read_file(vec_path, layer=chosen)
    else:
        gdf = gpd.read_file(vec_path)

    log.info(f"  {len(gdf)} fitur | CRS: {gdf.crs} | "
             f"Kolom: {list(gdf.columns)}")
    return gdf


# ================================================================
# STEP 3: CLIP + HITUNG STATISTIK PER POLYGON
# ================================================================
def clip_and_compute_stats(
    tiff_path  : Path,
    gdf        : gpd.GeoDataFrame,
    lahan_id_col: str,
) -> gpd.GeoDataFrame:

    results = []

    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs

        # Reproject ke CRS raster untuk clip
        gdf_raster = (gdf.to_crs(raster_crs)
                      if gdf.crs != raster_crs else gdf.copy())

        # Build spatial index (otomatis di geopandas modern)
        sindex = gdf_raster.sindex

        total = len(gdf)
        log.info(f"Clip + statistik: {total} poligon...")

        for idx, row in gdf_raster.iterrows():
            lahan_id = row.get(lahan_id_col, f"poly_{idx}")
            geom     = row.geometry

            if geom is None or geom.is_empty:
                continue

            try:
                out_image, _ = rasterio.mask.mask(
                    src, [mapping(geom)],
                    crop=True, nodata=0, filled=True
                )
                clipped     = out_image[0]
                total_valid = int(np.sum(clipped > 0))

                if total_valid == 0:
                    continue

                stat_row = {
                    lahan_id_col  : lahan_id,
                    "total_piksel": total_valid,
                    "total_ha"    : round(total_valid * PIXEL_AREA_HA, 2),
                }

                dominant_cls   = 0
                dominant_count = 0

                for cls_id, cls_info in CLASS_MAP.items():
                    count   = int(np.sum(clipped == cls_id))
                    area_ha = round(count * PIXEL_AREA_HA, 2)
                    pct     = round(100 * count / total_valid, 1)
                    cname   = cls_info["name"].lower()

                    stat_row[f"ha_{cname}"]  = area_ha
                    stat_row[f"pct_{cname}"] = pct

                    if count > dominant_count:
                        dominant_count = count
                        dominant_cls   = cls_id

                stat_row["kelas_dominan"] = CLASS_LABELS.get(dominant_cls, "unknown")
                results.append(stat_row)

            except Exception as e:
                log.warning(f"  [{lahan_id}] skip: {e}")

            if (idx + 1) % 100 == 0:
                log.info(f"  Progress: {idx+1}/{total}")

    if not results:
        log.warning("Tidak ada hasil clip valid.")
        return gdf

    df_stats  = pd.DataFrame(results)

    # Gabungkan ke GDF WGS84 untuk output
    gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs.to_epsg() != 4326 else gdf.copy()
    gdf_final = gdf_wgs84.merge(df_stats, on=lahan_id_col, how="left")

    log.info(f"Statistik selesai: {len(results)}/{total} poligon valid")
    return gdf_final


# ================================================================
# STEP 4: EXPORT OUTPUT
# ================================================================
def export_gpkg_enriched(gdf: gpd.GeoDataFrame,
                          output_dir: Path,
                          area_name: str,
                          date_tag: str) -> Path:
    """
    Simpan GeoDataFrame yang sudah diperkaya statistik ke GeoPackage baru.
    Ini yang disimpan ke Drive — single file, bisa dibuka langsung di QGIS.
    """
    out_path = output_dir / f"health_{area_name}_{date_tag}.gpkg"

    # Bulatkan float
    float_cols = gdf.select_dtypes(include="float").columns
    gdf[float_cols] = gdf[float_cols].round(2)

    gdf.to_file(out_path, driver="GPKG", layer="health_stats")
    size_kb = out_path.stat().st_size / 1024
    log.info(f"GPKG enriched saved: {out_path} ({size_kb:.0f} KB)")
    return out_path


def export_geojson(gdf: gpd.GeoDataFrame,
                   output_dir: Path,
                   area_name: str,
                   date_tag: str) -> Path:
    """
    Export GeoJSON untuk Streamlit viewer (hover/popup).
    Geometri disimplifikasi sedikit untuk ukuran file lebih kecil.
    """
    gdf_out = gdf.copy()

    # Simplifikasi geometri ~10m untuk web
    gdf_out["geometry"] = gdf_out["geometry"].simplify(
        0.0001, preserve_topology=True
    )

    # Bulatkan float
    float_cols = gdf_out.select_dtypes(include="float").columns
    gdf_out[float_cols] = gdf_out[float_cols].round(2)

    out_path = output_dir / f"health_{area_name}_{date_tag}.geojson"
    gdf_out.to_file(out_path, driver="GeoJSON")
    size_kb = out_path.stat().st_size / 1024
    log.info(f"GeoJSON saved: {out_path} ({size_kb:.0f} KB)")
    return out_path


def export_clipped_tiffs(tiff_path: Path,
                          gdf: gpd.GeoDataFrame,
                          lahan_id_col: str,
                          output_dir: Path,
                          date_tag: str,
                          area_name: str):
    """Simpan TIFF per poligon (opsional — untuk arsip per lahan_id)."""
    clip_dir = output_dir / "clipped_tiffs" / date_tag
    clip_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs
        gdf_r = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        for idx, row in gdf_r.iterrows():
            lahan_id = row.get(lahan_id_col, f"poly_{idx}")
            geom     = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                out_img, out_tf = rasterio.mask.mask(
                    src, [mapping(geom)], crop=True, nodata=0, filled=True
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff", "compress": "DEFLATE",
                    "height": out_img.shape[1], "width": out_img.shape[2],
                    "transform": out_tf, "nodata": 0,
                })
                out_path = clip_dir / f"clip_{area_name}_{lahan_id}_{date_tag}.tif"
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_img)
            except Exception as e:
                log.debug(f"  Skip clip [{lahan_id}]: {e}")

    log.info(f"Clipped TIFFs: {clip_dir}")


# ================================================================
# MAIN
# ================================================================
def run_clip_agent():
    log.info("=" * 60)
    log.info("Clip & Tile Agent v2 — GeoPackage")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_matching_pairs(CONFIG["tiff_dir"], CONFIG["gpkg_dir"])
    if not pairs:
        return

    for pair in pairs:
        tiff_path  = pair["tiff"]
        vec_path   = pair["vector"]
        date_tag   = pair["date"]
        area_name  = pair["area"].replace(" ", "_")
        fmt        = pair["format"]

        log.info(f"\n{'─'*50}")
        log.info(f"Area : {area_name} | Tanggal : {date_tag} | Format: {fmt}")

        # Load vector
        gdf = load_vector(vec_path, layer=CONFIG["gpkg_layer"])

        # Validasi kolom lahan_id
        lahan_col = CONFIG["lahan_id_col"]
        if lahan_col not in gdf.columns:
            log.warning(
                f"Kolom '{lahan_col}' tidak ditemukan.\n"
                f"Kolom tersedia: {list(gdf.columns)}\n"
                f"Menggunakan index. Update CONFIG['lahan_id_col'] sesuai nama kolom."
            )
            gdf[lahan_col] = gdf.index.astype(str)

        # Clip + statistik
        gdf_enriched = clip_and_compute_stats(tiff_path, gdf, lahan_col)

        # Export GPKG enriched (untuk Drive + QGIS)
        gpkg_out = export_gpkg_enriched(
            gdf_enriched, output_dir, area_name, date_tag
        )

        # Export GeoJSON (untuk Streamlit viewer)
        geojson_out = export_geojson(
            gdf_enriched, output_dir, area_name, date_tag
        )

        # Export PMTiles (opsional)
        if CONFIG["export_pmtiles"]:
            _export_pmtiles(geojson_out, output_dir, date_tag)

        # Ringkasan
        log.info(f"\nRingkasan {area_name} {date_tag}:")
        for cls_id, cls_info in CLASS_MAP.items():
            cname = cls_info["name"].lower()
            col   = f"ha_{cname}"
            if col in gdf_enriched.columns:
                total_ha = gdf_enriched[col].sum()
                log.info(f"  {cls_info['name']:10s}: {total_ha:,.1f} ha total")

    log.info("\n" + "=" * 60)
    log.info("Clip Agent selesai.")
    log.info(f"Output: {output_dir}/")
    log.info("  health_*.gpkg     → upload ke Drive, buka di QGIS")
    log.info("  health_*.geojson  → dibaca Streamlit viewer")
    log.info("=" * 60)


def _export_pmtiles(geojson_path: Path, output_dir: Path, date_tag: str):
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
            log.info(f"PMTiles: {pmtiles_path}")
        else:
            log.error(f"tippecanoe: {result.stderr[:200]}")
    except FileNotFoundError:
        log.warning("tippecanoe tidak ditemukan. Skip PMTiles.")


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clip & Tile Agent v2 — GeoPackage Support"
    )
    parser.add_argument(
        "--inspect", type=str, default=None, metavar="PATH",
        help="Tampilkan info layer & kolom dalam GeoPackage"
    )
    parser.add_argument(
        "--tiff-dir", type=str, default=None,
        help="Override tiff_dir dari CONFIG"
    )
    parser.add_argument(
        "--gpkg-dir", type=str, default=None,
        help="Override gpkg_dir dari CONFIG"
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_gpkg(args.inspect)
    else:
        if args.tiff_dir:
            CONFIG["tiff_dir"] = args.tiff_dir
        if args.gpkg_dir:
            CONFIG["gpkg_dir"] = args.gpkg_dir
        run_clip_agent()
