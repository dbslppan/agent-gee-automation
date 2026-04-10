"""
Clip & Tile Agent — LPP Agro Nusantara
=======================================
Memotong hasil prediksi per poligon lahan (dari GeoPackage/GeoJSON),
menghitung statistik kelas per lahan, dan export ke GeoJSON untuk viewer.

Jalankan setelah weekly_agent_pipeline.py selesai:
  python clip_tile_agent.py

Instalasi:
  pip install geopandas rasterio fiona shapely numpy pandas
"""

import json
import logging
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from datetime import datetime
from pathlib import Path
from shapely.geometry import mapping

import geopandas as gpd

from health_classes import CLASS_MAP, CLASS_LABELS

# ----------------------------------------------------------------
# KONFIGURASI — sesuaikan di sini
# ----------------------------------------------------------------
CONFIG = {
    # Folder prediction TIFF (output weekly_agent_pipeline.py)
    "tiff_dir"      : "./agent_output",

    # File vektor batas lahan — sesuaikan nama file
    "vector_file"   : "./shapefiles/mumbul.gpkg",

    # Layer name di GeoPackage (None = ambil layer pertama)
    "gpkg_layer"    : None,

    # Folder output GeoJSON + statistik
    "output_dir"    : "./clip_output",

    # Nama kolom ID lahan — sesuaikan dengan kolom di file kamu
    # Jalankan python clip_tile_agent.py --inspect untuk lihat kolom
    "lahan_id_col"  : "LahanID",

    # Nama area untuk penamaan file output
    "area_name"     : "mumbul",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("clip_agent.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ================================================================
# HITUNG PIXEL AREA — handle CRS geografis vs projected
# ================================================================
def compute_pixel_area_ha(tiff_path: Path) -> float:
    with rasterio.open(tiff_path) as src:
        res    = src.res
        crs    = src.crs
        bounds = src.bounds
    if crs.is_geographic:
        lat_c  = (bounds.top + bounds.bottom) / 2
        ph     = res[0] * 111320.0
        pw     = res[1] * 111320.0 * np.cos(np.radians(lat_c))
        return (ph * pw) / 10000
    return (res[0] * res[1]) / 10000


# ================================================================
# INSPECT — tampilkan info file vektor
# ================================================================
def inspect_vector(vec_path: str):
    path = Path(vec_path)
    print(f"\n{'='*60}")
    print(f"File: {path.name}")
    print(f"{'='*60}")

    if path.suffix.lower() == ".gpkg":
        import fiona
        layers = fiona.listlayers(str(path))
        print(f"Jumlah layer: {len(layers)}")
        for layer_name in layers:
            gdf = gpd.read_file(path, layer=layer_name)
            print(f"\nLayer: '{layer_name}'")
            print(f"  Jumlah fitur : {len(gdf)}")
            print(f"  CRS          : {gdf.crs}")
            print(f"  Kolom        : {list(gdf.columns)}")
            print(gdf.head(3).to_string(index=False))
    else:
        gdf = gpd.read_file(path)
        print(f"Jumlah fitur : {len(gdf)}")
        print(f"CRS          : {gdf.crs}")
        print(f"Kolom        : {list(gdf.columns)}")
        print(gdf.head(3).to_string(index=False))

    print(f"\n{'='*60}")
    print("Set CONFIG['lahan_id_col'] ke nama kolom ID lahan.")
    print("Set CONFIG['gpkg_layer'] ke nama layer jika GPKG multi-layer.\n")


# ================================================================
# LOAD VECTOR
# ================================================================
def load_vector(vec_path: Path, layer=None) -> gpd.GeoDataFrame:
    suffix = vec_path.suffix.lower()

    if suffix == ".gpkg":
        import fiona
        layers  = fiona.listlayers(str(vec_path))
        chosen  = layer if (layer and layer in layers) else layers[0]
        if len(layers) > 1 and not layer:
            log.warning(f"GPKG punya {len(layers)} layer: {layers}")
            log.warning(f"Menggunakan layer pertama: '{chosen}'")
            log.warning("Set CONFIG['gpkg_layer'] untuk pilih layer lain.")
        log.info(f"Membaca layer GPKG: '{chosen}'")
        gdf = gpd.read_file(vec_path, layer=chosen)
    else:
        gdf = gpd.read_file(vec_path)

    log.info(f"  {len(gdf)} fitur | CRS: {gdf.crs}")
    log.info(f"  Kolom: {list(gdf.columns)}")
    return gdf


# ================================================================
# CLIP + STATISTIK PER POLYGON
# ================================================================
def clip_and_compute_stats(
    tiff_path    : Path,
    gdf          : gpd.GeoDataFrame,
    lahan_id_col : str,
    pixel_area_ha: float,
) -> gpd.GeoDataFrame:

    results = []

    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs
        gdf_raster = (gdf.to_crs(raster_crs)
                      if gdf.crs != raster_crs else gdf.copy())

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
                    "total_ha"    : round(total_valid * pixel_area_ha, 2),
                }

                dominant_cls   = 0
                dominant_count = 0

                for cls_id, cls_info in CLASS_MAP.items():
                    count   = int(np.sum(clipped == cls_id))
                    area_ha = round(count * pixel_area_ha, 2)
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
    gdf_wgs84 = (gdf.to_crs("EPSG:4326")
                 if gdf.crs.to_epsg() != 4326 else gdf.copy())
    gdf_final = gdf_wgs84.merge(df_stats, on=lahan_id_col, how="left")

    log.info(f"Statistik selesai: {len(results)}/{total} poligon valid")
    return gdf_final


# ================================================================
# EXPORT GEOJSON
# ================================================================
def export_geojson(gdf: gpd.GeoDataFrame, output_path: Path):
    gdf_out = gdf.copy()

    # Simplifikasi geometri untuk web
    gdf_out["geometry"] = gdf_out["geometry"].simplify(
        0.0001, preserve_topology=True
    )

    # Bulatkan float
    float_cols = gdf_out.select_dtypes(include="float").columns
    gdf_out[float_cols] = gdf_out[float_cols].round(2)

    gdf_out.to_file(output_path, driver="GeoJSON")
    size_kb = output_path.stat().st_size / 1024
    log.info(f"GeoJSON saved: {output_path} ({size_kb:.0f} KB)")


# ================================================================
# MAIN
# ================================================================
def run_clip_agent():
    log.info("=" * 60)
    log.info("Clip & Tile Agent — LPP Agro Nusantara")
    log.info("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cari pred TIFF terbaru di agent_output
    tiff_dir   = Path(CONFIG["tiff_dir"])
    tiff_files = sorted(
        tiff_dir.glob("pred_*.tif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not tiff_files:
        log.error(
            "Tidak ada prediction TIFF di agent_output/.\n"
            "Jalankan weekly_agent_pipeline.py terlebih dahulu."
        )
        return

    tiff_path = tiff_files[0]
    log.info(f"TIFF  : {tiff_path.name}")

    # Load file vektor
    vec_path = Path(CONFIG["vector_file"])
    if not vec_path.exists():
        log.error(
            f"File vektor tidak ditemukan: {vec_path}\n"
            f"Pastikan file ada di folder shapefiles/"
        )
        return
    log.info(f"Vektor: {vec_path.name}")

    # Hitung pixel area dari TIFF aktual
    pixel_area_ha = compute_pixel_area_ha(tiff_path)
    log.info(f"Pixel area: {pixel_area_ha:.6f} ha/piksel")

    # Load GeoDataFrame
    gdf = load_vector(vec_path, layer=CONFIG["gpkg_layer"])

    # Validasi kolom lahan_id
    lahan_col = CONFIG["lahan_id_col"]
    if lahan_col not in gdf.columns:
        log.warning(f"Kolom '{lahan_col}' tidak ditemukan.")
        log.warning(f"Kolom tersedia: {list(gdf.columns)}")
        log.warning("Menggunakan index sebagai lahan_id.")
        log.warning("Update CONFIG['lahan_id_col'] sesuai nama kolom yang benar.")
        gdf[lahan_col] = gdf.index.astype(str)

    # Clip + statistik
    gdf_enriched = clip_and_compute_stats(
        tiff_path, gdf, lahan_col, pixel_area_ha
    )

    # Ambil tanggal dari nama TIFF atau gunakan hari ini
    date_tag  = datetime.now().strftime("%Y%m%d")
    area_name = CONFIG["area_name"]

    # Export GeoJSON
    geojson_path = output_dir / f"health_{area_name}_{date_tag}.geojson"
    export_geojson(gdf_enriched, geojson_path)

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
    log.info(f"  {geojson_path.name} -> buka di streamlit_viewer_v2.py")
    log.info("=" * 60)


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clip & Tile Agent — LPP Agro Nusantara"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Tampilkan info layer & kolom file vektor"
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_vector(CONFIG["vector_file"])
    else:
        run_clip_agent()