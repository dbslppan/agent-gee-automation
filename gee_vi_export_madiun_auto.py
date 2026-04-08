# gee_vi_export_madiun_auto.py

"""
GEE Python API — Export VI Mingguan Otomatis
Kebun Madiun, Jawa Timur | LPP Agro Nusantara

Cara pakai:
  pip install earthengine-api
  earthengine authenticate   (sekali saja)
  python gee_vi_export_madiun_auto.py

Untuk automasi: tambahkan ke cron atau Cloud Scheduler
  Cron contoh (setiap Senin jam 06:00 WIB = 23:00 UTC Minggu):
  0 23 * * 0 /usr/bin/python3 /path/to/gee_vi_export_madiun_auto.py
"""

import ee
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
GEE_PROJECT    = 'your-gee-project-id'   # ganti dengan project GEE kamu
DRIVE_FOLDER   = 'GEE_LPP_MADIUN'
FILE_PREFIX    = 'S2_VI_Madiun'
CLOUD_MAX      = 20        # % cloud cover maksimum per scene
LOOKBACK_DAYS  = 8         # ambil scene dalam N hari terakhir
SCALE          = 10        # resolusi output (meter)
CRS            = 'EPSG:32749'  # UTM Zone 49S — Jawa Timur

# State file — mencegah export scene yang sama dua kali
STATE_FILE = Path(__file__).parent / 'gee_export_state.json'

# AOI Kabupaten Madiun (lon_min, lat_min, lon_max, lat_max)
AOI_COORDS = [111.2800, -7.8500, 111.8200, -7.3500]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gee_export.log')
    ]
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------
# INISIALISASI GEE
# ----------------------------------------------------------------
def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        log.info("GEE initialized OK")
    except Exception as e:
        log.error(f"GEE init failed: {e}")
        raise


# ----------------------------------------------------------------
# STATE MANAGEMENT — simpan last export date
# ----------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'last_export_date': None, 'exported_scenes': []}


def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    log.info(f"State saved: {STATE_FILE}")


# ----------------------------------------------------------------
# MASKING AWAN — gunakan SCL band
# ----------------------------------------------------------------
def mask_s2_clouds(image):
    scl = image.select('SCL')
    cloud_mask = (scl.neq(3)   # cloud shadow
                    .And(scl.neq(8))   # cloud medium prob
                    .And(scl.neq(9))   # cloud high prob
                    .And(scl.neq(10))) # cirrus
    return (image
            .updateMask(cloud_mask)
            .divide(10000)
            .copyProperties(image, ['system:time_start', 'system:index']))


# ----------------------------------------------------------------
# HITUNG VEGETATION INDICES
# ----------------------------------------------------------------
def add_vi(image):
    # NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # GNDVI
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')

    # SAVI (L=0.5)
    L = 0.5
    nir = image.select('B8')
    red = image.select('B4')
    savi = (nir.subtract(red)
               .divide(nir.add(red).add(L))
               .multiply(1 + L)
               .rename('SAVI'))

    # RVI
    rvi = image.select('B8').divide(image.select('B4')).rename('RVI')

    # EVI
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
        {'NIR': image.select('B8'),
         'RED': image.select('B4'),
         'BLUE': image.select('B2')}
    ).rename('EVI')

    return image.addBands([ndvi, gndvi, savi, rvi, evi])


# ----------------------------------------------------------------
# CEK APAKAH ADA SCENE BARU
# ----------------------------------------------------------------
def check_new_scenes(aoi, start_date: str, end_date: str) -> int:
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX)))
    count = collection.size().getInfo()
    log.info(f"Scene tersedia ({start_date} → {end_date}): {count}")
    return count


# ----------------------------------------------------------------
# MAIN EXPORT FUNCTION
# ----------------------------------------------------------------
def run_export():
    init_gee()
    state = load_state()

    # Tentukan rentang tanggal
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str   = end_dt.strftime('%Y-%m-%d')

    # Cek apakah sudah pernah diexport periode ini
    export_key = f"{start_str}_{end_str}"
    if export_key in state.get('exported_scenes', []):
        log.info(f"Periode {export_key} sudah diexport sebelumnya. Skip.")
        return

    # Definisi AOI
    aoi = ee.Geometry.Rectangle(AOI_COORDS)

    # Cek ketersediaan scene
    n_scenes = check_new_scenes(aoi, start_str, end_str)
    if n_scenes == 0:
        log.warning("Tidak ada scene bersih tersedia. Pipeline dihentikan.")
        return

    # Build koleksi + hitung VI
    log.info("Memproses VI dari koleksi Sentinel-2...")
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start_str, end_str)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX))
                  .map(mask_s2_clouds)
                  .map(add_vi))

    # Komposit median
    vi_median = (collection
                 .select(['NDVI', 'GNDVI', 'SAVI', 'RVI', 'EVI'])
                 .median()
                 .clip(aoi))

    # Nama file dengan tanggal
    date_tag = end_dt.strftime('%Y%m%d')
    file_name = f"{FILE_PREFIX}_VI_{date_tag}"
    desc = f"VI_export_{date_tag}"

    log.info(f"Memulai export: {file_name} → Drive/{DRIVE_FOLDER}/")

    # Submit export task
    task = ee.batch.Export.image.toDrive(
        image          = vi_median,
        description    = desc,
        folder         = DRIVE_FOLDER,
        fileNamePrefix = file_name,
        region         = aoi,
        scale          = SCALE,
        crs            = CRS,
        maxPixels      = int(1e10),
        fileFormat     = 'GeoTIFF'
    )
    task.start()

    log.info(f"Task submitted: {task.id}")
    log.info(f"Pantau di: https://code.earthengine.google.com/tasks")

    # Simpan state
    state.setdefault('exported_scenes', []).append(export_key)
    state['last_export_date'] = end_str
    state['last_task_id']     = task.id
    save_state(state)

    return task.id


# ----------------------------------------------------------------
# MONITOR STATUS TASK (opsional — panggil terpisah)
# ----------------------------------------------------------------
def check_task_status(task_id: str):
    init_gee()
    tasks = ee.data.getTaskList()
    for t in tasks:
        if t['id'] == task_id:
            log.info(f"Task {task_id}: {t['state']} — {t.get('description','')}")
            return t['state']
    log.warning(f"Task {task_id} tidak ditemukan")
    return None


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == '__main__':
    task_id = run_export()
    if task_id:
        log.info(f"Pipeline selesai. Task ID: {task_id}")
        log.info("File akan muncul di Google Drive dalam 5–20 menit.")
