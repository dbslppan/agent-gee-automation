// gee_vi_export_madiun.js

// ============================================================
//  GEE Script: Export Vegetation Indices — Kebun Madiun
//  Target: Sentinel-2 SR | NDVI, GNDVI, SAVI, RVI
//  AOI  : Kabupaten Madiun, Jawa Timur
//  Author: LPP Agro Nusantara — Digital Intelligence
// ============================================================

// ------------------------------------------------------------
// 1. KONFIGURASI — ubah parameter di sini sesuai kebutuhan
// ------------------------------------------------------------
var CONFIG = {
  // Periode akuisisi (ISO 8601)
  START_DATE: '2024-06-01',
  END_DATE  : '2024-09-30',

  // Threshold cloud cover per scene (%)
  CLOUD_COVER_MAX: 20,

  // Resolusi output (meter) — S2 native = 10m
  SCALE: 10,

  // Google Drive folder tujuan
  DRIVE_FOLDER: 'GEE_LPP_MADIUN',

  // Prefix nama file output
  FILE_PREFIX: 'S2_VI_Madiun',

  // Nama project GEE (isi jika pakai Cloud Project)
  // PROJECT: 'your-gee-project-id',
};

// ------------------------------------------------------------
// 2. AOI — Kabupaten Madiun, Jawa Timur
//    Bounding box kasar; ganti dengan shapefile blok jika ada
// ------------------------------------------------------------
var AOI = ee.Geometry.Rectangle([
  111.2800,  // lon min (barat)
  -7.8500,   // lat min (selatan)
  111.8200,  // lon max (timur)
  -7.3500    // lat max (utara)
]);

// Visualisasi AOI di peta
Map.centerObject(AOI, 11);
Map.addLayer(AOI, {color: 'FF0000', fillColor: '00000000'}, 'AOI Madiun');

// ------------------------------------------------------------
// 3. FUNGSI MASKING AWAN — S2 Scene Classification Layer (SCL)
// ------------------------------------------------------------
function maskS2Clouds(image) {
  var scl = image.select('SCL');
  // Kelas yang di-mask: 3=shadow, 8=cloud_med, 9=cloud_high, 10=cirrus
  var cloudMask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10));
  return image.updateMask(cloudMask)
    .divide(10000)  // konversi DN → reflectance (0–1)
    .copyProperties(image, ['system:time_start', 'system:index']);
}

// ------------------------------------------------------------
// 4. LOAD & FILTER KOLEKSI SENTINEL-2 SR
// ------------------------------------------------------------
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(CONFIG.START_DATE, CONFIG.END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.CLOUD_COVER_MAX))
  .map(maskS2Clouds);

print('Jumlah scene tersedia:', s2Collection.size());
print('Tanggal scene:', s2Collection.aggregate_array('system:index'));

// ------------------------------------------------------------
// 5. FUNGSI HITUNG VEGETATION INDICES
// ------------------------------------------------------------

// NDVI = (NIR - RED) / (NIR + RED)
// Band: B8=NIR (842nm), B4=RED (665nm)
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

// GNDVI = (NIR - GREEN) / (NIR + GREEN)
// Band: B8=NIR, B3=GREEN (560nm)
function addGNDVI(image) {
  var gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI');
  return image.addBands(gndvi);
}

// SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
// L = 0.5 (soil adjustment factor umum)
// Band: B8=NIR, B4=RED
function addSAVI(image) {
  var L = 0.5;
  var nir = image.select('B8');
  var red = image.select('B4');
  var savi = nir.subtract(red)
    .divide(nir.add(red).add(L))
    .multiply(1 + L)
    .rename('SAVI');
  return image.addBands(savi);
}

// RVI = NIR / RED  (Ratio Vegetation Index)
// Band: B8=NIR, B4=RED
function addRVI(image) {
  var rvi = image.select('B8').divide(image.select('B4')).rename('RVI');
  return image.addBands(rvi);
}

// EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
// Bonus index — berguna untuk kanopi padat sawit
// Band: B8=NIR, B4=RED, B2=BLUE (490nm)
function addEVI(image) {
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {
      'NIR' : image.select('B8'),
      'RED' : image.select('B4'),
      'BLUE': image.select('B2')
    }
  ).rename('EVI');
  return image.addBands(evi);
}

// ------------------------------------------------------------
// 6. TERAPKAN SEMUA VI KE KOLEKSI
// ------------------------------------------------------------
var s2WithVI = s2Collection
  .map(addNDVI)
  .map(addGNDVI)
  .map(addSAVI)
  .map(addRVI)
  .map(addEVI);

// Pilih hanya band VI + band spektral utama
var viBands = ['NDVI', 'GNDVI', 'SAVI', 'RVI', 'EVI'];
var specBands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B11', 'B12'];

// ------------------------------------------------------------
// 7. BUAT KOMPOSIT MEDIAN (mengurangi noise residual awan)
//    Untuk export bulanan, bisa loop per bulan (lihat Catatan)
// ------------------------------------------------------------
var viMedian = s2WithVI
  .select(viBands)
  .median()
  .clip(AOI);

var specMedian = s2WithVI
  .select(specBands)
  .median()
  .clip(AOI);

// ------------------------------------------------------------
// 8. VISUALISASI DI GEE CODE EDITOR
// ------------------------------------------------------------
var ndviVis = {min: -0.1, max: 0.9, palette: [
  '#d73027', '#f46d43', '#fdae61',
  '#fee08b', '#d9ef8b', '#a6d96a',
  '#66bd63', '#1a9850'
]};

var gndviVis = {min: -0.1, max: 0.8, palette: [
  '#8c510a', '#d8b365', '#f6e8c3',
  '#c7eae5', '#5ab4ac', '#01665e'
]};

Map.addLayer(viMedian.select('NDVI'),  ndviVis,  'NDVI Median');
Map.addLayer(viMedian.select('GNDVI'), gndviVis, 'GNDVI Median');
Map.addLayer(viMedian.select('SAVI'),  ndviVis,  'SAVI Median');

// RGB natural color untuk referensi
Map.addLayer(specMedian, {
  bands: ['B4', 'B3', 'B2'],
  min: 0, max: 0.3
}, 'RGB Natural Color');

// False color NIR (vegetasi = merah terang)
Map.addLayer(specMedian, {
  bands: ['B8', 'B4', 'B3'],
  min: 0, max: 0.4
}, 'False Color NIR');

// ------------------------------------------------------------
// 9. STATISTIK PER KELAS (opsional — lihat di Console)
// ------------------------------------------------------------
var viStats = viMedian.reduceRegion({
  reducer: ee.Reducer.mean()
    .combine(ee.Reducer.stdDev(), '', true)
    .combine(ee.Reducer.min(), '', true)
    .combine(ee.Reducer.max(), '', true),
  geometry: AOI,
  scale: CONFIG.SCALE,
  maxPixels: 1e10,
  bestEffort: true
});

print('Statistik VI Median (AOI Madiun):', viStats);

// ------------------------------------------------------------
// 10. EXPORT KE GOOGLE DRIVE
// ------------------------------------------------------------

// --- Export A: Semua VI sebagai single multi-band GeoTIFF ---
Export.image.toDrive({
  image      : viMedian,
  description: CONFIG.FILE_PREFIX + '_VI_Median_' +
               CONFIG.START_DATE.replace(/-/g,'') + '_' +
               CONFIG.END_DATE.replace(/-/g,''),
  folder     : CONFIG.DRIVE_FOLDER,
  fileNamePrefix: CONFIG.FILE_PREFIX + '_VI_Median',
  region     : AOI,
  scale      : CONFIG.SCALE,
  crs        : 'EPSG:32749',  // UTM Zone 49S — cocok untuk Jawa Timur
  maxPixels  : 1e10,
  fileFormat : 'GeoTIFF'
});

// --- Export B: Band spektral utama (untuk keperluan training tambahan) ---
Export.image.toDrive({
  image      : specMedian,
  description: CONFIG.FILE_PREFIX + '_Bands_Median',
  folder     : CONFIG.DRIVE_FOLDER,
  fileNamePrefix: CONFIG.FILE_PREFIX + '_Bands_Median',
  region     : AOI,
  scale      : CONFIG.SCALE,
  crs        : 'EPSG:32749',
  maxPixels  : 1e10,
  fileFormat : 'GeoTIFF'
});

// --- Export C: Per-scene individual (opsional, uncomment jika perlu) ---
/*
var sceneList = s2WithVI.toList(s2WithVI.size());
var nScenes   = s2WithVI.size().getInfo();

for (var i = 0; i < nScenes; i++) {
  var scene = ee.Image(sceneList.get(i));
  var date  = ee.Date(scene.get('system:time_start')).format('YYYYMMdd').getInfo();

  Export.image.toDrive({
    image         : scene.select(viBands).clip(AOI),
    description   : CONFIG.FILE_PREFIX + '_VI_' + date,
    folder        : CONFIG.DRIVE_FOLDER,
    fileNamePrefix: CONFIG.FILE_PREFIX + '_VI_' + date,
    region        : AOI,
    scale         : CONFIG.SCALE,
    crs           : 'EPSG:32749',
    maxPixels     : 1e10,
    fileFormat    : 'GeoTIFF'
  });
}
*/

// ============================================================
// CATATAN PENGGUNAAN
// ============================================================
// 1. Paste script ini ke https://code.earthengine.google.com
// 2. Klik "Run" untuk preview peta dan statistik di Console
// 3. Klik "Tasks" → klik "RUN" di setiap task export
// 4. File akan muncul di Google Drive folder: GEE_LPP_MADIUN/
//
// UNTUK AUTOMASI MINGGUAN (Python API):
//   Ganti export manual di atas dengan ee.batch.Export.image.toDrive(...)
//   dan jalankan via Python scheduler — script terpisah akan dibuat.
//
// EPSG:32749 = WGS84 / UTM Zone 49S (cocok Madiun ~111°E)
// Jika AOI lebih ke barat (Jawa Tengah), pakai EPSG:32748 (UTM 48S)
//
// BAND REFERENCE SENTINEL-2:
//   B2  = Blue    490nm   10m
//   B3  = Green   560nm   10m
//   B4  = Red     665nm   10m
//   B5  = Red Edge 705nm  20m
//   B8  = NIR     842nm   10m
//   B8A = NIR     865nm   20m
//   B11 = SWIR   1610nm   20m
//   B12 = SWIR   2190nm   20m
// ============================================================
