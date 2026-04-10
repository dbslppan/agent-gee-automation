"""
Revisi Kelas Kesehatan — 3 Kelas
=================================
Sehat  : NDVI > 0.5  (dan indeks lain proporsional)
Sedang : NDVI 0.2–0.5
Stres  : NDVI < 0.2

File ini menggantikan bagian THRESHOLDS dan CLASS_MAP
di bootstrap_labeler_rf_trainer.py dan weekly_agent_pipeline.py
Cukup import dari sini agar konsisten di semua modul.
"""

# ----------------------------------------------------------------
# KELAS & WARNA — 3 kelas
# ----------------------------------------------------------------
CLASS_MAP = {
    1: {"name": "Sehat",  "color": (46,  204,  64), "hex": "#2ecc40"},
    2: {"name": "Sedang", "color": (255, 220,   0), "hex": "#ffdc00"},
    3: {"name": "Stres",  "color": (231,  76,  60), "hex": "#e74c3c"},
}

CLASS_NAMES  = ["Sehat", "Sedang", "Stres"]
CLASS_LABELS = {1: "Sehat", 2: "Sedang", 3: "Stres"}

# ----------------------------------------------------------------
# THRESHOLD RULE-BASED — dipakai pseudo-labeling bootstrap
# Voting mayoritas dari 4 indeks
# ----------------------------------------------------------------
THRESHOLDS = {
    "Sehat": {
        "NDVI_min":  0.50, "NDVI_max":  1.00,
        "GNDVI_min": 0.45, "GNDVI_max": 1.00,
        "SAVI_min":  0.45, "SAVI_max":  1.00,
        "RVI_min":   3.00, "RVI_max": 999.0,
    },
    "Sedang": {
        "NDVI_min":  0.20, "NDVI_max":  0.50,
        "GNDVI_min": 0.18, "GNDVI_max": 0.45,
        "SAVI_min":  0.16, "SAVI_max":  0.45,
        "RVI_min":   1.50, "RVI_max":   3.00,
    },
    "Stres": {
        "NDVI_min": -1.00, "NDVI_max":  0.20,
        "GNDVI_min":-1.00, "GNDVI_max": 0.18,
        "SAVI_min": -1.00, "SAVI_max":  0.16,
        "RVI_min":   0.00, "RVI_max":   1.50,
    },
}

# ----------------------------------------------------------------
# WARNA UNTUK COLORIZE RASTER (lookup array R,G,B per cls_id)
# ----------------------------------------------------------------
COLOR_LOOKUP = {
    0: (0,   0,   0,   0),    # nodata → transparan
    1: (46,  204,  64, 210),  # Sehat
    2: (255, 220,   0, 210),  # Sedang
    3: (231,  76,  60, 210),  # Stres
}
