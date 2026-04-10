import rasterio
import numpy as np

with rasterio.open("./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif") as src:
    # Baca B4 (Red) dan B8 (NIR) — band index 4 dan 8
    B4 = src.read(4, out_dtype="float32")
    B8 = src.read(8, out_dtype="float32")
    
    # Hitung NDVI
    eps = 1e-10
    NDVI = (B8 - B4) / (B8 + B4 + eps)
    
    # Statistik distribusi NDVI
    valid = B4 > 0  # piksel non-zero
    print("Total piksel non-zero:", valid.sum())
    print("NDVI di area non-zero:")
    print("  min :", NDVI[valid].min())
    print("  max :", NDVI[valid].max())
    print("  mean:", NDVI[valid].mean())
    
    # Distribusi per rentang
    ndvi_vals = NDVI[valid]
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        pct = 100 * np.sum(ndvi_vals > thr) / len(ndvi_vals)
        print(f"  NDVI > {thr:.1f}: {pct:.1f}% piksel")