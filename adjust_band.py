import rasterio
import numpy as np

with rasterio.open("./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif") as src:
    print(f"Jumlah band : {src.count}")
    print(f"Deskripsi   : {src.descriptions}")
    
    for i in range(1, src.count + 1):
        band = src.read(i)
        valid = band[(band > -9999) & np.isfinite(band)]
        if len(valid) > 0:
            print(f"Band {i:2d}: min={valid.min():.4f}, max={valid.max():.4f}")
        else:
            print(f"Band {i:2d}: tidak ada data valid")