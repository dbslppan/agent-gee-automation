# import rasterio
# with rasterio.open("./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif") as src:
#     print("Jumlah band:", src.count)
#     print("Shape:", src.shape)

# import rasterio
# with rasterio.open("./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif") as src:
#     print("Jumlah band:", src.count)
#     print("Deskripsi band:", src.descriptions)
#     print("Resolusi (m):", src.res)
#     # Baca satu piksel sampel dari tiap band
#     sample = src.read()[:, 100, 100]
#     print("Nilai piksel sampel per band:", sample)

import rasterio
import numpy as np

with rasterio.open("./GEE_LPP_MADIUN/S2_VI_Madiun_VI_Median.tif") as src:
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    
    # Cari piksel yang tidak nol
    sample = src.read()  # baca semua band
    valid_pixels = np.where(sample[3] > 0)  # cari di band B4
    if len(valid_pixels[0]) > 0:
        r, c = valid_pixels[0][0], valid_pixels[1][0]
        print(f"Piksel valid pertama di row={r}, col={c}")
        print("Nilai per band:", sample[:, r, c])
    else:
        print("Tidak ada piksel valid sama sekali!")