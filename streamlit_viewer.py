# streamlit_viewer.py
"""
Streamlit Viewer — Peta Kesehatan Kebun
LPP Agro Nusantara | Madiun, Jawa Timur
========================================
Fitur:
  - Login sederhana (password tunggal)
  - Peta interaktif kelas kesehatan (folium + leafmap)
  - Statistik luas per kelas (tabel + chart)
  - Pilih periode historis
  - Download laporan PDF (opsional)

Instalasi:
  pip install streamlit folium leafmap streamlit-folium \
              rasterio numpy pandas matplotlib

Jalankan:
  streamlit run streamlit_viewer.py
"""

import json
from datetime import datetime
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium import plugins
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
APP_PASSWORD   = "lppagro2025"          # ganti sesuai kebutuhan
OUTPUT_DIR     = Path("./agent_output") # folder output agent pipeline
VIEWER_TITLE   = "Peta Kesehatan Kebun — LPP Agro Nusantara"

CLASS_MAP = {
    1: {"name": "Sehat",        "color": "#2ecc40", "icon": "🟢"},
    2: {"name": "Stres Ringan", "color": "#ffdc00", "icon": "🟡"},
    3: {"name": "Stres Berat",  "color": "#ff851b", "icon": "🟠"},
    4: {"name": "Kritis",       "color": "#e74c3c", "icon": "🔴"},
}

# ----------------------------------------------------------------
# HELPER — load data
# ----------------------------------------------------------------
def get_available_dates() -> list[str]:
    """Scan output dir untuk menemukan semua tanggal stats yang tersedia."""
    if not OUTPUT_DIR.exists():
        return []
    stats_files = sorted(OUTPUT_DIR.glob("stats_*.json"), reverse=True)
    return [f.stem.replace("stats_", "") for f in stats_files]


def load_stats(date_tag: str) -> dict | None:
    stats_path = OUTPUT_DIR / f"stats_{date_tag}.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        return json.load(f)


def load_prediction_raster(date_tag: str):
    """Cari pred_*.tif yang sesuai dengan date_tag."""
    pred_files = list(OUTPUT_DIR.glob(f"pred_*{date_tag}*.tif"))
    if not pred_files:
        # Fallback: ambil pred terbaru
        pred_files = sorted(OUTPUT_DIR.glob("pred_*.tif"), reverse=True)
    if not pred_files:
        return None, None
    pred_path = pred_files[0]
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        bounds = src.bounds
        crs = src.crs
        # Konversi ke WGS84 jika perlu
        if str(crs) != "EPSG:4326":
            from rasterio.crs import CRS
            wgs84 = CRS.from_epsg(4326)
            left, bottom, right, top = transform_bounds(crs, wgs84, *bounds)
        else:
            left, bottom, right, top = bounds
    return pred, [bottom, left, top, right]  # format folium: [S, W, N, E]


def pred_to_rgba_png(pred: np.ndarray) -> np.ndarray:
    """Konversi prediction array ke RGBA image untuk folium overlay."""
    H, W = pred.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    color_lookup = {
        1: (46,  204, 64,  200),
        2: (255, 220,  0,  200),
        3: (255, 133, 27,  200),
        4: (231, 76,  60,  200),
        0: (0,   0,   0,   0),   # nodata = transparan
    }
    for cls_id, rgba_val in color_lookup.items():
        mask = pred == cls_id
        rgba[mask] = rgba_val
    return rgba


# ----------------------------------------------------------------
# HALAMAN LOGIN
# ----------------------------------------------------------------
def page_login():
    st.markdown(
        "<h2 style='text-align:center;margin-top:60px'>🌿 LPP Agro Nusantara</h2>"
        "<p style='text-align:center;color:gray'>Sistem Monitoring Kesehatan Kebun</p>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        pwd = st.text_input("Password", type="password",
                            placeholder="Masukkan password akses")
        if st.button("Masuk", use_container_width=True, type="primary"):
            if pwd == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Password salah. Hubungi tim Digital Intelligence LPP.")


# ----------------------------------------------------------------
# HALAMAN UTAMA
# ----------------------------------------------------------------
def page_dashboard():
    # Header
    st.markdown(
        f"<h2 style='margin-bottom:0'>{VIEWER_TITLE}</h2>"
        f"<p style='color:gray;margin-top:4px'>Madiun, Jawa Timur — Update otomatis setiap 5–7 hari</p>",
        unsafe_allow_html=True
    )

    # Tombol logout
    if st.sidebar.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    # ---- Sidebar: Pilih Periode ----
    st.sidebar.markdown("### Pengaturan")
    available_dates = get_available_dates()

    if not available_dates:
        st.warning(
            "Belum ada data tersedia. Jalankan `weekly_agent_pipeline.py` "
            "terlebih dahulu untuk memproses data pertama."
        )
        _render_demo_mode()
        return

    date_options = {d: f"Periode {d[:4]}-{d[4:6]}-{d[6:]}" for d in available_dates}
    selected_raw = st.sidebar.selectbox(
        "Pilih periode",
        options=available_dates,
        format_func=lambda d: date_options[d]
    )
    selected_date = selected_raw

    show_rgb     = st.sidebar.checkbox("Tampilkan warna peta", value=True)
    show_legend  = st.sidebar.checkbox("Tampilkan legenda", value=True)
    map_tiles    = st.sidebar.selectbox(
        "Basemap",
        ["OpenStreetMap", "Satellite (Esri)", "Terrain"]
    )

    # ---- Load data ----
    stats = load_stats(selected_date)
    pred, bounds = load_prediction_raster(selected_date)

    if stats is None:
        st.error(f"Data statistik untuk {selected_date} tidak ditemukan.")
        return

    # ---- KPI Cards ----
    st.markdown("#### Ringkasan Kondisi Kebun")
    cols = st.columns(4)
    for i, (cls_id, cls_info) in enumerate(CLASS_MAP.items()):
        cls_stats = stats["classes"].get(cls_info["name"], {})
        area_ha   = cls_stats.get("area_ha", 0)
        pct       = cls_stats.get("percentage", 0)
        with cols[i]:
            st.metric(
                label=f"{cls_info['icon']} {cls_info['name']}",
                value=f"{area_ha:,.0f} ha",
                delta=f"{pct:.1f}% dari total",
                delta_color="off"
            )

    st.markdown(
        f"<p style='color:gray;font-size:0.85em'>"
        f"Total area teranalisis: <b>{stats['total_valid_ha']:,.0f} ha</b> | "
        f"Resolusi: <b>{stats.get('resolution_m', 10):.0f} m/piksel</b> | "
        f"Tanggal: <b>{selected_date[:4]}-{selected_date[4:6]}-{selected_date[6:]}</b>"
        f"</p>",
        unsafe_allow_html=True
    )

    # ---- Layout: Peta + Chart ----
    col_map, col_chart = st.columns([2, 1])

    with col_map:
        st.markdown("#### Peta Kesehatan Kebun")
        _render_map(pred, bounds, map_tiles, show_legend)

    with col_chart:
        st.markdown("#### Distribusi Kelas")
        _render_donut_chart(stats)

        st.markdown("#### Detail per Kelas")
        rows = []
        for cls_name, cls_data in stats["classes"].items():
            rows.append({
                "Kelas"     : cls_name,
                "Luas (ha)" : f"{cls_data['area_ha']:,.1f}",
                "Persentase": f"{cls_data['percentage']:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- Trend historis (jika ada >1 periode) ----
    if len(available_dates) > 1:
        st.markdown("---")
        st.markdown("#### Tren Historis Kesehatan Kebun")
        _render_trend_chart(available_dates)

    # ---- Download ----
    st.markdown("---")
    pie_path = OUTPUT_DIR / f"pie_chart_{selected_date}.png"
    stats_path = OUTPUT_DIR / f"stats_{selected_date}.json"

    dl_col1, dl_col2, _ = st.columns([1, 1, 3])
    if pie_path.exists():
        with open(pie_path, "rb") as f:
            dl_col1.download_button(
                "⬇️ Download Chart",
                data=f,
                file_name=f"health_chart_{selected_date}.png",
                mime="image/png"
            )
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            dl_col2.download_button(
                "⬇️ Download Statistik",
                data=f,
                file_name=f"health_stats_{selected_date}.json",
                mime="application/json"
            )


# ----------------------------------------------------------------
# RENDER MAP
# ----------------------------------------------------------------
def _render_map(pred, bounds, map_tiles: str, show_legend: bool):
    tile_providers = {
        "OpenStreetMap"    : ("OpenStreetMap", {}),
        "Satellite (Esri)" : ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                               "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                               {"attr": "Esri", "name": "Satellite"}),
        "Terrain"          : ("Stamen Terrain", {}),
    }

    if bounds:
        center_lat = (bounds[0] + bounds[2]) / 2
        center_lon = (bounds[1] + bounds[3]) / 2
    else:
        center_lat, center_lon = -7.60, 111.52  # Madiun default

    tile_url, tile_kwargs = tile_providers.get(
        map_tiles, ("OpenStreetMap", {})
    )

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tile_url if "http" not in tile_url else None,
    )

    if "http" in tile_url:
        folium.TileLayer(tile_url, **tile_kwargs).add_to(m)

    # Overlay prediksi sebagai image layer
    if pred is not None and bounds is not None:
        import io
        from PIL import Image as PILImage

        rgba = pred_to_rgba_png(pred)
        pil_img = PILImage.fromarray(rgba, "RGBA")

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        import base64
        img_b64 = base64.b64encode(buf.read()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
            opacity=0.75,
            name="Kelas Kesehatan"
        ).add_to(m)

    # Legenda
    if show_legend:
        legend_html = """
        <div style='position:absolute;bottom:30px;left:30px;z-index:999;
                    background:white;padding:10px 14px;border-radius:8px;
                    border:1px solid #ccc;font-size:13px;line-height:1.8'>
        <b>Kelas Kesehatan</b><br>
        <span style='color:#2ecc40'>&#9632;</span> Sehat<br>
        <span style='color:#ffdc00'>&#9632;</span> Stres Ringan<br>
        <span style='color:#ff851b'>&#9632;</span> Stres Berat<br>
        <span style='color:#e74c3c'>&#9632;</span> Kritis
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    plugins.Fullscreen().add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, height=480, use_container_width=True)


# ----------------------------------------------------------------
# RENDER DONUT CHART
# ----------------------------------------------------------------
def _render_donut_chart(stats: dict):
    labels, sizes, colors = [], [], []
    for cls_name, cls_data in stats["classes"].items():
        if cls_data["pixel_count"] > 0:
            labels.append(f"{cls_name}\n{cls_data['area_ha']:.0f} ha")
            sizes.append(cls_data["area_ha"])
            colors.append(cls_data["color_hex"])

    if not sizes:
        st.info("Belum ada data untuk ditampilkan.")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5}
    )
    ax.legend(wedges, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------------------------------------------
# RENDER TREND CHART
# ----------------------------------------------------------------
def _render_trend_chart(available_dates: list[str]):
    all_stats = []
    for d in sorted(available_dates):
        s = load_stats(d)
        if s:
            row = {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}"}
            for cls_name, cls_data in s["classes"].items():
                row[cls_name] = cls_data["percentage"]
            all_stats.append(row)

    if len(all_stats) < 2:
        st.info("Minimal 2 periode diperlukan untuk melihat tren.")
        return

    df_trend = pd.DataFrame(all_stats).set_index("date")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors_list = ["#2ecc40", "#ffdc00", "#ff851b", "#e74c3c"]
    cls_cols = ["Sehat", "Stres Ringan", "Stres Berat", "Kritis"]

    for col, color in zip(cls_cols, colors_list):
        if col in df_trend.columns:
            ax.plot(df_trend.index, df_trend[col], marker="o",
                    label=col, color=color, linewidth=2, markersize=5)

    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------------------------------------------
# DEMO MODE (jika belum ada data)
# ----------------------------------------------------------------
def _render_demo_mode():
    st.markdown("---")
    st.info(
        "**Demo mode** — tampilan ini akan terisi otomatis setelah "
        "pipeline pertama selesai berjalan.\n\n"
        "Langkah selanjutnya:\n"
        "1. Pastikan GEE export selesai dan file `.tif` ada di `GEE_LPP_MADIUN/`\n"
        "2. Jalankan `python bootstrap_labeler_rf_trainer.py` untuk buat `model.pkl`\n"
        "3. Jalankan `python weekly_agent_pipeline.py` untuk proses pertama\n"
        "4. Refresh halaman ini"
    )

    # Preview dummy peta
    m = folium.Map(location=[-7.60, 111.52], zoom_start=11)
    folium.Marker(
        [-7.60, 111.52],
        popup="AOI Kebun Madiun",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)
    st_folium(m, height=350, use_container_width=True)


# ================================================================
# MAIN
# ================================================================
def main():
    st.set_page_config(
        page_title="Kesehatan Kebun — LPP Agro",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sembunyikan menu dan footer Streamlit
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        page_login()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
