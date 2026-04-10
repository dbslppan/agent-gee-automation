"""
Streamlit Viewer v2 — Interactive Tiles + Hover Popup
LPP Agro Nusantara | Madiun, Jawa Timur
======================================================
Perubahan dari v1:
  - 3 kelas: Sehat / Sedang / Stres
  - Peta berbasis GeoJSON per poligon (bukan image overlay)
  - Hover + klik → popup detail per lahan_id
  - Informasi: lahan_id, kelas dominan, ha per kelas, % per kelas
  - Semua atribut asli shapefile ikut ditampilkan
  - Pilih file GeoJSON dari Drive (multi-periode)

Instalasi:
  pip install streamlit geopandas folium streamlit-folium pandas matplotlib
"""

import json
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from folium import plugins
from streamlit_folium import st_folium

# ----------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------
APP_PASSWORD  = "lppagro2025"
CLIP_OUT_DIR  = Path("./clip_output")   # output clip_tile_agent.py
VIEWER_TITLE  = "Peta Kesehatan Kebun — LPP Agro Nusantara"

CLASS_MAP = {
    "Sehat" : {"color": "#2ecc40", "icon": "🟢", "fill_opacity": 0.65},
    "Sedang": {"color": "#ffdc00", "icon": "🟡", "fill_opacity": 0.65},
    "Stres" : {"color": "#e74c3c", "icon": "🔴", "fill_opacity": 0.65},
}

# Kolom atribut shapefile yang ingin ditampilkan di popup
# Kosongkan [] untuk tampilkan semua kolom
POPUP_EXTRA_COLS = []  # contoh: ["nama_blok", "afdeling", "tahun_tanam"]


# ----------------------------------------------------------------
# HELPER
# ----------------------------------------------------------------
def get_available_geojson() -> list[Path]:
    if not CLIP_OUT_DIR.exists():
        return []
    files = sorted(CLIP_OUT_DIR.glob("health_*.geojson"), reverse=True)
    return files


def load_geojson(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def dominant_color(row) -> str:
    kelas = row.get("kelas_dominan", "")
    return CLASS_MAP.get(kelas, {}).get("color", "#888888")


# ----------------------------------------------------------------
# LOGIN
# ----------------------------------------------------------------
def page_login():
    st.markdown(
        "<h2 style='text-align:center;margin-top:60px'>🌿 LPP Agro Nusantara</h2>"
        "<p style='text-align:center;color:gray'>Sistem Monitoring Kesehatan Kebun</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("---")
        pwd = st.text_input("Password", type="password")
        if st.button("Masuk", use_container_width=True, type="primary"):
            if pwd == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Password salah.")


# ----------------------------------------------------------------
# POPUP HTML per poligon
# ----------------------------------------------------------------
def build_popup_html(row: pd.Series, extra_cols: list) -> str:
    lahan_id    = row.get("lahan_id", "—")
    kelas_dom   = row.get("kelas_dominan", "—")
    total_ha    = row.get("total_ha", 0)
    icon        = CLASS_MAP.get(kelas_dom, {}).get("icon", "⚪")
    color       = CLASS_MAP.get(kelas_dom, {}).get("color", "#888")

    # Baris per kelas
    kelas_rows = ""
    for cls_name in ["Sehat", "Sedang", "Stres"]:
        ckey  = cls_name.lower()
        ha    = row.get(f"ha_{ckey}", 0) or 0
        pct   = row.get(f"pct_{ckey}", 0) or 0
        c     = CLASS_MAP[cls_name]["color"]
        kelas_rows += f"""
        <tr>
          <td><span style='color:{c};font-size:14px'>&#9632;</span> {cls_name}</td>
          <td style='text-align:right'>{ha:,.1f} ha</td>
          <td style='text-align:right'>{pct:.1f}%</td>
        </tr>"""

    # Kolom atribut tambahan dari shapefile
    extra_rows = ""
    for col in extra_cols:
        val = row.get(col, "—")
        if val and val == val:  # skip NaN
            extra_rows += f"<tr><td><b>{col}</b></td><td colspan='2'>{val}</td></tr>"

    html = f"""
    <div style='font-family:sans-serif;min-width:220px'>
      <div style='background:{color};color:white;padding:6px 10px;
                  border-radius:6px 6px 0 0;font-weight:600;font-size:13px'>
        {icon} Lahan ID: {lahan_id}
      </div>
      <div style='padding:8px 10px;border:1px solid #ddd;
                  border-top:none;border-radius:0 0 6px 6px'>
        <p style='margin:4px 0;font-size:12px;color:#555'>
          Kelas dominan: <b style='color:{color}'>{kelas_dom}</b>
          &nbsp;|&nbsp; Total: <b>{total_ha:,.1f} ha</b>
        </p>
        <table style='width:100%;font-size:12px;border-collapse:collapse'>
          <tr style='border-bottom:1px solid #eee'>
            <th style='text-align:left'>Kelas</th>
            <th style='text-align:right'>Luas</th>
            <th style='text-align:right'>%</th>
          </tr>
          {kelas_rows}
          {extra_rows}
        </table>
      </div>
    </div>
    """
    return html


# ----------------------------------------------------------------
# RENDER MAP — GeoJSON tiles dengan hover + popup
# ----------------------------------------------------------------
def render_map(gdf: gpd.GeoDataFrame, basemap: str, opacity: float):
    # Hitung center
    bounds    = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    basemap_tiles = {
        "Satellite (Esri)": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri"
        ),
        "OpenStreetMap": ("OpenStreetMap", None),
        "CartoDB Dark" : ("CartoDB dark_matter", None),
    }

    tile_url, tile_attr = basemap_tiles.get(basemap, ("OpenStreetMap", None))

    if tile_attr:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=tile_url,
            attr=tile_attr,
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=tile_url,
        )

    # Tambahkan OpenStreetMap sebagai layer opsional
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # ---- GeoJSON Layer dengan style per polygon + popup ----
    def style_function(feature):
        kelas = feature["properties"].get("kelas_dominan", "")
        color = CLASS_MAP.get(kelas, {}).get("color", "#888888")
        return {
            "fillColor"  : color,
            "color"      : "#ffffff",
            "weight"     : 0.8,
            "fillOpacity": opacity,
        }

    def highlight_function(feature):
        return {
            "fillColor"  : "#ffffff",
            "color"      : "#333333",
            "weight"     : 2.5,
            "fillOpacity": 0.85,
        }

    # Konversi ke dict untuk folium
    geojson_data = json.loads(gdf.to_json())

    # Inject popup HTML ke properties
    for feature in geojson_data["features"]:
        props = feature["properties"]
        row   = pd.Series(props)
        popup_html = build_popup_html(row, POPUP_EXTRA_COLS)
        props["_popup_html"] = popup_html

    folium.GeoJson(
        geojson_data,
        name="Kesehatan Kebun",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["lahan_id", "kelas_dominan", "total_ha"],
            aliases=["Lahan ID:", "Kelas Dominan:", "Total (ha):"],
            localize=True,
            sticky=True,
            style=(
                "background-color: white; color: #333; "
                "font-family: sans-serif; font-size: 12px; "
                "padding: 6px 8px; border-radius: 4px; "
                "border: 1px solid #ccc;"
            ),
        ),
        popup=folium.GeoJsonPopup(
            fields=["_popup_html"],
            aliases=[""],
            labels=False,
            style="max-width:280px;padding:0",
        ),
        zoom_on_click=True,
    ).add_to(m)

    # ---- Legenda ----
    legend_html = """
    <div style='position:absolute;bottom:28px;left:28px;z-index:1000;
                background:rgba(255,255,255,0.95);padding:10px 14px;
                border-radius:8px;border:1px solid #ddd;font-size:12px;
                box-shadow:0 2px 6px rgba(0,0,0,0.15);line-height:2'>
      <b style='font-size:13px'>Kelas Kesehatan</b><br>
      <span style='color:#2ecc40;font-size:18px'>&#9632;</span>&nbsp;Sehat (NDVI &gt; 0.5)<br>
      <span style='color:#ffdc00;font-size:18px'>&#9632;</span>&nbsp;Sedang (0.2–0.5)<br>
      <span style='color:#e74c3c;font-size:18px'>&#9632;</span>&nbsp;Stres (NDVI &lt; 0.2)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Fit bounds ke layer
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ----------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------
def page_dashboard():
    st.markdown(
        f"<h2 style='margin-bottom:2px'>{VIEWER_TITLE}</h2>"
        f"<p style='color:gray;margin-top:0'>Madiun, Jawa Timur — "
        f"GeoJSON Tiles | Hover/Klik poligon untuk detail lahan</p>",
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    # ---- Sidebar ----
    st.sidebar.markdown("### Layer & Tampilan")

    geojson_files = get_available_geojson()
    if not geojson_files:
        _demo_mode()
        return

    file_options = {f: f.stem.replace("health_", "").replace("_", " ").title()
                    for f in geojson_files}
    selected_file = st.sidebar.selectbox(
        "Pilih area & periode",
        options=geojson_files,
        format_func=lambda f: file_options[f],
    )

    basemap  = st.sidebar.selectbox(
        "Basemap", ["Satellite (Esri)", "OpenStreetMap", "CartoDB Dark"]
    )
    opacity  = st.sidebar.slider("Opacity layer", 0.3, 1.0, 0.65, 0.05)
    show_tbl = st.sidebar.checkbox("Tampilkan tabel statistik", value=True)

    # ---- Load GeoJSON ----
    gdf = load_geojson(selected_file)
    date_label = selected_file.stem.split("_")[-1]

    # ---- KPI Cards ----
    st.markdown(f"#### Ringkasan — {file_options[selected_file]}")
    cols = st.columns(4)

    total_ha = gdf["total_ha"].sum() if "total_ha" in gdf.columns else 0
    n_lahan  = len(gdf)

    kpi_data = [
        ("Total Lahan", f"{n_lahan:,}", "poligon"),
        ("Total Area",  f"{total_ha:,.0f}", "ha"),
    ]
    for cls_name in ["Sehat", "Sedang", "Stres"]:
        ckey = cls_name.lower()
        ha   = gdf[f"ha_{ckey}"].sum() if f"ha_{ckey}" in gdf.columns else 0
        pct  = 100 * ha / total_ha if total_ha > 0 else 0
        kpi_data.append((cls_name, f"{ha:,.0f}", f"ha ({pct:.1f}%)"))

    for i, (label, val, sub) in enumerate(kpi_data[:4]):
        cols[i].metric(label=label, value=val, delta=sub, delta_color="off")

    # ---- Layout: Peta + Chart ----
    col_map, col_right = st.columns([3, 1])

    with col_map:
        m = render_map(gdf, basemap, opacity)
        map_result = st_folium(
            m, height=520, use_container_width=True, returned_objects=["last_clicked"]
        )

    with col_right:
        st.markdown("#### Distribusi")
        _donut_chart(gdf)

        st.markdown("#### Kelas dominan")
        if "kelas_dominan" in gdf.columns:
            counts = gdf["kelas_dominan"].value_counts()
            for cls_name, cnt in counts.items():
                icon = CLASS_MAP.get(cls_name, {}).get("icon", "⚪")
                st.write(f"{icon} **{cls_name}**: {cnt} lahan")

    # ---- Tabel statistik ----
    if show_tbl and not gdf.empty:
        st.markdown("---")
        st.markdown("#### Detail per Lahan")

        display_cols = ["lahan_id", "kelas_dominan", "total_ha",
                        "ha_sehat", "pct_sehat",
                        "ha_sedang", "pct_sedang",
                        "ha_stres", "pct_stres"]
        # Tambahkan kolom ekstra jika ada
        display_cols += [c for c in POPUP_EXTRA_COLS if c in gdf.columns]
        display_cols  = [c for c in display_cols if c in gdf.columns]

        df_display = gdf[display_cols].copy()
        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]

        # Highlight baris Stres
        def highlight_stress(row):
            kd = row.get("Kelas Dominan", "")
            if kd == "Stres":
                return ["background-color: #ffe5e5"] * len(row)
            elif kd == "Sedang":
                return ["background-color: #fffde0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_stress, axis=1),
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Download GeoJSON
        with open(selected_file, "rb") as f:
            st.download_button(
                "⬇️ Download GeoJSON",
                data=f,
                file_name=selected_file.name,
                mime="application/geo+json",
            )

        # Download CSV
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=selected_file.stem + ".csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------
# DONUT CHART
# ----------------------------------------------------------------
def _donut_chart(gdf: gpd.GeoDataFrame):
    total_ha = gdf["total_ha"].sum() if "total_ha" in gdf.columns else 0
    if total_ha == 0:
        return

    sizes, colors, labels = [], [], []
    for cls_name in ["Sehat", "Sedang", "Stres"]:
        ckey = cls_name.lower()
        ha   = gdf[f"ha_{ckey}"].sum() if f"ha_{ckey}" in gdf.columns else 0
        pct  = 100 * ha / total_ha
        if ha > 0:
            sizes.append(ha)
            colors.append(CLASS_MAP[cls_name]["color"])
            labels.append(f"{cls_name}\n{ha:,.0f} ha ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.legend(labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------------------------------------------
# DEMO MODE
# ----------------------------------------------------------------
def _demo_mode():
    st.info(
        "**Belum ada data GeoJSON.**\n\n"
        "Jalankan urutan berikut:\n"
        "1. `python bootstrap_labeler_rf_trainer.py`\n"
        "2. `python weekly_agent_pipeline.py`\n"
        "3. `python clip_tile_agent.py` ← menghasilkan GeoJSON per poligon\n"
        "4. Refresh halaman ini"
    )
    m = folium.Map(location=[-7.60, 111.52], zoom_start=11)
    folium.Marker([-7.60, 111.52], popup="AOI Madiun",
                  icon=folium.Icon(color="green", icon="leaf")).add_to(m)
    st_folium(m, height=380, use_container_width=True)


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Kesehatan Kebun — LPP Agro",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        .stDeployButton {visibility:hidden;}
        [data-testid="stMetricDelta"] svg {display:none;}
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
