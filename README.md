# Flood‑Context Microzonation (OpenCV + OSM/ESRI)

A single‑file, open‑source tool to render **street‑level flood‑context maps** with a **multi‑factor risk overlay** on top of OSM or ESRI basemaps. It fetches OpenStreetMap vectors (waterways, wetlands, embankments, buildings), samples SRTM90m elevations, pulls Open‑Meteo rainfall, and blends a colorized risk surface **only over land**—never over oceans/seas. High‑risk buildings are highlighted in solid red with a **small, non‑overlapping red centroid dot**. Legend and info panel are compact and readable.

---

## ✨ Features

- **Land‑only risk overlay** using a basemap‑derived water mask (no color cast over ocean/sea).
- **Vector layers**: waterways, wetlands, embankments/levees, buildings.
- **High‑risk buildings** visibly marked (red fill + outline) with **tiny non‑overlapping dots**.
- **Multi‑factor risk model** (elevation, distance to water, wetlands proximity, recent rain).
- **Cache‑aware** fetching (basemap, Overpass, SRTM, rainfall) keyed by `zoom/lat/lon`.
- **Tile‑aligned** stitching to keep overlays perfectly registered.
- **Compact legend** + **rain info panel** drawn last for visibility.
- **Dynamic output scaling** by zoom/radius with pixel cap.
- **Fully configurable** via CLI flags (weights, thresholds, dot size/spacing, cache, etc.).

---

## 🧰 Installation

- **Python** ≥ 3.8
- **Packages**
  ```bash
  pip install opencv-python numpy pillow requests
  ```

> If you use a Conda environment on Windows:
> ```bash
> conda create -n floodcv python=3.11 -y
> conda activate floodcv
> pip install opencv-python numpy pillow requests
> ```

---

## 🚀 Quick start

1. Save the script as `Microzonation_Flood_Analysis.py` (or use your filename).
2. Run with default AOI:
   ```bash
   python Microzonation_Flood_Analysis.py --output flood_risk.png
   ```
3. Open the generated `flood_risk.png`.

**Example (General Santos area, OSM tiles, larger radius, stronger overlay):**
```bash
python Microzonation_Flood_Analysis.py \
  --lat 6.1164 --lon 125.1716 --zoom 12 \
  --radius 2 --provider osm --alpha 0.42 \
  --grid 36 --output gensan_flood.png
```

**Example (Bulacan‑adjacent AOI, ESRI tiles):**
```bash
python Microzonation_Flood_Analysis.py \
  --lat 14.8424 --lon 120.7948 --zoom 12 \
  --provider esri --radius 2 --alpha 0.38 \
  --output bulacan_flood.png
```

---

## 🧪 CLI Overview

Run `-h` for the full list:
```bash
python Microzonation_Flood_Analysis.py -h
```

Key flags (selection):

| Flag | Type / Default | Purpose |
|---|---|---|
| `--lat`, `--lon`, `--zoom` | float, float, int (14.8424, 120.7948, 12) | Map center & zoom |
| `--dlat`, `--dlon`, `--dz` | float, float, int (0,0,0) | Nudge the AOI without recomputing everything |
| `--radius` | int (2) | Tiles around the center; final map spans `(2*radius+1) × (2*radius+1)` tiles |
| `--provider` | `osm` \| `esri` (osm) | Pick basemap |
| `--grid` | int (36) | Internal risk grid resolution before interpolation |
| `--alpha` | float (0.38) | Overlay opacity for the risk surface |
| `--draw_risk` / `--draw_vectors` | flags (on) | Toggle major layers |
| `--w_elev`, `--w_dist`, `--w_wet`, `--w_rain` | floats (0.5, 0.3, 0.1, 0.1) | Risk weights (sum not required to be 1) |
| `--elev_low`, `--elev_high` | floats (0, 20) | Elevation scoring breakpoints (m) |
| `--D0_water`, `--D1_wet` | floats (300, 200) | Distance decay (m) to water/wetlands |
| `--rain_cap` | float (100.0) | Cap for 3‑day rain (mm) in risk scaling |
| `--bld_high_thr` | float (0.6) | Mean‑risk threshold to mark buildings red |
| `--dot_radius` | int (2) | Red centroid dot radius (px) |
| `--dot_min_sep` | int (6) | Min. center‑to‑center spacing between dots (px) |
| `--dynamic`, `--max_px` | flag, int (on, 3600) | Scale output based on zoom/radius with size cap |
| `--base_scale`, `--scale_per_zoom` | floats (1.0, 0.35) | Controls dynamic scaling behavior |
| `--cache_base` | str (`cache/`) | Cache root folder (auto‑created) |
| `--no_cache` | flag | Force refetch all data |
| `--sleep_ms` | int (180) | SRTM request pacing per chunk |

---

## 🧮 How the risk is computed (high level)

For each grid cell:
- **Elevation score**: 1 at/below `elev_low`, linearly decreasing to 0 at `elev_high`.
- **Distance to water**: distance transform on waterway lines → `exp(-d / D0_water)`.
- **Wetland proximity**: distance transform on wetland lines → `exp(-d / D1_wet)`.
- **Rain**: 3‑day precipitation (Open‑Meteo), scaled to `[0,1]` by `rain_cap`.

Final risk (clamped to `[0,1]`):
```
R = w_elev * S_elev + w_dist * S_water + w_wet * S_wet + w_rain * S_rain
```
It is **interpolated to full image size** and colorized **green→yellow→red** before blending over **land only**.

**High‑risk buildings**: for each building polygon, the mean of covered risk pixels is compared with `--bld_high_thr`. If ≥ threshold, the building gets red fill + red outline and a **small red centroid dot** (dots avoid overlaps using min‑distance logic).

---

## 📁 Caching

A cache folder is created per AOI:
```
cache/zoom{Z}_lat{LAT}_lon{LON}/
├── basemap_{provider}_r{radius}.png
├── overpass.json
├── elev_pts_grid{N}.json
└── rain.json
```
Use `--no_cache` to bypass caches. Re‑runs with the same AOI reuse prior results to save time and rate limits.

---

## 🗺️ Data sources & attribution

- **Basemaps**: OpenStreetMap standard tiles or ESRI World Imagery
- **Vectors**: Overpass (OpenStreetMap data)
- **Elevation**: OpenTopodata SRTM90m
- **Rain**: Open‑Meteo forecast API

> Please respect each service’s **Terms of Use** and rate limits. Attribute OSM data as required.


---

## 🧑‍💻 Contributing

Issues and PRs welcome! Ideas:
- Pluggable risk models & weights presets
- Optional flow‑accumulation layer
- Configurable color maps and legends
- Export of per‑building risk CSV/GeoJSON

---

## 📜 License

This repository is offered under the **MIT License** (code) with data usage subject to the respective providers’ Terms of Use (OSM, ESRI, OpenTopodata, Open‑Meteo).

---

## 🙏 Acknowledgments

- OpenStreetMap contributors
- ESRI World Imagery
- OpenTopodata (SRTM90m)
- Open‑Meteo
- OpenCV & NumPy communities
