Version 2

Note: We wrapped the first working version of this tool and, only hours later, reports came in of heavy flooding in parts of Davao City. That timing flipped our next milestone on its head. Instead of adding another visual layer, we needed to answer a more basic question:

If we had run the tool just before the flood, would it have highlighted the same blocks that were hit?

That simple question exposed a practical problem with live, API-driven maps: once data moves on, it’s hard to reproduce the exact inputs that led to a result. Elevation is static, but OSM edits evolve, basemaps refresh, and weather forecasts roll forward. We needed a way to “time-travel” the pipeline so we could validate our assumptions against a real event—offline, reproducibly, and without guessing what the inputs looked like yesterday.

That’s how Snapshot & Replay was born:

--snapshot_dir: a self-contained folder capturing everything needed for a run (basemap tiles, Overpass JSON, SRTM samples, rain JSON, and the derived risk_grid.npy).

--snapshot_mode {write, read, auto}:

write freezes a point-in-time dataset,

read replays it offline (no network),

auto prefers existing files but fetches if missing.

--compare_to: renders side-by-side (old vs. new) and a diff map so changes are visible at a glance.

When we applied this to the Davao case, the side-by-side/diff workflow helped us sanity-check the model: low-lying clusters and segments near waterways stood out more consistently, while rainfall weighting and wetland proximity needed tuning to avoid over-painting broad areas. Snapshot & Replay didn’t “prove” the model—it made our checks honest and repeatable. That’s the point.

The feature reflects our project’s values: open data, transparent methods, and humility about uncertainty. If you’ve got better factors (flow accumulation, soil, drainage capacity) or stronger priors for weights and thresholds, plug them in—then freeze a snapshot, replay it, and let the results speak.

# Flood Context Microzonation (OpenCV + OSM) — Snapshot & Replay

A lightweight **Python/OpenCV** tool that renders **street‑level flood risk context** on top of OSM/ESRI basemaps using a multi‑factor score (elevation, distance to waterways, wetland proximity, recent rainfall).  
This version adds **Snapshot & Replay** so you can **replay older data** from a place that flooded today and **compare** it with a fresh pull.

> ⚠️ **Disclaimer**: This project is for research/education. It is **not** a life‑safety or early‑warning system. Always follow official advisories.

---

## ✨ Features

- **Microzonation risk overlay** (Green→Yellow→Red) with ocean/sea masking (no overlay on water).
- **High‑risk buildings** highlighted in red (fill + outline) with small non‑overlapping red dots (optional).
- **Snapshot & Replay**
  - `--snapshot_dir` stores a *self‑contained* run (basemap, OSM, elevation, rain, risk grid).
  - `--snapshot_mode {auto,read,write}` controls online/offline behavior.
  - Export **`risk_grid.npy`** for reproducible comparisons.
  - `--compare_to` renders **side‑by‑side** (old vs new) and **diff** maps.
- **Tile‑aligned stitching**; vectors drawn on top; **dynamic output sizing**.
- Clean legend + rain info panel; compact UI boxes.

---

## 🧰 Requirements

- Python **3.9+**
- **pip** and a C++ build toolchain (only standard wheels are used here, so usually not required).

### Install (Windows / macOS / Linux)

```bash
# optional, but recommended
python -m venv .venv            # on Windows you can use: py -m venv .venv
source .venv/bin/activate       # Windows: .\.venv\Scripts\activate

pip install --upgrade pip
pip install numpy opencv-python pillow requests
```

> If you plan to edit/extend, also consider: `pip install matplotlib` (only for your own plots; not required by the tool).

---

## 🚀 Quickstart

1) **Save the script** as `flood_context.py` in this repository.
2) Run a **fresh snapshot** for your area (writes everything to a folder you choose):

```bash
python flood_context.py \
  --lat 14.8424 --lon 120.7948 --zoom 12 \
  --radius 2 --provider osm --grid 36 \
  --snapshot_dir snapshots/2025-09-16-bulacan-current \
  --snapshot_mode write \
  --output map_current.png
```

3) **Replay** an **older** snapshot **offline** (e.g., earlier today or yesterday):

```bash
python flood_context.py \
  --lat 14.8424 --lon 120.7948 --zoom 12 \
  --radius 2 --provider osm --grid 36 \
  --snapshot_dir snapshots/2025-09-15-bulacan-pre-flood \
  --snapshot_mode read \
  --output map_preflood.png
```

4) **Compare** current vs old (creates `comparison_side_by_side.png` and `risk_diff.png` inside the *current* snapshot folder):

```bash
python flood_context.py \
  --lat 14.8424 --lon 120.7948 --zoom 12 \
  --radius 2 --provider osm --grid 36 \
  --snapshot_dir snapshots/2025-09-16-bulacan-current \
  --snapshot_mode auto \
  --compare_to snapshots/2025-09-15-bulacan-pre-flood \
  --output map_current.png
```

- **Side-by-side**: Left = old, Right = current (same color scale).  
- **Diff map**: current − old (blue = decreased risk, red = increased risk).

---

## 📁 Snapshot Folder Layout

Each run creates a self‑contained folder (you choose the path via `--snapshot_dir`). Typical files:

```
snapshots/2025-09-16-bulacan-current/
├─ basemap_osm_r2.png
├─ overpass.json
├─ elev_pts_grid36.json
├─ rain.json
├─ risk_grid.npy            # float32 grid in 0..1 (for reproducible comparisons)
├─ flood_risk.png           # same as --output if you kept default name
├─ comparison_side_by_side.png   # only if --compare_to was used
└─ risk_diff.png                 # only if --compare_to was used
```

> When `--snapshot_mode read`, **no network calls** are made. Missing files will raise clear errors telling you what to provide.

---

## ⚙️ Risk Model (overview)

Per grid cell, the risk score `R ∈ [0,1]` is a weighted sum of normalized components:

- **Elevation** (favoring low terrain within `[elev_low, elev_high]`)
- **Distance to waterways** (exponential decay with scale `D0_water`)
- **Wetland proximity** (exponential decay with scale `D1_wetland`)
- **Recent rainfall** (3‑day total capped by `rain_cap`)

Tune via CLI flags:
```
--w_elev --w_dist --w_wet --w_rain
--elev_low --elev_high --D0_water --D1_wet --rain_cap
```

---

## 🔧 Common Flags (CLI)

```text
--lat/--lon/--zoom            Map center and zoom
--radius                      Tile radius (final canvas = (2*radius+1) tiles per side)
--provider                    basemap: osm | esri
--grid                        Sampling grid resolution (e.g., 36 -> 36x36)
--alpha                       Overlay opacity (0..1)
--draw_risk / --draw_vectors  Toggle layers
--bld_high_thr                Threshold (0..1) for marking buildings as high risk

# Snapshot & Replay
--snapshot_dir PATH           Folder for all inputs/outputs of this run
--snapshot_mode MODE          auto | read | write
--compare_to PATH             Another snapshot_dir to compare against (renders side-by-side & diff)

# Output sizing
--dynamic --max_px --base_scale --scale_per_zoom

# Red dot styling
--dot_radius --dot_min_sep
```

<details>
<summary><strong>Show full help</strong></summary>

```bash
python flood_context.py -h
```
</details>

---

## 🧪 Reproducibility Tips

- Use **`--snapshot_mode write`** to freeze a dataset for a given time/place.
- Re‑run with **`--snapshot_mode read`** to verify the same result offline.
- Keep your **`risk_grid.npy`** — it’s the canonical numeric output for comparisons.
- For publications, keep your exact CLI and snapshot folder hash (or pack the folder).

---

## ❓ FAQ

**Q: What exactly happens in `--snapshot_mode read`?**  
A: The script **only** reads from `--snapshot_dir`. If any of the required files are missing (`basemap_*.png`, `overpass.json`, `elev_pts_grid*.json`, `rain.json`), it raises a clear error. No web requests are made.

**Q: What if I change `--grid`?**  
A: Elevation samples are stored per grid density (e.g., `elev_pts_grid36.json`). If you change `--grid`, the script will expect the matching file in read mode (or fetch data in auto/write mode).

**Q: Can I export GeoTIFFs?**  
A: Not yet, but it’s straightforward to add from `risk_grid.npy`. PRs welcome!

**Q: Why are oceans masked?**  
A: The overlay uses a basemap‑derived water mask (HSV + BGR heuristics) so you don’t paint risk on open water.

---

## 🧾 License & Data Credits

- **Code**: MIT License (see `LICENSE` — you can choose another if you prefer).
- **Data**: © OpenStreetMap contributors, **Open‑Meteo** (forecast), **OpenTopodata** (SRTM90m).  
  Respect each provider’s terms of use & rate limits.

---

## 🤝 Contributing

Issues and PRs are welcome! If you add new factors (e.g., flow accumulation, soil), please document flags and update this README.

---

## 🖼️ Screenshots (placeholders)

Add your outputs here so the repo shows visual results:

```
![Current Map](snapshots/2025-09-16-bulacan-current/map_current.png)
![Side-by-side](snapshots/2025-09-16-bulacan-current/comparison_side_by_side.png)
![Risk Diff](snapshots/2025-09-16-bulacan-current/risk_diff.png)
```

---

## 🧩 Minimal Example Command

```bash
python flood_context.py \
  --lat 14.8424 --lon 120.7948 --zoom 12 \
  --radius 2 --provider osm --grid 36 \
  --snapshot_dir snapshots/example \
  --snapshot_mode write \
  --output flood_risk.png
```
