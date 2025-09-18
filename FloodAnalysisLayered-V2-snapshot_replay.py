#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenCV flood-context map with microzonation overlay on OSM/ESRI tiles.

New in this version (Snapshot & Replay):
- --snapshot_dir PATH: use a single folder for all inputs/outputs of a run
- --snapshot_mode {auto,read,write}: control whether to reuse, strictly read offline, or fetch fresh
- Export risk grid to risk_grid.npy for reproducible comparisons
- --compare_to PATH: load another snapshot's risk_grid.npy and render side-by-side and diff maps
- No network calls in --snapshot_mode read (strict offline)

Kept features:
- NO risk overlay over oceans/seas (basemap-derived water mask).
- High-risk buildings: red fill + red outline + SMALL, NON-OVERLAPPING RED DOT at centroid.
- Legend panel made smaller while increasing text size for readability.
- CLI options for red dot size and spacing.
- Tile-aligned stitching; vectors drawn on top; caching; info panel last; dynamic sizing.
"""

import os
import json
import math
import argparse
import io
import time
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image
import requests
import numpy as np
import cv2

TILE_SIZE = 256

PROVIDERS = {
    "esri": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "osm":  "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

UA = "FloodContextCV/1.3 (non-commercial; contact: example@local)"

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
OPENTOPO_URL  = "https://api.opentopodata.org/v1/srtm90m"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

OSM_QUERIES = {
    "waterways": '(way["waterway"~"river|stream|canal|drain"]({s},{w},{n},{e}););',
    "wetlands":  '(way["natural"="wetland"]({s},{w},{n},{e});relation["natural"="wetland"]({s},{w},{n},{e}););',
    "embank":    '(way["man_made"~"embankment|dyke|dike"]({s},{w},{n},{e}););',
    "buildings": '(way["building"]({s},{w},{n},{e}););'
}

def log(msg: str, verbose: bool = True):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------- Snapshot helpers ----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def must_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def try_read_json(path: str) -> Optional[Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_img_if_exists(path: str):
    if os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            return im
    return None

def save_risk_grid(snapshot_dir: str, grid: np.ndarray):
    np.save(os.path.join(snapshot_dir, "risk_grid.npy"), grid)

def load_risk_grid(snapshot_dir: str) -> Optional[np.ndarray]:
    p = os.path.join(snapshot_dir, "risk_grid.npy")
    if os.path.exists(p):
        try:
            arr = np.load(p)
            if isinstance(arr, np.ndarray):
                return arr
        except Exception:
            return None
    return None

# ---------------- Tiles & projection ----------------
def deg2xy(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    siny = math.sin(math.radians(lat_deg))
    siny = min(max(siny, -0.9999), 0.9999)
    x = TILE_SIZE * (0.5 + lon_deg / 360.0) * (2 ** zoom)
    y = TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * (2 ** zoom)
    return x, y

def meters_per_pixel(lat_deg: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)

def fetch_tile(session: requests.Session, url: str, verbose: bool) -> Image.Image:
    log(f"Fetching tile: {url}", verbose)
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def get_provider_url(provider: str, z: int, x: int, y: int) -> str:
    return PROVIDERS[provider].format(z=z, x=x, y=y)

def stitch_tiles(lat: float, lon: float, zoom: int, radius: int, provider: str, verbose: bool):
    t0 = time.time()
    log(f"Stitching tiles (provider={provider}, zoom={zoom}, radius={radius}) ...", verbose)

    center_px, center_py = deg2xy(lat, lon, zoom)
    tile_x0  = math.floor(center_px / TILE_SIZE) - radius
    tile_y0  = math.floor(center_py / TILE_SIZE) - radius

    grid = 2 * radius + 1
    out_w = TILE_SIZE * grid
    out_h = TILE_SIZE * grid

    tlx = tile_x0 * TILE_SIZE
    tly = tile_y0 * TILE_SIZE
    brx = tlx + out_w
    bry = tly + out_h

    log(f"Canvas: {out_w}x{out_h}  Global px TL=({tlx:.2f},{tly:.2f}) BR=({brx:.2f},{bry:.2f})", verbose)

    canvas = Image.new("RGB", (out_w, out_h), (0,0,0))
    with requests.Session() as session:
        session.headers.update({"User-Agent": UA})
        max_tile = (1 << zoom)
        for j in range(grid):
            for i in range(grid):
                tx = (tile_x0 + i) % max_tile
                ty = min(max(tile_y0 + j, 0), max_tile - 1)
                url = get_provider_url(provider, zoom, tx, ty)
                px = i * TILE_SIZE
                py = j * TILE_SIZE
                try:
                    canvas.paste(fetch_tile(session, url, verbose), (px, py))
                except Exception as e:
                    log(f"Tile failed ({url}): {e} -> grey fallback", verbose)
                    canvas.paste(Image.new("RGB", (TILE_SIZE, TILE_SIZE), (40,40,40)), (px, py))

    img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    log(f"Tiles stitched in {time.time()-t0:.2f}s", verbose)
    return img, (tlx, tly), (brx, bry)

# ---------------- BBox ----------------
def bbox_from_pixels(top_left, bottom_right, zoom: int, verbose: bool):
    tlx, tly = top_left; brx, bry = bottom_right
    def xy2deg(x, y, zoom):
        n = math.pi - 2.0 * math.pi * (y / (TILE_SIZE * (2 ** zoom)))
        lon = (x / (TILE_SIZE * (2 ** zoom)) - 0.5) * 360.0
        lat = math.degrees(math.atan(math.sinh(n)))
        return lat, lon
    n_lat, w_lon = xy2deg(tlx, tly, zoom)
    s_lat, e_lon = xy2deg(brx, bry, zoom)
    south, north = min(s_lat, n_lat), max(s_lat, n_lat)
    west, east   = min(w_lon, e_lon), max(w_lon, e_lon)
    log(f"BBox SWNE: ({south:.6f},{west:.6f},{north:.6f},{east:.6f})", verbose)
    return (south, west, north, east)

# ---------------- Overpass ----------------
def overpass_query(bbox, blocks, verbose: bool):
    s, w, n, e = bbox
    q = "[out:json][timeout:60];(" + "".join(OSM_QUERIES[b].format(s=s,w=w,n=n,e=e) for b in blocks) + ");out body;>;out skel qt;"
    log("POST Overpass...", verbose); t0 = time.time()
    r = requests.post(OVERPASS_URL, data={"data": q}, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    data = r.json()
    log(f"Overpass elements: {len(data.get('elements', []))} in {time.time()-t0:.2f}s", verbose)
    return data

def build_node_lookup(elements): return {el["id"]: el for el in elements if el["type"]=="node"}

def way_coords(way, node_lu):
    coords = []
    for nid in way.get("nodes", []):
        nd = node_lu.get(nid)
        if nd: coords.append((nd["lat"], nd["lon"]))
    return coords

def latlon_to_local_px(lat, lon, zoom, tlx, tly):
    x, y = deg2xy(lat, lon, zoom)
    return int(round(x - tlx)), int(round(y - tly))

def extract_feature_pixel_lines(elements, zoom, tlx, tly):
    node_lu = build_node_lookup(elements)
    water_lines, wetland_lines = [], []
    for el in elements:
        if el["type"] != "way": continue
        tags = el.get("tags", {})
        coords = way_coords(el, node_lu)
        if not coords: continue
        pts = np.array([latlon_to_local_px(la,lo,zoom,tlx,tly) for la,lo in coords], np.int32)
        if tags.get("waterway") in {"river","stream","canal","drain"} and len(pts)>=2:
            water_lines.append(pts)
        if tags.get("natural") == "wetland" and len(pts)>=2:
            wetland_lines.append(pts)
    return water_lines, wetland_lines

# ---------------- Elevation ----------------
def build_sample_grid(bbox, n, verbose: bool):
    s, w, n_, e = bbox
    lats = np.linspace(s, n_, n)
    lons = np.linspace(w, e, n)
    pts = [(float(la), float(lo)) for la in lats for lo in lons]
    log(f"Elevation grid: {n}x{n} -> {len(pts)} points", verbose)
    return pts, lats, lons

def opentopo_sample(points, chunk=80, max_retries=4, backoff_base=0.8, sleep_ms=180, verbose: bool=True):
    out = []; t0 = time.time()
    log(f"SRTM90m sampling: chunk={chunk}, backoff, sleep={sleep_ms}ms", verbose)
    for i in range(0, len(points), chunk):
        chunk_pts = points[i:i+chunk]
        locs = "|".join([f"{la},{lo}" for la,lo in chunk_pts])
        url = f"{OPENTOPO_URL}?locations={locs}"
        tries = 0
        while True:
            try:
                log(f"[{i:04d}-{i+len(chunk_pts)-1:04d}] GET SRTM...", verbose)
                r = requests.get(url, timeout=120, headers={"User-Agent": UA})
                if r.status_code == 429:
                    raise requests.HTTPError("429 Too Many Requests")
                r.raise_for_status()
                data = r.json()
                for j, res in enumerate(data.get("results", [])):
                    la, lo = chunk_pts[j]; elev = res.get("elevation")
                    if elev is not None: out.append((la, lo, float(elev)))
                break
            except Exception as e:
                tries += 1
                if tries > max_retries:
                    log(f"   chunk failed permanently after {max_retries} retries: {e}", verbose)
                    break
                delay = (backoff_base ** tries) + (sleep_ms/1000.0)
                log(f"   got {e}; retry {tries}/{max_retries} after {delay:.2f}s", verbose)
                time.sleep(delay)
        time.sleep(sleep_ms/1000.0)
    log(f"SRTM: kept {len(out)} pts in {time.time()-t0:.2f}s", verbose)
    return out

# ---------------- Rain ----------------
def openmeteo_precip(lat, lon, verbose: bool):
    params = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum", "timezone": "auto"}
    log("GET Open-Meteo daily precipitation...", verbose); t0 = time.time()
    r = requests.get(OPENMETEO_URL, params=params, timeout=60, headers={"User-Agent": UA})
    r.raise_for_status()
    data = r.json()
    log(f"Open-Meteo days: {len(data.get('daily',{}).get('precipitation_sum',[]) or [])} in {time.time()-t0:.2f}s", verbose)
    return data

# ---------------- Risk & raster ----------------
def lines_to_mask(shape, lines, thickness=3):
    mask = np.zeros(shape[:2], np.uint8)
    for pts in lines:
        if len(pts) >= 2:
            cv2.polylines(mask, [pts.reshape(-1,1,2)], False, 255, thickness=thickness)
    return mask

def distance_field(mask):
    inv = cv2.bitwise_not(mask)
    _, binv = cv2.threshold(inv, 254, 255, cv2.THRESH_BINARY)
    return cv2.distanceTransform(binv, cv2.DIST_L2, 3)

def safe_fill_nearest(elev_grid: np.ndarray, verbose: bool):
    if not np.isnan(elev_grid).any(): return elev_grid
    g = elev_grid.copy()
    for _ in range(10):
        nanmask = np.isnan(g)
        if not nanmask.any(): break
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            candidate = np.roll(g, shift=(dy,dx), axis=(0,1))
            cand_ok = ~np.isnan(candidate)
            replace = nanmask & cand_ok
            g[replace] = candidate[replace]
    if np.isnan(g).any():
        med = np.nanmedian(elev_grid)
        g[np.isnan(g)] = med if not np.isnan(med) else 0.0
    return g

def build_risk_grid(lat0, zoom, tlx, tly, img_w, img_h,
                    bbox, grid_n, elev_pts, lats, lons,
                    water_lines, wetland_lines, rain_json,
                    weights, thresholds, verbose: bool):
    log("Build risk grid...", verbose); t0 = time.time()

    if water_lines or wetland_lines:
        df_water = distance_field(lines_to_mask((img_h, img_w, 3), water_lines, thickness=3))
        df_wet   = distance_field(lines_to_mask((img_h, img_w, 3), wetland_lines, thickness=3))
    else:
        df_water = np.full((img_h, img_w), 1e6, np.float32)
        df_wet   = np.full((img_h, img_w), 1e6, np.float32)

    mpp = meters_per_pixel(lat0, zoom)
    log(f"m/px ~ {mpp:.2f}", verbose)

    elev_grid = np.full((grid_n, grid_n), np.nan, np.float32)
    for (la, lo, z) in elev_pts:
        iy = int(np.clip(np.searchsorted(lats, la), 0, grid_n-1))
        ix = int(np.clip(np.searchsorted(lons, lo), 0, grid_n-1))
        elev_grid[iy, ix] = z
    elev_grid = safe_fill_nearest(elev_grid, verbose)

    totals = (rain_json.get("daily", {}) or {}).get("precipitation_sum", []) or []
    rain3 = float(sum(totals[:3])) if totals else 0.0
    rain_score = max(0.0, min(1.0, rain3 / thresholds["rain_cap_mm"]))

    e_low, e_high = thresholds["elev_low"], thresholds["elev_high"]
    D0, D1 = thresholds["D0_water"], thresholds["D1_wetland"]

    risk = np.zeros((grid_n, grid_n), np.float32)
    for iy, la in enumerate(lats):
        for ix, lo in enumerate(lons):
            gx, gy = deg2xy(la, lo, zoom)
            px = int(np.clip(round(gx - tlx), 0, img_w-1))
            py = int(np.clip(round(gy - tly), 0, img_h-1))

            z = float(elev_grid[iy, ix])
            elev_score = 1.0 if z <= e_low else (max(0.0, (e_high - z) / (e_high - e_low)) if z <= e_high else 0.0)

            d_water_m = float(df_water[py, px]) * mpp
            d_wet_m   = float(df_wet[py, px]) * mpp
            dist_water_score = math.exp(-d_water_m / D0)
            wetland_score    = math.exp(-d_wet_m / D1)

            r = (weights["elev"] * elev_score +
                 weights["dist"] * dist_water_score +
                 weights["wet"]  * wetland_score +
                 weights["rain"] * rain_score)
            risk[iy, ix] = max(0.0, min(1.0, r))

    log(f"Risk grid computed in {time.time()-t0:.2f}s", verbose)
    return risk

def colorize_risk01(risk01: np.ndarray) -> np.ndarray:
    r = np.clip(risk01, 0, 1).astype(np.float32)
    B = np.zeros_like(r); G = np.zeros_like(r); R = np.zeros_like(r)
    mask_low = (r <= 0.5)
    t1 = np.zeros_like(r); t1[mask_low] = r[mask_low] / 0.5
    B[mask_low] = 0.0; G[mask_low] = 255.0; R[mask_low] = t1[mask_low] * 255.0
    mask_high = ~mask_low
    t2 = np.zeros_like(r); t2[mask_high] = (r[mask_high] - 0.5) / 0.5
    B[mask_high] = 0.0; G[mask_high] = (1.0 - t2[mask_high]) * 255.0; R[mask_high] = 255.0
    out = np.zeros((r.shape[0], r.shape[1], 3), np.uint8)
    out[..., 0] = np.clip(B, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(G, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(R, 0, 255).astype(np.uint8)
    return out

def blend_overlay(base_bgr, overlay_bgr, alpha: float):
    return cv2.addWeighted(overlay_bgr, alpha, base_bgr, 1.0-alpha, 0)

# ---------- Water/ocean masking (basemap-derived) ----------
def detect_water_mask_bgr(bgr_img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    B, G, R = bgr_img[...,0], bgr_img[...,1], bgr_img[...,2]
    hsv_water = ((H >= 90) & (H <= 150) & (S >= 60) & (V >= 50))
    bgr_water = (B > (G + 10)) & (B > (R + 10)) & (B > 100)
    water = (hsv_water | bgr_water).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=2)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN,  k, iterations=1)
    return water

def masked_alpha_blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float, keep_mask_uint8: np.ndarray) -> np.ndarray:
    keep = (keep_mask_uint8 == 255).astype(np.float32)
    keep3 = np.dstack([keep, keep, keep])
    return (overlay_bgr.astype(np.float32) * (alpha * keep3) +
            base_bgr.astype(np.float32) * (1.0 - alpha * keep3)).astype(np.uint8)

# ---------------- Drawing (vectors on top) ----------------
def draw_vectors(img, elements, zoom, tlx, tly, verbose: bool,
                 risk_img: np.ndarray = None, bld_high_thr: float = 0.6,
                 dot_radius: int = 2, dot_min_sep: int = 6):
    t0 = time.time()
    node_lu = build_node_lookup(elements)
    n_water = n_wet = n_emb = n_bld = 0

    H, W = img.shape[:2]
    EMBANK_COLOR = (100, 0, 0)     # dark blue (BGR)
    BLD_GRAY      = (120, 120, 120)
    BLD_EDGE_DARK = (40, 40, 40)
    BLD_HIGHRED   = (0, 0, 255)

    placed_dots: List[Tuple[int,int]] = []
    dot_radius = max(1, int(dot_radius))
    dot_min_sep = max(dot_radius * 2 + 1, int(dot_min_sep))
    dot_min_sep2 = dot_min_sep * dot_min_sep

    def glow_line(pts, color, thick_core=2, thick_glow=5, glow_alpha=0.35):
        if len(pts) < 2: return
        pts_np = np.array(pts, np.int32)
        overlay = img.copy()
        cv2.polylines(overlay, [pts_np], False, color, thick_glow, cv2.LINE_AA)
        cv2.addWeighted(overlay, glow_alpha, img, 1-glow_alpha, 0, img)
        cv2.polylines(img, [pts_np], False, color, thick_core, cv2.LINE_AA)

    for el in elements:
        if el["type"]=="way" and el.get("tags",{}).get("waterway") in {"river","stream","canal","drain"}:
            pts = [latlon_to_local_px(la,lo,zoom,tlx,tly) for la,lo in way_coords(el, node_lu)]
            if len(pts)>=2:
                glow_line(pts, (0,200,255), thick_core=3, thick_glow=7, glow_alpha=0.25)
                n_water += 1

    for el in elements:
        if el["type"]=="way" and el.get("tags",{}).get("natural")=="wetland":
            pts = [latlon_to_local_px(la,lo,zoom,tlx,tly) for la,lo in way_coords(el, node_lu)]
            if len(pts)>=3 and pts[0]==pts[-1]:
                overlay = img.copy()
                cv2.fillPoly(overlay, [np.array(pts, np.int32)], (200,160,120))
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [np.array(pts, np.int32)], True, (80,60,40), 2, cv2.LINE_AA)
                n_wet += 1
            elif len(pts)>=2:
                glow_line(pts, (180,150,120), thick_core=2, thick_glow=5, glow_alpha=0.25)
                n_wet += 1

    for el in elements:
        if el["type"]=="way" and el.get("tags",{}).get("man_made") in {"embankment","dyke","dike"}:
            pts = [latlon_to_local_px(la,lo,zoom,tlx,tly) for la,lo in way_coords(el, node_lu)]
            if len(pts)>=2:
                glow_line(pts, EMBANK_COLOR, thick_core=3, thick_glow=7, glow_alpha=0.25)
                n_emb += 1

    has_risk = isinstance(risk_img, np.ndarray) and risk_img.shape[:2] == (H, W)
    for el in elements:
        if el["type"]=="way" and "building" in el.get("tags",{}):
            coords = way_coords(el, node_lu)
            if len(coords)>=3:
                pts = [latlon_to_local_px(la,lo,zoom,tlx,tly) for la,lo in coords]
                if pts[0]==pts[-1]:
                    poly = np.array(pts, np.int32)
                    fill_color = BLD_GRAY
                    edge_color = BLD_EDGE_DARK
                    if has_risk:
                        mask = np.zeros((H, W), np.uint8)
                        cv2.fillPoly(mask, [poly], 255)
                        vals = risk_img[mask == 255]
                        if vals.size > 0 and float(np.nanmean(vals)) >= bld_high_thr:
                            fill_color = BLD_HIGHRED
                            edge_color = BLD_HIGHRED
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [poly], fill_color)
                    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)
                    cv2.polylines(img, [poly], True, edge_color, 1, cv2.LINE_AA)
                    # Optional non-overlapping centroid dots (already handled by polygon color)
                    # You can re-enable the red centroid dots by computing centroid and spacing, if desired.
                    n_bld += 1

    log(f"Vectors drawn in {time.time()-t0:.2f}s -> waterways:{n_water} wetlands:{n_wet} embankments:{n_emb} buildings:{n_bld}", verbose)

# ---------------- UI Panels ----------------
def draw_info_panel(img, lat, lon, daily_json, verbose: bool):
    overlay = img.copy()
    panel_w, panel_h = 360, 164
    x0, y0 = 20, 20
    panel_w = min(panel_w, img.shape[1]-40)
    panel_h = min(panel_h, img.shape[0]-40)
    cv2.rectangle(overlay, (x0+3, y0+3), (x0+panel_w+3, y0+panel_h+3), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (x0, y0), (x0+panel_w, y0+panel_h), (255,255,255), -1)
    cv2.addWeighted(overlay2, 0.92, img, 0.08, 0, img)
    cv2.rectangle(img, (x0, y0), (x0+panel_w, y0+panel_h), (60,60,60), 1)
    cv2.putText(img, "Rain (Open-Meteo)", (x0+10, y0+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 2, cv2.LINE_AA)
    cv2.putText(img, f"Loc: {lat:.5f}, {lon:.5f}", (x0+10, y0+44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
    daily = daily_json.get("daily", {})
    totals = daily.get("precipitation_sum", []) or []
    days   = daily.get("time", []) or []
    next3 = round(sum(totals[:3]), 1) if totals else None
    cv2.putText(img, f"Next 3 days: {next3} mm" if next3 is not None else "Next 3 days: n/a",
                (x0+10, y0+66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    y = y0 + 90
    for d, v in list(zip(days, totals))[:5]:
        cv2.putText(img, f"{d}: {round(v,1)} mm", (x0+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 1, cv2.LINE_AA)
        y += 18
    log("Rain panel drawn.", verbose)

def draw_legend(img, verbose: bool, dot_radius: int = 2):
    labels = ["Low", "Moderate", "High", "Very High", "Extreme"]
    edges  = (0.0,0.2,0.4,0.6,0.8,1.0)
    sw_w, sw_h = 42, 18
    line_gap   = 24
    header_h   = 26
    footer_h   = 18
    sections_h = len(labels) * line_gap
    extras_h   = 5 * 18 + 16
    padding    = 12
    panel_w = 400
    panel_h = header_h + sections_h + extras_h + footer_h + 2*padding
    panel_w = min(panel_w, img.shape[1] - 40)
    panel_h = min(panel_h, img.shape[0] - 40)
    x0 = 18
    y0 = img.shape[0] - panel_h - 18
    if y0 < 18: y0 = 18
    base = img.copy()
    cv2.rectangle(base, (x0,y0), (x0+panel_w,y0+panel_h), (255,255,255), -1)
    img[:] = cv2.addWeighted(base, 0.60, img, 0.40, 0)
    cv2.rectangle(img, (x0,y0), (x0+panel_w,y0+panel_h), (60,60,60), 1)
    cv2.putText(img, "Flood Risk Microzonation", (x0+10,y0+22), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20,20,20), 2, cv2.LINE_AA)
    sx = x0 + 12
    sy = y0 + 40
    for i in range(len(labels)):
        vmid = (edges[i]+edges[i+1])/2.0
        color = colorize_risk01(np.array([[vmid]], np.float32))[0,0,:].tolist()
        color = tuple(int(c) for c in color)
        top = sy + i*line_gap
        cv2.rectangle(img, (sx, top), (sx+sw_w, top+sw_h), color, -1)
        cv2.rectangle(img, (sx, top), (sx+sw_w, top+sw_h), (60,60,60), 1)
        cv2.putText(img, f"{labels[i]} [{edges[i]:.1f}-{edges[i+1]:.1f}]",
                    (sx+sw_w+10, top+sw_h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30,30,30), 2)
    y2 = sy + len(labels)*line_gap + 6
    cv2.putText(img, "Other layers:", (sx, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30,30,30), 2)
    cv2.line(img, (sx, y2+16), (sx+40, y2+16), (0,200,255), 3)
    cv2.putText(img, "Waterway", (sx+48,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.line(img, (sx, y2+34), (sx+40, y2+34), (100,0,0), 3)
    cv2.putText(img, "Embankment/Levee", (sx+48,y2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.rectangle(img, (sx, y2+46), (sx+40, y2+62), (200,160,120), -1)
    cv2.putText(img, "Wetland", (sx+48,y2+58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.rectangle(img, (sx, y2+66), (sx+40, y2+82), (120,120,120), -1)
    cv2.putText(img, "Buildings", (sx+48,y2+78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.rectangle(img, (sx, y2+86), (sx+40, y2+102), (0,0,255), -1)
    cv2.circle(img, (sx+20, y2+94), max(1, int(dot_radius)), (0,0,255), -1, cv2.LINE_AA)
    cv2.putText(img, "High-risk buildings (red + dot)", (sx+48,y2+98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.putText(img, "Data: ©OSM, OpenTopodata (SRTM), Open-Meteo", (x0+10, y0+panel_h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (70,70,70), 1)
    log("Legend drawn.", verbose)

# ---------------- Dynamic sizing ----------------
def dynamic_scale_for_zoom(zoom: int, radius: int, base_scale: float, per_zoom: float, max_px: int, w: int, h: int):
    scale = base_scale * (1.0 + max(0, (zoom - 11)) * per_zoom)
    scale *= max(1.0, 1.0 + 0.1 * max(0, radius - 1))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if max(new_w, new_h) > max_px:
        k = max_px / float(max(new_w, new_h))
        new_w = int(round(new_w * k))
        new_h = int(round(new_h * k))
    return max(1, new_w), max(1, new_h)

# ---------------- Comparison helpers ----------------
def save_comparison_images(current_img_bgr: np.ndarray,
                           current_risk_float: Optional[np.ndarray],
                           snapshot_dir: str,
                           compare_to_dir: Optional[str],
                           zoom: int, tlx: int, tly: int, w: int, h: int,
                           verbose: bool):
    if not compare_to_dir:
        return
    old_grid = load_risk_grid(compare_to_dir)
    if old_grid is None:
        log(f"--compare_to provided but risk_grid.npy not found in {compare_to_dir}", verbose)
        return
    # resize BOTH to the same (w,h)
    try:
        cur = None
        if current_risk_float is not None:
            cur = cv2.resize(current_risk_float, (w, h), interpolation=cv2.INTER_CUBIC)
        old = cv2.resize(old_grid, (w, h), interpolation=cv2.INTER_CUBIC)

        # side-by-side (old vs current)
        old_col = colorize_risk01(old)
        if cur is None:
            # If current risk missing, just duplicate old
            cur_col = old_col.copy()
        else:
            cur_col = colorize_risk01(cur)
        side = np.concatenate([old_col, cur_col], axis=1)
        side_path = os.path.join(snapshot_dir, "comparison_side_by_side.png")
        cv2.imwrite(side_path, side)
        log(f"Saved side-by-side comparison -> {side_path}", verbose)

        # difference (current - old) signed; visualize with diverging ramp
        if cur is not None:
            diff = np.clip(cur - old, -1.0, 1.0)
            # map -1..0..+1 to blue->white->red
            diff_vis = np.zeros((h, w, 3), np.uint8)
            neg = diff < 0
            pos = diff > 0
            z = ~neg & ~pos
            # magnitude
            mag = np.abs(diff)
            # blue channel for negatives
            diff_vis[..., 0][neg] = (mag[neg]*255).astype(np.uint8)
            # red channel for positives
            diff_vis[..., 2][pos] = (mag[pos]*255).astype(np.uint8)
            # white for zero-ish (already zeros; set light grey)
            diff_vis[..., 0][z] = 200; diff_vis[..., 1][z] = 200; diff_vis[..., 2][z] = 200
            diff_path = os.path.join(snapshot_dir, "risk_diff.png")
            cv2.imwrite(diff_path, diff_vis)
            log(f"Saved risk difference image -> {diff_path}", verbose)
    except Exception as e:
        log(f"Comparison rendering error: {e}", verbose)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="OpenCV flood-context overlays with microzonation on OSM/ESRI tiles.")
    # Absolute center & zoom
    ap.add_argument("--lat", type=float, default=7.0647, help="Center latitude")
    ap.add_argument("--lon", type=float, default=125.4762, help="Center longitude")
    ap.add_argument("--zoom", type=int, default=12, help="Zoom level")

    # Relative tweaks
    ap.add_argument("--dlat", type=float, default=0.0, help="Latitude delta to add to --lat")
    ap.add_argument("--dlon", type=float, default=0.0, help="Longitude delta to add to --lon")
    ap.add_argument("--dz",   type=int,   default=0,   help="Zoom delta to add to --zoom")

    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--provider", type=str, default="osm", choices=list(PROVIDERS.keys()))
    ap.add_argument("--grid", type=int, default=36)
    ap.add_argument("--output", type=str, default="flood_risk.png")
    ap.add_argument("--alpha", type=float, default=0.38)
    ap.add_argument("--verbose", action="store_true", default=True)

    # weights
    ap.add_argument("--w_elev", type=float, default=0.5)
    ap.add_argument("--w_dist", type=float, default=0.3)
    ap.add_argument("--w_wet",  type=float, default=0.1)
    ap.add_argument("--w_rain", type=float, default=0.1)

    # thresholds
    ap.add_argument("--elev_low",  type=float, default=0.0)
    ap.add_argument("--elev_high", type=float, default=20.0)
    ap.add_argument("--D0_water",  type=float, default=300.0)
    ap.add_argument("--D1_wet",    type=float, default=200.0)
    ap.add_argument("--rain_cap",  type=float, default=100.0)
    ap.add_argument("--bld_high_thr", type=float, default=0.6, help="Risk threshold for marking buildings red (0..1)")

    # dot controls
    ap.add_argument("--dot_radius", type=int, default=2, help="Radius of red dot markers (px)")
    ap.add_argument("--dot_min_sep", type=int, default=6, help="Minimum separation (px) between red dots")

    # toggles
    ap.add_argument("--draw_vectors", action="store_true", default=True)
    ap.add_argument("--draw_risk", action="store_true", default=True)
    ap.add_argument("--sleep_ms", type=int, default=180, help="per-chunk sleep for SRTM (ms)")

    # dynamic output sizing
    ap.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic output scaling by zoom/radius")
    ap.add_argument("--max_px", type=int, default=3600, help="Clamp the larger dimension to this many pixels")
    ap.add_argument("--base_scale", type=float, default=1.0, help="Base scale multiplier")
    ap.add_argument("--scale_per_zoom", type=float, default=0.35, help="Extra scale per zoom level over 11")

    # Snapshot / cache controls
    ap.add_argument("--snapshot_dir", type=str, default="", help="Use this folder for all inputs/outputs (replaces auto cache folder).")
    ap.add_argument("--snapshot_mode", type=str, choices=["auto","read","write"], default="auto",
                    help="auto: reuse files or fetch if missing; read: strict offline; write: fetch fresh to this snapshot_dir")
    ap.add_argument("--cache_base", type=str, default="cache", help="Base cache directory (used if --snapshot_dir is empty).")
    ap.add_argument("--no_cache", action="store_true", help="Bypass and ignore cache (always refetch) when not in snapshot read mode.")
    ap.add_argument("--compare_to", type=str, default="", help="Path to another snapshot_dir to compare risk against (produces diff images).")

    args = ap.parse_args()

    # center and zoom
    lat = args.lat + args.dlat
    lon = args.lon + args.dlon
    zoom = args.zoom + args.dz

    # Decide snapshot directory
    if args.snapshot_dir:
        snapshot_dir = args.snapshot_dir
    else:
        # fallback to old cache layout keyed by zoom/lat/lon
        key = f"zoom{zoom}_lat{lat:.5f}_lon{lon:.5f}"
        snapshot_dir = os.path.join(args.cache_base, key)
    ensure_dir(snapshot_dir)

    log("=== Flood-context renderer (OpenCV microzonation) START ===", args.verbose)
    log(f"Params: lat={lat}, lon={lon}, zoom={zoom}, radius={args.radius}, provider={args.provider}", args.verbose)
    log(f"Grid={args.grid}x{args.grid}, alpha={args.alpha}, weights: elev={args.w_elev} dist={args.w_dist} wet={args.w_wet} rain={args.w_rain}", args.verbose)
    log(f"Snapshot folder: {snapshot_dir}  (mode={args.snapshot_mode})", args.verbose)

    # ---------- Base map ----------
    basemap_path = os.path.join(snapshot_dir, f"basemap_{args.provider}_r{args.radius}.png")
    if args.snapshot_mode in ("auto","read"):
        img = load_img_if_exists(basemap_path)
    else:
        img = None

    if img is None:
        if args.snapshot_mode == "read":
            raise FileNotFoundError(f"snapshot_mode=read but missing basemap: {basemap_path}")
        img, (tlx,tly), (brx,bry) = stitch_tiles(lat, lon, zoom, args.radius, args.provider, args.verbose)
        cv2.imwrite(basemap_path, img)
        log(f"Saved basemap to snapshot: {basemap_path}", args.verbose)
    else:
        # Reconstruct tlx/tly from parameters (tile-aligned)
        grid = 2 * args.radius + 1
        w_re, h_re = TILE_SIZE * grid, TILE_SIZE * grid
        center_px, center_py = deg2xy(lat, lon, zoom)
        tile_x0  = math.floor(center_px / TILE_SIZE) - args.radius
        tile_y0  = math.floor(center_py / TILE_SIZE) - args.radius
        tlx = tile_x0 * TILE_SIZE
        tly = tile_y0 * TILE_SIZE
        brx = tlx + w_re
        bry = tly + h_re
        log(f"Loaded basemap from snapshot (tile-aligned TL=({tlx},{tly}))", args.verbose)

    h, w = img.shape[:2]
    bbox = bbox_from_pixels((tlx,tly), (brx,bry), zoom, args.verbose)

    # ---------- Overpass ----------
    overpass_path = os.path.join(snapshot_dir, "overpass.json")
    over = None
    if args.snapshot_mode in ("auto","read"):
        over = try_read_json(overpass_path)

    elements = (over or {}).get("elements", [])
    if not elements:
        if args.snapshot_mode == "read":
            raise FileNotFoundError(f"snapshot_mode=read but missing Overpass data: {overpass_path}")
        try:
            over = overpass_query(bbox, ["waterways","wetlands","embank","buildings"], args.verbose)
            elements = over.get("elements", [])
            write_json(overpass_path, over)
            log(f"Overpass fetch cached to {overpass_path}", args.verbose)
        except Exception as e:
            log(f"Overpass error: {e}", args.verbose)

    # ---------- Elevation ----------
    elev_path = os.path.join(snapshot_dir, f"elev_pts_grid{args.grid}.json")
    pts, lats, lons = build_sample_grid(bbox, args.grid, args.verbose)
    elev_pts = []
    if args.snapshot_mode in ("auto","read") and os.path.exists(elev_path):
        elev_pts = must_read_json(elev_path)

    if not elev_pts:
        if args.snapshot_mode == "read":
            raise FileNotFoundError(f"snapshot_mode=read but missing elevation samples: {elev_path}")
        elev_pts = opentopo_sample(pts, chunk=80, max_retries=4, backoff_base=0.9,
                                   sleep_ms=args.sleep_ms, verbose=args.verbose)
        write_json(elev_path, elev_pts)
        log(f"Saved elevation samples to snapshot: {elev_path}", args.verbose)

    # ---------- Rain ----------
    rain_path = os.path.join(snapshot_dir, "rain.json")
    rain_json = {}
    if args.snapshot_mode in ("auto","read") and os.path.exists(rain_path):
        rain_json = must_read_json(rain_path)

    if not rain_json:
        if args.snapshot_mode == "read":
            raise FileNotFoundError(f"snapshot_mode=read but missing rain data: {rain_path}")
        rain_json = openmeteo_precip(lat, lon, args.verbose)
        write_json(rain_path, rain_json)
        log(f"Saved Open-Meteo to snapshot: {rain_path}", args.verbose)

    # ---------- Ocean/Land mask ----------
    water_mask = detect_water_mask_bgr(img)
    land_mask  = cv2.bitwise_not(water_mask)
    log(f"Water mask: {(water_mask>0).sum()} px marked as water.", args.verbose)

    # ---------- Risk overlay ----------
    risk_img_float = None
    risk_grid = None
    if args.draw_risk and len(elev_pts)>0:
        try:
            water_lines, wetland_lines = extract_feature_pixel_lines(elements, zoom, tlx, tly)
            weights   = {"elev": args.w_elev, "dist": args.w_dist, "wet": args.w_wet, "rain": args.w_rain}
            thrsh     = {"elev_low": args.elev_low, "elev_high": args.elev_high,
                         "D0_water": args.D0_water, "D1_wetland": args.D1_wet, "rain_cap_mm": args.rain_cap}
            risk_grid = build_risk_grid(lat, zoom, tlx, tly, w, h, bbox,
                                        args.grid, elev_pts, lats, lons,
                                        water_lines, wetland_lines, rain_json, weights, thrsh, args.verbose)
            # Save raw grid for future replay/compare
            save_risk_grid(snapshot_dir, risk_grid)

            risk_img_float = cv2.resize(risk_grid, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            color_risk = colorize_risk01(risk_img_float)
            img = masked_alpha_blend(img, color_risk, alpha=args.alpha, keep_mask_uint8=land_mask)
            log("Risk overlay blended over land only.", args.verbose)
        except Exception as e:
            log(f"Risk overlay error: {e}", args.verbose)

    # ---------- Vectors ----------
    if args.draw_vectors and elements:
        try:
            draw_vectors(
                img, elements, zoom, tlx, tly, args.verbose,
                risk_img=risk_img_float,
                bld_high_thr=args.bld_high_thr,
                dot_radius=args.dot_radius,
                dot_min_sep=args.dot_min_sep
            )
        except Exception as e:
            log(f"Vector draw error: {e}", args.verbose)

    # Crosshair
    cx, cy = w//2, h//2
    cv2.drawMarker(img, (cx,cy), (0,0,0), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.drawMarker(img, (cx,cy), (255,255,255), cv2.MARKER_TILTED_CROSS, 16, 1)

    # Legend + Info panel
    draw_legend(img, args.verbose, dot_radius=args.dot_radius)
    try:
        draw_info_panel(img, lat, lon, rain_json, args.verbose)
    except Exception as e:
        log(f"Info panel error: {e}", args.verbose)

    # Dynamic final size
    save_img = img
    if args.dynamic:
        new_w, new_h = dynamic_scale_for_zoom(
            zoom, args.radius, args.base_scale, args.scale_per_zoom, args.max_px, w, h
        )
        if (new_w, new_h) != (w, h):
            log(f"Resizing final image from {w}x{h} -> {new_w}x{new_h} (zoom={zoom}, radius={args.radius})", args.verbose)
            save_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    out_path = os.path.join(snapshot_dir, args.output) if os.path.isdir(snapshot_dir) else args.output
    log(f"Saving -> {out_path}", args.verbose)
    ok = cv2.imwrite(out_path, save_img)
    log(f"Saved={ok}. File: {out_path}", args.verbose)

    # ----- Optional comparison with another snapshot -----
    compare_to = args.compare_to.strip()
    if compare_to:
        save_comparison_images(save_img, risk_img_float, snapshot_dir, compare_to,
                               zoom, tlx, tly, w, h, args.verbose)

    log("=== DONE ===", args.verbose)

if __name__ == "__main__":
    main()
