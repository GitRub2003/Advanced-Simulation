import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from roads_cleaner2 import INFRA, BMMS, repaired_long

OFF_ROAD_THRESHOLD_KM = 3.0
CHAINAGE_EXACT_TOL_KM = 0.05


def normalize_key(val):
    return str(val).strip().upper()


def plausible_bd_coord(lat, lon):
    """Very lightweight sanity check for Bangladesh-ish bounds."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    return (20.0 <= float(lat) <= 27.5) and (88.0 <= float(lon) <= 93.0)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = radians(float(lat1))
    phi2 = radians(float(lat2))
    dphi = radians(float(lat2) - float(lat1))
    dlambda = radians(float(lon2) - float(lon1))
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def nearest_point_distance_km(lat, lon, info):
    """Distance from a bridge coordinate to nearest known road LRP vertex."""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    dists = [haversine(lat, lon, la, lo) for la, lo in zip(info["lat"], info["lon"])]
    return min(dists) if dists else np.nan


def build_road_index(roads_long_df):
    """
    Precompute per-road arrays sorted by chainage for fast lookup + interpolation.
    Requires columns: road, lrp, lat, lon, chainage.
    """
    idx = {}
    seq_col = "seq" if "seq" in roads_long_df.columns else None
    sort_cols = ["road", "chainage"] + ([seq_col] if seq_col else [])

    df = roads_long_df.dropna(subset=["road", "lat", "lon", "chainage"]).copy()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df = df.dropna(subset=["chainage"])
    df = df.sort_values(sort_cols)

    for road, g in df.groupby("road", sort=False):
        road_key = normalize_key(road)
        lat_arr = g["lat"].to_numpy(float)
        lon_arr = g["lon"].to_numpy(float)
        ch_arr = g["chainage"].to_numpy(float)
        lrp_arr = g["lrp"].astype(str).str.strip().str.upper().to_numpy()

        idx[road_key] = {
            "chainage": ch_arr,
            "lat": lat_arr,
            "lon": lon_arr,
            "lrp": lrp_arr,
            "lrp_to_row": {lrp: (float(la), float(lo)) for lrp, la, lo in zip(lrp_arr, lat_arr, lon_arr)},
        }
    return idx


def estimate_bridge_latlon(row, road_idx, tol_chainage_km=CHAINAGE_EXACT_TOL_KM):
    """
    Returns (lat, lon, estimatedLoc) or (np.nan, np.nan, "error").
    Strategy priority:
    1) exact LRP match
    2) chainage exact/interpolated/clamped
    3) nearest road vertex from current coordinate (only if current point is plausible in BD)
    If none of the above are available, do not force a guess.
    """
    road = normalize_key(row.get("road", ""))
    if road not in road_idx:
        return np.nan, np.nan, "error_no_road"

    info = road_idx[road]
    lat_arr = info["lat"]
    lon_arr = info["lon"]
    ch_arr = info["chainage"]

    lrpname = normalize_key(row.get("LRPName", ""))
    if lrpname and lrpname in info["lrp_to_row"]:
        la, lo = info["lrp_to_row"][lrpname]
        return la, lo, "road_precise"

    ch = pd.to_numeric(row.get("chainage", np.nan), errors="coerce")
    if pd.notna(ch):
        ch = float(ch)
        diffs = np.abs(ch_arr - ch)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_chainage_km:
            return float(lat_arr[j]), float(lon_arr[j]), "road_chainage"

        if ch <= ch_arr[0]:
            return float(lat_arr[0]), float(lon_arr[0]), "road_chainage_clamped_start"
        if ch >= ch_arr[-1]:
            return float(lat_arr[-1]), float(lon_arr[-1]), "road_chainage_clamped_end"

        k = int(np.searchsorted(ch_arr, ch, side="left"))
        ch1, ch2 = ch_arr[k - 1], ch_arr[k]
        if ch2 == ch1:
            return float((lat_arr[k - 1] + lat_arr[k]) / 2), float((lon_arr[k - 1] + lon_arr[k]) / 2), "road_interpolate"

        t = (ch - ch1) / (ch2 - ch1)
        la = lat_arr[k - 1] + t * (lat_arr[k] - lat_arr[k - 1])
        lo = lon_arr[k - 1] + t * (lon_arr[k] - lon_arr[k - 1])
        return float(la), float(lo), "road_interpolate"

    # Nearest-vertex fallback intentionally disabled:
    # this can snap to a geometrically close but semantically wrong road point.
    # lat0 = row.get("lat", np.nan)
    # lon0 = row.get("lon", np.nan)
    # if plausible_bd_coord(lat0, lon0):
    #     dists = [haversine(lat0, lon0, la, lo) for la, lo in zip(lat_arr, lon_arr)]
    #     if dists:
    #         j = int(np.argmin(dists))
    #         return float(lat_arr[j]), float(lon_arr[j]), "road_nearest_vertex"

    return np.nan, np.nan, "error_no_logical_fix"


def bridge_is_broken(row, road_idx, offroad_threshold_km=OFF_ROAD_THRESHOLD_KM):
    lat = row.get("lat", np.nan)
    lon = row.get("lon", np.nan)
    road = normalize_key(row.get("road", ""))

    outside_bd = not plausible_bd_coord(lat, lon)
    if outside_bd:
        return True

    if road in road_idx:
        dist_km = nearest_point_distance_km(lat, lon, road_idx[road])
        if pd.notna(dist_km) and dist_km > offroad_threshold_km:
            return True
    return False


road_idx = build_road_index(repaired_long)
BMMS_fixed = BMMS.copy()

broken_mask = BMMS_fixed.apply(lambda r: bridge_is_broken(r, road_idx), axis=1)
print("Broken bridges detected:", int(broken_mask.sum()))

new_lat = pd.Series(index=BMMS_fixed.index, dtype="float64")
new_lon = pd.Series(index=BMMS_fixed.index, dtype="float64")
new_src = pd.Series(index=BMMS_fixed.index, dtype="object")

for row_idx, row in BMMS_fixed.loc[broken_mask].iterrows():
    la, lo, src = estimate_bridge_latlon(row, road_idx)
    new_lat.loc[row_idx] = la
    new_lon.loc[row_idx] = lo
    new_src.loc[row_idx] = src

fixable_mask = broken_mask & new_lat.notna() & new_lon.notna()
print("Fixable bridges (got new coords):", int(fixable_mask.sum()))
print("Unfixable bridges (still missing):", int((broken_mask & ~fixable_mask).sum()))

BMMS_fixed.loc[fixable_mask, "lat"] = new_lat.loc[fixable_mask].values
BMMS_fixed.loc[fixable_mask, "lon"] = new_lon.loc[fixable_mask].values
BMMS_fixed.loc[fixable_mask, "EstimatedLoc"] = new_src.loc[fixable_mask].values

still_broken_mask = BMMS_fixed.apply(lambda r: bridge_is_broken(r, road_idx), axis=1)
print("Still broken after repair:", int(still_broken_mask.sum()))

repaired_path = INFRA / "BMMS_overview.xlsx"
with pd.ExcelWriter(repaired_path, engine="openpyxl") as writer:
    BMMS_fixed.to_excel(writer, sheet_name="BMMS_overview", index=False)
print(f"Repaired bridges saved to: {repaired_path}")
