import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from statsmodels.nonparametric.smoothers_lowess import lowess

# =============================================================================
# CONFIGURATION – adjust these parameters as needed
# =============================================================================
K_MAD = 3
ABS_THRESHOLD_KM = 5
JUMP_RATIO = 1.5
JUMP_ABS_KM = 2.0
MAX_STEP_KM = 50.0
MAX_TRAILING_DROP= 0
MAX_LEADING_DROP =0


# -----------------------------------------------------------------------------
# Helper functions for road filtering
# -----------------------------------------------------------------------------
def split_road(road_str):
    s = str(road_str).strip()
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    prefix = s[:i] if i > 0 else ""
    num = None
    if i < len(s):
        try:
            num = int(s[i:])
        except Exception:
            num = None
    return prefix, num



# -----------------------------------------------------------------------------
# Geometric calculations
# -----------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def cumulative_haversine(lats, lons):
    if len(lats) == 0:
        return np.array([])
    steps = [0.0]
    for i in range(1, len(lats)):
        steps.append(haversine(lats[i-1], lons[i-1], lats[i], lons[i]))
    return np.cumsum(steps)

def step_metrics(lats, lons, chainage):
    n = len(lats)
    step_dist = np.zeros(n)
    chainage_step = np.zeros(n)
    step_ratio = np.full(n, np.nan)
    for i in range(1, n):
        step_dist[i] = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        chainage_step[i] = float(chainage[i] - chainage[i-1]) if chainage is not None else np.nan
        if chainage_step[i] > 0:
            step_ratio[i] = step_dist[i] / chainage_step[i]
    return step_dist, chainage_step, step_ratio

def walk_chainage_filter(lats, lons, chainage, jump_abs_km, jump_ratio):
    n = len(lats)
    walk_outlier = np.zeros(n, dtype=bool)
    last_good = 0
    for i in range(1, n):
        ch_delta = float(chainage[i] - chainage[last_good])
        if ch_delta <= 0:
            continue
        dist = haversine(lats[last_good], lons[last_good], lats[i], lons[i])
        if dist > jump_abs_km and dist > jump_ratio * ch_delta:
            walk_outlier[i] = True
        else:
            last_good = i
    return walk_outlier

# -----------------------------------------------------------------------------
# Coordinate scale handling (fixed‑point to degrees)
# -----------------------------------------------------------------------------
def infer_coord_scale(df):
    lat_cols = [c for c in df.columns if str(c).lower().startswith("lat")]
    lon_cols = [c for c in df.columns if str(c).lower().startswith("lon")]
    if not lat_cols and not lon_cols:
        return None
    vals = pd.to_numeric(df[lat_cols + lon_cols].stack(), errors="coerce").dropna()
    if vals.empty:
        return None
    median_abs = float(np.median(np.abs(vals)))
    if median_abs <= 180:
        return None
    for scale in (1e5, 1e6, 1e7):
        if (median_abs / scale) <= 90:
            return scale
    return None

def normalize_coord_columns(df, scale):
    if not scale:
        return df
    df = df.copy()
    lat_lon_cols = [c for c in df.columns if str(c).lower().startswith(("lat", "lon"))]
    for col in lat_lon_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.dropna().empty:
            continue
        df[col] = vals.where(np.abs(vals) <= 180, vals / scale)
    return df

# -----------------------------------------------------------------------------
# Data loading and conversion
# -----------------------------------------------------------------------------
try:
    BASE = Path(__file__).resolve().parents[1]
except NameError:
    BASE = Path.cwd()

# If using a notebook instead:
# BASE = Path().resolve().parents[0]

DATA = BASE / "data" / "raw" / "WBSIM_Lab1_2024"


# Infrastructure folder
INFRA = DATA / "infrastructure"

BMMS_file = INFRA / "BMMS_overview.xlsx"
roads_file = INFRA / "_roads.tsv"
road_info_file = INFRA / "Roads_InfoAboutEachLRP.csv"

BMMS = pd.read_excel(BMMS_file)
roads = pd.read_csv(roads_file, sep="\t", low_memory= False)
road_info = pd.read_csv(road_info_file)


COORD_SCALE = infer_coord_scale(roads)
if COORD_SCALE:
    print(f"Detected fixed-point coordinates: divide by {int(COORD_SCALE)} to get degrees.")
else:
    print("Coordinates appear to already be in degrees.")

def wide_to_long(df):
    rows = []
    for _, row in df.iterrows():
        vals = row.values
        road = vals[0]
        seq = 0
        for j in range(1, len(vals) - 2, 3):
            lrp, lat, lon = vals[j], vals[j+1], vals[j+2]
            if pd.isna(lrp):
                continue
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                continue
            if COORD_SCALE:
                lat_f /= COORD_SCALE
                lon_f /= COORD_SCALE
            rows.append({
                "road": road,
                "lrp": str(lrp).strip(),
                "lat": lat_f,
                "lon": lon_f,
                "seq": seq,
            })
            seq += 1
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Functions for start‑point correction, trimming, and outlier detection
# -----------------------------------------------------------------------------
def correct_start_points(long_df, road_list, multiplier=3, min_abs_km=5.0):
    """
    For roads in road_list that have exactly one large step among the first two steps,
    reposition the first two points so that distances become proportional to chainage.
    """
    df = long_df.copy()
    df = df.sort_values(['road', 'chainage', 'seq']).reset_index(drop=True)

    for road in road_list:
        mask = df['road'] == road
        if not mask.any():
            continue
        idx = df[mask].index
        if len(idx) < 3:
            continue

        sub = df.loc[idx].copy()
        lats = sub['lat'].values
        lons = sub['lon'].values
        n = len(sub)

        steps = []
        for i in range(1, n):
            steps.append(haversine(lats[i-1], lons[i-1], lats[i], lons[i]))
        if len(steps) < 2:
            continue

        median_step = np.median(steps)
        mad_step = np.median(np.abs(steps - median_step))
        rel_thresh = median_step + multiplier * mad_step
        outlier_mask = [(s > rel_thresh) and (s > min_abs_km) for s in steps]

        if not (outlier_mask[0] ^ outlier_mask[1]):   # not exactly one outlier
            continue

        ch = sub['chainage'].values
        if outlier_mask[0] and not outlier_mask[1]:
            # step0 outlier, step1 normal → point0 is wrong
            p1 = (lats[1], lons[1])
            p2 = (lats[2], lons[2])
            dist_p1p2 = steps[1]
            if dist_p1p2 > 0:
                target_AB = ch[1] - ch[0]
                u = ((p2[0] - p1[0]) / dist_p1p2, (p2[1] - p1[1]) / dist_p1p2)
                new_lat0 = p1[0] - target_AB * u[0]
                new_lon0 = p1[1] - target_AB * u[1]
                df.loc[idx[0], 'lat'] = new_lat0
                df.loc[idx[0], 'lon'] = new_lon0
                print(f"Road {road}: corrected point 0 using direction from point1→point2")
        elif not outlier_mask[0] and outlier_mask[1]:
            # step0 normal, step1 outlier → point1 and point2? Actually point1 is good? Wait: step1 is between point1 and point2,
            # so point2 might be wrong. We'll move point1 and point0 accordingly.
            p1_orig = (lats[1], lons[1])
            p2 = (lats[2], lons[2])
            dist_p1p2_orig = steps[1]
            if dist_p1p2_orig > 0:
                target_BC = ch[2] - ch[1]
                target_AB = ch[1] - ch[0]
                u = ((p1_orig[0] - p2[0]) / dist_p1p2_orig, (p1_orig[1] - p2[1]) / dist_p1p2_orig)
                new_lat1 = p2[0] + target_BC * u[0]
                new_lon1 = p2[1] + target_BC * u[1]
                new_lat0 = new_lat1 + target_AB * u[0]
                new_lon0 = new_lon1 + target_AB * u[1]
                df.loc[idx[0], 'lat'] = new_lat0
                df.loc[idx[0], 'lon'] = new_lon0
                df.loc[idx[1], 'lat'] = new_lat1
                df.loc[idx[1], 'lon'] = new_lon1
                print(f"Road {road}: corrected points 0 and 1 using direction from point2→original point1")
    return df

def choose_direction_and_trim(group, max_trailing=MAX_TRAILING_DROP, max_leading=MAX_LEADING_DROP):
    # (unchanged)
    g_fwd = group.dropna(subset=["chainage"]).sort_values(["chainage","seq"]).reset_index(drop=True)
    if len(g_fwd) < 2:
        return g_fwd, "forward"
    lats_fwd = g_fwd["lat"].to_numpy()
    lons_fwd = g_fwd["lon"].to_numpy()
    ch_fwd = g_fwd["chainage"].to_numpy()
    walk_outlier_fwd = walk_chainage_filter(lats_fwd, lons_fwd, ch_fwd, JUMP_ABS_KM, JUMP_RATIO)
    count_fwd = int(walk_outlier_fwd.sum())
    g_rev = g_fwd.iloc[::-1].reset_index(drop=True)
    lats_rev = g_rev["lat"].to_numpy()
    lons_rev = g_rev["lon"].to_numpy()
    ch_rev_raw = g_rev["chainage"].to_numpy()
    ch_rev = ch_rev_raw.max() - ch_rev_raw
    walk_outlier_rev = walk_chainage_filter(lats_rev, lons_rev, ch_rev, JUMP_ABS_KM, JUMP_RATIO)
    count_rev = int(walk_outlier_rev.sum())
    if count_rev < count_fwd:
        chosen = "backward"
        g_chosen = g_rev
        walk_chosen = walk_outlier_rev
    else:
        chosen = "forward"
        g_chosen = g_fwd
        walk_chosen = walk_outlier_fwd
    if len(g_chosen) > 0 and (max_trailing > 0 or max_leading > 0):
        drop_idx = []
        trimmed = 0
        for i in range(len(g_chosen)-1, -1, -1):
            if trimmed >= max_trailing: break
            if walk_chosen[i]: drop_idx.append(i); trimmed += 1
            else: break
        trimmed = 0
        for i in range(len(g_chosen)):
            if trimmed >= max_leading: break
            if walk_chosen[i]: drop_idx.append(i); trimmed += 1
            else: break
        if drop_idx:
            g_chosen = g_chosen.drop(index=drop_idx).reset_index(drop=True)
    return g_chosen, chosen

def count_step_outliers_per_road(df, multiplier=3, min_abs_km=ABS_THRESHOLD_KM):
    # (unchanged)
    df = df.sort_values(['road','chainage','seq']).reset_index(drop=True)
    results = []
    for road, group in df.groupby('road', sort=False):
        lats = group['lat'].values
        lons = group['lon'].values
        n = len(group)
        if n < 2:
            results.append({'road': road, 'step_outlier_count': 0, 'first_two_exactly_one': False})
            continue
        steps = []
        for i in range(1, n):
            steps.append(haversine(lats[i-1], lons[i-1], lats[i], lons[i]))
        median_step = np.median(steps)
        mad_step = np.median(np.abs(steps - median_step))
        rel_threshold = median_step + multiplier * mad_step
        outlier_mask = [(s > rel_threshold) and (s > min_abs_km) for s in steps]
        outlier_count = sum(outlier_mask)
        first_two_exactly_one = False
        if len(steps) >= 2:
            first_two = outlier_mask[:2]
            if sum(first_two) == 1:
                first_two_exactly_one = True
        results.append({'road': road, 'step_outlier_count': outlier_count,
                        'first_two_exactly_one': first_two_exactly_one})
    return pd.DataFrame(results)

def analyze_single_group(road, group):
    # (unchanged, but ensure it's present)
    if len(group) < 5:
        return None
    group = group.reset_index(drop=True)
    chainage_vals = group["chainage"].to_numpy()
    if len(chainage_vals) >= 2 and chainage_vals[-1] < chainage_vals[0]:
        chainage_work = chainage_vals.max() - chainage_vals
    else:
        chainage_work = chainage_vals
    start_chainage = chainage_work[0]
    lats = group["lat"].to_numpy()
    lons = group["lon"].to_numpy()
    step_dist, chainage_step, step_ratio = step_metrics(lats, lons, chainage_work)
    walk_outlier = walk_chainage_filter(lats, lons, chainage_work, JUMP_ABS_KM, JUMP_RATIO)
    step_outlier = (chainage_step > 0) & (step_dist > JUMP_ABS_KM) & (step_dist > JUMP_RATIO * chainage_step)
    step_outlier |= (step_dist > MAX_STEP_KM)
    endpoint_outlier = np.zeros(len(group), dtype=bool)
    if len(group) >= 2:
        typical_steps = step_dist[(chainage_step > 0) & (~step_outlier)]
        typical_step = float(np.median(typical_steps)) if typical_steps.size else 0.0
        def is_jump(step_km, ch_step_km):
            if ch_step_km <= 0: return False
            if step_km > MAX_STEP_KM: return True
            if step_km > JUMP_ABS_KM and step_km > JUMP_RATIO * ch_step_km: return True
            if typical_step > 0 and step_km > max(JUMP_ABS_KM, 3.0 * typical_step): return True
            return False
        last_good = 0
        for i in range(1, len(group)):
            ch_delta = float(chainage_work[i] - chainage_work[last_good])
            if ch_delta <= 0: continue
            dist = haversine(lats[last_good], lons[last_good], lats[i], lons[i])
            if is_jump(dist, ch_delta):
                endpoint_outlier[i] = True
            else:
                last_good = i
                break
        last_good = len(group) - 1
        for i in range(len(group)-2, -1, -1):
            ch_delta = float(chainage_work[last_good] - chainage_work[i])
            if ch_delta <= 0: continue
            dist = haversine(lats[last_good], lons[last_good], lats[i], lons[i])
            if is_jump(dist, ch_delta):
                endpoint_outlier[i] = True
            else:
                last_good = i
                break
    step_dist_clean = step_dist.copy()
    step_dist_clean[step_outlier] = 0.0
    dist_coords_cum = np.cumsum(step_dist_clean)
    road_data = []
    for i, row in group.iterrows():
        road_data.append({
            "road": road, "lrp": row["lrp"], "lat": row["lat"], "lon": row["lon"],
            "seq": row["seq"], "dist_coords": float(dist_coords_cum[i]),
            "chainage": row["chainage"], "dist_chainage": chainage_work[i] - start_chainage,
            "step_dist": float(step_dist[i]), "chainage_step": float(chainage_step[i]) if np.isfinite(chainage_step[i]) else np.nan,
            "step_ratio": float(step_ratio[i]) if np.isfinite(step_ratio[i]) else np.nan,
            "step_outlier": bool(step_outlier[i]), "walk_outlier": bool(walk_outlier[i]),
            "endpoint_outlier": bool(endpoint_outlier[i]),
        })
    road_df = pd.DataFrame(road_data)
    road_df["lowess_fit"] = lowess(road_df["dist_chainage"], road_df["dist_coords"],
                                    frac=0.2, it=3, return_sorted=False)
    road_df["residual"] = road_df["dist_chainage"] - road_df["lowess_fit"]
    fit_nonzero = road_df["lowess_fit"].replace(0, np.nan)
    road_df["residual_pct"] = road_df["residual"] / fit_nonzero
    pct = road_df["residual_pct"].dropna()
    if len(pct) >= 3:
        median_pct = np.median(pct)
        mad_pct = np.median(np.abs(pct - median_pct))
        thresh_pos = median_pct + K_MAD * mad_pct
        thresh_neg = median_pct - K_MAD * mad_pct
    else:
        thresh_pos = thresh_neg = np.inf   # fallback
    lowess_outlier = ((road_df["residual"] > ABS_THRESHOLD_KM) & (road_df["residual_pct"] > thresh_pos)) | \
                     ((road_df["residual"] < -ABS_THRESHOLD_KM) & (road_df["residual_pct"] < thresh_neg))
    road_df["outlier"] = lowess_outlier | road_df["step_outlier"] | road_df["walk_outlier"] | road_df["endpoint_outlier"]
    is_endpoint = (road_df.index == 0) | (road_df.index == len(road_df)-1)
    jump_based = road_df["step_outlier"] | road_df["walk_outlier"]
    road_df.loc[is_endpoint, "outlier"] = (road_df["endpoint_outlier"] | (lowess_outlier & jump_based))[is_endpoint]
    return road_df

def flag_step_mad_outliers(df, multiplier=3, min_abs_km=5.0):
    # (unchanged)
    df = df.copy()
    df = df.sort_values(['road','chainage','seq']).reset_index(drop=True)
    outlier_flags = []
    for road, group in df.groupby('road', sort=False):
        lats = group['lat'].values; lons = group['lon'].values; n = len(group)
        if n < 2:
            outlier_flags.extend([False]*n); continue
        steps = []
        for i in range(1, n):
            steps.append(haversine(lats[i-1], lons[i-1], lats[i], lons[i]))
        median_step = np.median(steps)
        mad_step = np.median(np.abs(steps - median_step))
        threshold = median_step + multiplier * mad_step
        step_outliers = [(s > threshold) and (s > min_abs_km) for s in steps]
        road_outliers = [False] + step_outliers
        outlier_flags.extend(road_outliers)
    df['step_mad_outlier'] = outlier_flags
    return df

def flag_large_steps(df, threshold_km=20.0):
    """
    Adds a boolean column 'large_step' to the DataFrame.
    A step is considered large if the Haversine distance between consecutive points
    exceeds threshold_km. The first point of each road gets False (no incoming step).
    Also returns a summary DataFrame with columns:
        road, max_step_km, large_step_count, has_large_step
    """
    df = df.copy()
    df = df.sort_values(['road', 'chainage', 'seq']).reset_index(drop=True)

    large_flags = []
    summary = []

    for road, group in df.groupby('road', sort=False):
        lats = group['lat'].values
        lons = group['lon'].values
        n = len(group)

        if n < 2:
            large_flags.extend([False] * n)
            summary.append({'road': road, 'max_step_km': 0, 'large_step_count': 0, 'has_large_step': False})
            continue

        steps = []
        for i in range(1, n):
            steps.append(haversine(lats[i-1], lons[i-1], lats[i], lons[i]))

        max_step = max(steps) if steps else 0
        large_count = sum(1 for s in steps if s > threshold_km)
        has_large = large_count > 0

        # For points: first point gets False, then each step's status
        road_large = [False] + [s > threshold_km for s in steps]
        large_flags.extend(road_large)

        summary.append({
            'road': road,
            'max_step_km': max_step,
            'large_step_count': large_count,
            'has_large_step': has_large
        })

    df['large_step'] = large_flags
    summary_df = pd.DataFrame(summary)
    return df, summary_df

def detect_typos_latlon(long_df, abs_lat_deg=0.3, abs_lon_deg=0.3, rel_mult=15.0, min_typ_deg=1e-5):
    # (unchanged, returns dict)
    required = {"road","lrp","lat","lon"}
    if required - set(long_df.columns):
        raise ValueError("missing columns")
    df = long_df.copy()
    errors = {}
    for road, g in df.groupby("road", sort=False):
        if len(g) < 3:
            continue
        lrps = g["lrp"].astype(str).to_numpy()
        lat = g["lat"].astype(float).to_numpy()
        lon = g["lon"].astype(float).to_numpy()
        lat_steps = np.abs(np.diff(lat))
        lon_steps = np.abs(np.diff(lon))
        lat_steps_capped = lat_steps[lat_steps < abs_lat_deg]
        lon_steps_capped = lon_steps[lon_steps < abs_lon_deg]
        typ_lat = np.median(lat_steps_capped) if lat_steps_capped.size else np.median(lat_steps)
        typ_lon = np.median(lon_steps_capped) if lon_steps_capped.size else np.median(lon_steps)
        typ_lat = max(float(typ_lat), min_typ_deg)
        typ_lon = max(float(typ_lon), min_typ_deg)
        road_err = {}
        for i in range(len(g)):
            lrp_i, lat_i, lon_i = lrps[i], lat[i], lon[i]
            if i > 0:
                lat_prev = abs(lat_i - lat[i-1]); lon_prev = abs(lon_i - lon[i-1])
            else:
                lat_prev = lon_prev = None
            if i < len(g)-1:
                lat_next = abs(lat_i - lat[i+1]); lon_next = abs(lon_i - lon[i+1])
            else:
                lat_next = lon_next = None
            if lat_prev is None or lat_next is None:
                lat_one = lat_prev if lat_next is None else lat_next
                lon_one = lon_prev if lon_next is None else lon_next
                reasons = []
                if lat_one is not None and (lat_one > abs_lat_deg and lat_one > rel_mult * typ_lat):
                    reasons.append("lat_endpoint_jump")
                if lon_one is not None and (lon_one > abs_lon_deg and lon_one > rel_mult * typ_lon):
                    reasons.append("lon_endpoint_jump")
                if reasons:
                    road_err[lrp_i] = {"lat": float(lat_i), "lon": float(lon_i), "rule": ",".join(reasons)}
            else:
                lat_both_abs = (lat_prev > abs_lat_deg and lat_next > abs_lat_deg)
                lon_both_abs = (lon_prev > abs_lon_deg and lon_next > abs_lon_deg)
                lat_both_rel = (lat_prev > rel_mult * typ_lat and lat_next > rel_mult * typ_lat)
                lon_both_rel = (lon_prev > rel_mult * typ_lon and lon_next > rel_mult * typ_lon)
                reasons = []
                if lat_both_abs or lat_both_rel:
                    reasons.append("lat_both_sides_jump")
                if lon_both_abs or lon_both_rel:
                    reasons.append("lon_both_sides_jump")
                if reasons:
                    road_err[lrp_i] = {"lat": float(lat_i), "lon": float(lon_i), "rule": ",".join(reasons)}
        if road_err:
            errors[str(road)] = road_err
    return errors

def analyze_roads_with_lowess(df):
    results = []
    for road, group in df.groupby("road"):
        group_ordered, _ = choose_direction_and_trim(group)
        road_df = analyze_single_group(road, group_ordered)
        if road_df is not None:
            results.append(road_df)
    return pd.concat(results) if results else pd.DataFrame()

def repair_outliers(long_df_with_chainage, outlier_df):
    # (unchanged)
    repaired = long_df_with_chainage.copy()
    outlier_flags = outlier_df[["road","lrp","seq","outlier"]].copy()
    repaired = repaired.merge(outlier_flags, on=["road","lrp","seq"], how="left")
    repaired["outlier"] = repaired["outlier"].fillna(False)
    repaired_lat = []; repaired_lon = []
    for road, g in repaired.groupby("road", sort=False):
        
        g = g.copy()
        g_sorted = g.sort_values(["chainage","seq"], na_position="last").reset_index()
        idx_map = g_sorted["index"].to_numpy()
        lat = g_sorted["lat"].to_numpy(copy=True)
        lon = g_sorted["lon"].to_numpy(copy=True)
        ch = g_sorted["chainage"].to_numpy(copy=True)
        out = g_sorted["outlier"].to_numpy(copy=True)
        for i in range(len(g_sorted)):
            if not out[i]:
                continue
            prev_i = i-1
            while prev_i >= 0 and out[prev_i]:
                prev_i -= 1
            next_i = i+1
            while next_i < len(g_sorted) and out[next_i]:
                next_i += 1
            if prev_i >= 0 and next_i < len(g_sorted):
                if not np.isnan(ch[prev_i]) and not np.isnan(ch[next_i]) and not np.isnan(ch[i]):
                    denom = ch[next_i] - ch[prev_i]
                    if denom != 0:
                        t = (ch[i] - ch[prev_i]) / denom
                        lat[i] = lat[prev_i] + t * (lat[next_i] - lat[prev_i])
                        lon[i] = lon[prev_i] + t * (lon[next_i] - lon[prev_i])
                    else:
                        lat[i] = (lat[prev_i] + lat[next_i]) / 2
                        lon[i] = (lon[prev_i] + lon[next_i]) / 2
                else:
                    lat[i] = (lat[prev_i] + lat[next_i]) / 2
                    lon[i] = (lon[prev_i] + lon[next_i]) / 2
            elif prev_i >= 0:
                lat[i] = lat[prev_i]; lon[i] = lon[prev_i]
            elif next_i < len(g_sorted):
                lat[i] = lat[next_i]; lon[i] = lon[next_i]
        repaired_lat.extend(zip(idx_map, lat))
        repaired_lon.extend(zip(idx_map, lon))
    repaired_lat = dict(repaired_lat)
    repaired_lon = dict(repaired_lon)
    repaired["lat"] = repaired.index.map(lambda i: repaired_lat.get(i, repaired.loc[i, "lat"]))
    repaired["lon"] = repaired.index.map(lambda i: repaired_lon.get(i, repaired.loc[i, "lon"]))
    return repaired

# =============================================================================
# MAIN PIPELINE
# =============================================================================
# 1. Convert to long format and add chainage
long_df = wide_to_long(roads)
road_info["lrp"] = road_info["lrp"].astype(str).str.strip()
long_df = long_df.merge(road_info[["road","lrp","chainage"]], on=["road","lrp"], how="left")

# 2. Identify roads that need start‑point correction (odd count and first_two_exactly_one)
summary = count_step_outliers_per_road(long_df, multiplier=10)  # using a high multiplier to be strict?
# Actually we want to detect obvious start problems; multiplier=3 is more common. Use 3.
summary = count_step_outliers_per_road(long_df, multiplier=3, min_abs_km=5.0)
roads_to_correct = summary[(summary['step_outlier_count'] % 2 != 0) & (summary['first_two_exactly_one'])]['road'].tolist()
print(f"Roads to correct: {roads_to_correct}")

# 3. Apply start‑point correction
if roads_to_correct:
    long_df = correct_start_points(long_df, roads_to_correct, multiplier=3, min_abs_km=5.0)

# 4. Add step‑MAD outlier column
long_df = flag_step_mad_outliers(long_df, multiplier=3, min_abs_km=5.0)

# 5. LOWESS outlier detection
lowess_results = analyze_roads_with_lowess(long_df)

# 6. Typo detection (adds column 'typo_outlier')
typo_dict = detect_typos_latlon(long_df)
long_df['typo_outlier'] = False
for road, points in typo_dict.items():
    for lrp in points:
        mask = (long_df['road'] == road) & (long_df['lrp'] == lrp)
        long_df.loc[mask, 'typo_outlier'] = True

# 7. Merge LOWESS outlier flags into long_df
long_df = long_df.merge(
    lowess_results[['road','lrp','seq','outlier']].rename(columns={'outlier':'lowess_outlier'}),
    on=['road','lrp','seq'], how='left'
)
long_df['lowess_outlier'] = long_df['lowess_outlier'].fillna(False)

# 8. Combine all outlier flags
long_df['combined_outlier'] = long_df['step_mad_outlier'] | long_df['lowess_outlier'] | long_df['typo_outlier']

# 9. Repair using combined outlier flag
repaired_long = repair_outliers(long_df, long_df.rename(columns={'combined_outlier':'outlier'}))


### DIAGNOSTIC TEST TO SPOT HOW WELL WE HAVE DONE ###
# After all outlier detection and repair, or right after creating long_df
repaired_long_df, large_step_summary = flag_large_steps(repaired_long, threshold_km=50.0)

# Show roads with large steps
print("Roads with steps > 10 km:")
print(large_step_summary[large_step_summary['has_large_step']])

for road in large_step_summary[large_step_summary['has_large_step']]['road']:
    road_data = repaired_long_df[repaired_long_df['road'] == road].sort_values(['chainage', 'seq'])
    plt.figure(figsize=(8,6))
    plt.plot(road_data['lon'], road_data['lat'], 'o-', label='all points')
    # highlight large steps
    for i in range(1, len(road_data)):
        if road_data.iloc[i]['large_step']:
            plt.plot([road_data.iloc[i-1]['lon'], road_data.iloc[i]['lon']],
                     [road_data.iloc[i-1]['lat'], road_data.iloc[i]['lat']],
                     'r-', linewidth=2, label='large step' if i==1 else '')
    plt.title(f"Road {road} – steps >20 km highlighted")
    plt.legend()
    plt.axis('equal')
    plt.show()

# 10. Rebuild wide format
repaired_roads = roads.copy()
for col in repaired_roads.columns:
    if str(col).lower().startswith(('lat','lon')):
        repaired_roads[col] = np.nan

lookup = {(row['road'], row['lrp'], int(row['seq'])): (row['lat'], row['lon'])
          for _, row in repaired_long.iterrows()}

for i, row in repaired_roads.iterrows():
    road = row.iloc[0]
    seq = 0
    for j in range(1, len(row)-2, 3):
        lrp = row.iloc[j]
        if pd.isna(lrp):
            continue
        key = (road, str(lrp).strip(), seq)
        if key in lookup:
            lat, lon = lookup[key]
            repaired_roads.iat[i, j+1] = lat
            repaired_roads.iat[i, j+2] = lon
        seq += 1



# 12. Export final roads
repaired_path = BASE / "data" / "processed"  / 'WBSIM_Lab1_2024'/'infrastructure'/"_roads.tsv"
repaired_roads_export = normalize_coord_columns(repaired_roads, COORD_SCALE)
repaired_roads_export.to_csv(repaired_path, sep="\t", index=False)
print(f"Repaired roads saved to: {repaired_path}")