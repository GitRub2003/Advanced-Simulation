
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#from pathlib import Path
import pandas as pd
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "raw" / "WBSIM_Lab1_2024"
INFRA = DATA / "infrastructure"


# Path of the current script/notebook
BASE = Path(__file__).resolve().parents[1]
# If using a notebook instead:
# BASE = Path().resolve().parents[0]

DATA = BASE / "data" / "raw" / "WBSIM_Lab1_2024"

print(DATA)

# Infrastructure folder
INFRA = DATA / "infrastructure"

BMMS_file = INFRA / "BMMS_overview.xlsx"
roads_file = INFRA / "_roads.tsv"
road_info_file = INFRA / "Roads_InfoAboutEachLRP.csv"

BMMS = pd.read_excel(BMMS_file)
roads = pd.read_csv(roads_file, sep="\t")
road_info = pd.read_csv(road_info_file)

print(BMMS.head())
print(roads.head())
print(road_info.head())


def wide_to_long(df):
    """
    Convert wide road format:
    road, (lrp, lat, lon), (lrp, lat, lon), ...

    Into long format:
    road | lrp | lat | lon
    """


    rows = []

    for _, row in df.iterrows():
        vals = row.values
        road = vals[0]
        order = 0  # keeps order of LRPs along this road

        for j in range(1, len(vals) - 2, 3):
            lrp, lat, lon = vals[j], vals[j + 1], vals[j + 2]

            if pd.isna(lrp):
                continue

            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                continue

            rows.append({
                "road": road,
                "lrp": str(lrp).strip(),
                "lat": lat_f,
                "lon": lon_f,
                "order": order
            })

            order += 1

    return pd.DataFrame(rows)


def detect_typos_latlon(
    long_df,
    abs_lat_deg=0.3,
    abs_lon_deg=0.3,
    rel_mult=15.0,
    min_typ_deg=1e-5,
    sort_by=("road", "lrp")  # change if you have a better ordering column
):
    """
    Flags a point if its lat and/or lon is very different from BOTH neighbours.
    long_df must have columns: road, lrp, lat, lon
    Returns: {road: {lrp: info_dict}}
    """

    required = {"road", "lrp", "lat", "lon"}
    missing = required - set(long_df.columns)
    if missing:  #redundant?
        raise ValueError(f"long_df is missing columns: {missing}")

    # Make sure we're in the right order within each road
    df = long_df.copy()


    errors = {}

    for road, g in df.groupby("road", sort=False):
        g = g.sort_values("order")
        if len(g) < 3:
            continue

        lrps = g["lrp"].astype(str).to_numpy()
        lat = g["lat"].astype(float).to_numpy()
        lon = g["lon"].astype(float).to_numpy()

        # Per-road step sizes
        lat_steps = np.abs(np.diff(lat))
        lon_steps = np.abs(np.diff(lon))

        # Exclude extreme steps so "typical" isn't polluted
        lat_steps_capped = lat_steps[lat_steps < abs_lat_deg]
        lon_steps_capped = lon_steps[lon_steps < abs_lon_deg]

        typ_lat = np.median(lat_steps_capped) if lat_steps_capped.size else np.median(lat_steps)
        typ_lon = np.median(lon_steps_capped) if lon_steps_capped.size else np.median(lon_steps)

        typ_lat = max(float(typ_lat), min_typ_deg)
        typ_lon = max(float(typ_lon), min_typ_deg)

        road_err = {}

        for i in range(len(g)):
            lrp_i, lat_i, lon_i = lrps[i], lat[i], lon[i]

            # neighbour deltas
            if i > 0:
                lat_prev = abs(lat_i - lat[i - 1])
                lon_prev = abs(lon_i - lon[i - 1])
            else:
                lat_prev = lon_prev = None

            if i < len(g) - 1:
                lat_next = abs(lat_i - lat[i + 1])
                lon_next = abs(lon_i - lon[i + 1])
            else:
                lat_next = lon_next = None

            # endpoints
            if lat_prev is None or lat_next is None:
                lat_one = lat_prev if lat_next is None else lat_next
                lon_one = lon_prev if lon_next is None else lon_next

                reasons = []
                if lat_one is not None and (lat_one > abs_lat_deg and lat_one > rel_mult * typ_lat):
                    reasons.append("lat_endpoint_jump")
                if lon_one is not None and (lon_one > abs_lon_deg and lon_one > rel_mult * typ_lon):
                    reasons.append("lon_endpoint_jump")

                if reasons:
                    road_err[lrp_i] = {
                        "lat": float(lat_i),
                        "lon": float(lon_i),
                        "lat_jump_deg": float(lat_one) if lat_one is not None else None,
                        "lon_jump_deg": float(lon_one) if lon_one is not None else None,
                        "typ_lat_step_deg": typ_lat,
                        "typ_lon_step_deg": typ_lon,
                        "rule": ",".join(reasons),
                    }
                continue

            # interior: must be weird vs BOTH neighbours
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
                road_err[lrp_i] = {
                    "lat": float(lat_i),
                    "lon": float(lon_i),
                    "lat_prev_deg": float(lat_prev),
                    "lat_next_deg": float(lat_next),
                    "lon_prev_deg": float(lon_prev),
                    "lon_next_deg": float(lon_next),
                    "typ_lat_step_deg": typ_lat,
                    "typ_lon_step_deg": typ_lon,
                    "rule": ",".join(reasons),
                }

        if road_err:
            errors[str(road)] = road_err

    return errors


# ---- Usage ----

# long_df is your converted dataframe
long_df = wide_to_long(roads)
subset = long_df[long_df["road"] == "N1"]

typos = detect_typos_latlon(subset)

print("Roads with flagged LRPs:", len(typos))
some_road = next(iter(typos), None)
if some_road:
    print(some_road, typos[some_road])
    print(len(typos[some_road]))
subset = long_df[long_df["road"] == "N1"].copy()

# If LRPs are already in order in the dataset, this is enough:
subset = subset.reset_index(drop=True)


# Collect flagged LRPs
flagged_lrps = set()
if "N1" in typos:
    flagged_lrps = set(typos["N1"].keys())

# Mark rows
subset["flagged"] = subset["lrp"].astype(str).isin(flagged_lrps)

# Plot
plt.figure()

# Line connecting all LRPs
plt.plot(subset["lon"], subset["lat"], linewidth=1, color="gray", alpha=0.7)

# Normal points
plt.scatter(
    subset.loc[~subset["flagged"], "lon"],
    subset.loc[~subset["flagged"], "lat"],
    s=15,
    label="Normal LRPs"
)

# Flagged points (typos)
plt.scatter(
    subset.loc[subset["flagged"], "lon"],
    subset.loc[subset["flagged"], "lat"],
    s=40,
    label="Flagged LRPs (possible typos)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("LRPs for road N1 (typos highlighted)")
plt.axis("equal")
plt.legend()
plt.show()

def correct_flagged_lrps(long_df, typos):
    corrected = long_df.copy()
    corrected["was_corrected"] = False

    for road, flagged_points in typos.items():
        road_mask = corrected["road"].astype(str) == str(road)
        road_df = corrected.loc[road_mask].sort_values("order").reset_index()

        if road_df.empty:
            continue

        lrps = road_df["lrp"].astype(str).to_numpy()
        lat  = road_df["lat"].astype(float).to_numpy()
        lon  = road_df["lon"].astype(float).to_numpy()

        flagged_set = set(map(str, flagged_points.keys()))  # <- important

        for i in range(len(road_df)):
            lrp_i = str(lrps[i])
            if lrp_i not in flagged_set:
                continue

            # need both neighbours to do an average
            if i == 0 or i == len(road_df) - 1:
                continue

            lat_new = (lat[i - 1] + lat[i + 1]) / 2.0
            lon_new = (lon[i - 1] + lon[i + 1]) / 2.0

            orig_idx = road_df.loc[i, "index"]
            corrected.loc[orig_idx, ["lat", "lon"]] = [lat_new, lon_new]
            corrected.loc[orig_idx, "was_corrected"] = True

    return corrected


corrected_df = correct_flagged_lrps(long_df, typos)

plt.figure()

road = "N1"
before = long_df[long_df["road"] == road].sort_values("order")
after  = corrected_df[corrected_df["road"] == road].sort_values("order")

plt.figure()

plt.scatter(after["lon"], after["lat"], s=30, label="After (only flagged moved)")

# highlight corrected ones
m = after["was_corrected"]
plt.scatter(after.loc[m, "lon"], after.loc[m, "lat"], s=60, label="Corrected points")

plt.axis("equal")
plt.legend()
plt.show()

def long_to_wide(long_df, road_col="road", order_col="order", lrp_col="lrp", lat_col="lat", lon_col="lon"):
    """
    Convert long road format:
      road | lrp | lat | lon | order

    Back into wide format:
      road | lrp1 | lat1 | lon1 | lrp2 | lat2 | lon2 | ...

    Returns a wide DataFrame suitable for saving as TSV.
    """

    # Ensure correct types and ordering
    df = long_df.copy()
    df[order_col] = df[order_col].astype(int)

    # Determine maximum number of LRPs across roads
    max_n = df.groupby(road_col)[order_col].max().max() + 1  # since order starts at 0

    wide_rows = []

    for road, g in df.groupby(road_col, sort=False):
        g = g.sort_values(order_col)

        # start row with road
        row = [road]

        # create lookup by order to be safe even if some orders are missing
        by_order = {int(r[order_col]): r for _, r in g.iterrows()}

        for k in range(max_n):
            if k in by_order:
                r = by_order[k]
                row.extend([r[lrp_col], r[lat_col], r[lon_col]])
            else:
                row.extend(["", "", ""])  # pad missing with blanks like typical TSV exports

        wide_rows.append(row)

    # Build column names: first col is road, then triplets
    cols = ["road"]
    for k in range(max_n):
        cols += [f"lrp_{k}", f"lat_{k}", f"lon_{k}"]

    wide_df = pd.DataFrame(wide_rows, columns=cols)
    return wide_df


# --- Convert corrected long -> wide and save in processed data folder---
repaired_wide = long_to_wide(corrected_df)

out_file = BASE / "data" / "processed" / "repaired_roads.tsv"
repaired_wide.to_csv(out_file, sep="\t", index=False)

print("Saved:", out_file)










