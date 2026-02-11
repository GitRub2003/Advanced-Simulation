import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Change data folder to allign with own location
data_folder = r"C:\Users\sybed\Downloads\Project_EPA133a"
os.chdir(data_folder)


BMMS_file = r"WBSIM_Lab1_2024_copy\WBSIM_Lab1_2024\infrastructure\BMMS_overview.xlsx"
BMMS = pd.read_excel(f'{data_folder}/{BMMS_file}')

roads_file = r"WBSIM_Lab1_2024_copy\WBSIM_Lab1_2024\infrastructure\_roads.tsv"
roads = pd.read_csv(f'{data_folder}/{roads_file}', sep='\t')

road_info_file = r"WBSIM_Lab1_2024_copy\WBSIM_Lab1_2024\infrastructure\Roads_InfoAboutEachLRP.csv"
road_info = pd.read_csv(f'{data_folder}/{road_info_file}')

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

        for j in range(1, len(vals) - 2, 3):
            lrp, lat, lon = vals[j], vals[j + 1], vals[j + 2]

            # skip empty
            if pd.isna(lrp) and pd.isna(lat) and pd.isna(lon):
                continue
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
                "lon": lon_f
            })

    long_df = pd.DataFrame(rows)

    return long_df

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
    if missing:
        raise ValueError(f"long_df is missing columns: {missing}")

    # Make sure we're in the right order within each road
    df = long_df.copy()


    errors = {}

    for road, g in df.groupby("road", sort=False):
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

