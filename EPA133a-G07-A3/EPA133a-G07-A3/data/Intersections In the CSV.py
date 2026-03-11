from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

INPUT_ROAD_OBJECTS = Path("road_objects.csv")
INPUT_INTERSECTIONS_RAW = Path("intersections_raw_matches.csv")
OUTPUT_FINAL = Path("network_model.csv")

# keep these consistent with your detector
INTERSECTION_ID_START = 9_000_000
CLUSTER_THRESHOLD_M = 80.0


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def ensure_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_inputs():
    roads = pd.read_csv(INPUT_ROAD_OBJECTS)
    inter = pd.read_csv(INPUT_INTERSECTIONS_RAW)

    ensure_columns(
        roads,
        ["id", "road", "chainage", "lat", "lon", "name", "model_type", "length", "condition"],
        "road_objects"
    )
    ensure_columns(
        inter,
        ["road_a", "road_b", "chainage_a", "chainage_b", "lat", "lon", "dist_m"],
        "intersections_raw_matches"
    )

    roads = roads.copy()
    roads["road"] = roads["road"].astype(str).str.strip().str.upper()
    roads["chainage"] = pd.to_numeric(roads["chainage"], errors="coerce")
    roads["lat"] = pd.to_numeric(roads["lat"], errors="coerce")
    roads["lon"] = pd.to_numeric(roads["lon"], errors="coerce")
    roads["id"] = pd.to_numeric(roads["id"], errors="raise").astype(int)

    inter = inter.copy()
    inter["road_a"] = inter["road_a"].astype(str).str.strip().str.upper()
    inter["road_b"] = inter["road_b"].astype(str).str.strip().str.upper()
    inter["chainage_a"] = pd.to_numeric(inter["chainage_a"], errors="coerce")
    inter["chainage_b"] = pd.to_numeric(inter["chainage_b"], errors="coerce")
    inter["lat"] = pd.to_numeric(inter["lat"], errors="coerce")
    inter["lon"] = pd.to_numeric(inter["lon"], errors="coerce")
    inter["dist_m"] = pd.to_numeric(inter["dist_m"], errors="coerce")

    roads = roads.sort_values(["road", "chainage", "id"]).reset_index(drop=True)
    inter = inter.dropna(subset=["road_a", "road_b", "chainage_a", "chainage_b", "lat", "lon"]).reset_index(drop=True)

    return roads, inter


def cluster_raw_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster near-duplicate raw pair matches into one final intersection.
    Keeps all member rows so we know which roads/chainages to insert into.
    """
    clusters = []

    for _, row in raw.iterrows():
        assigned = False
        for cluster in clusters:
            d = haversine_m(row["lat"], row["lon"], cluster["lat"], cluster["lon"])
            if d <= CLUSTER_THRESHOLD_M:
                cluster["members"].append(row.to_dict())
                cluster["lat"] = float(np.mean([m["lat"] for m in cluster["members"]]))
                cluster["lon"] = float(np.mean([m["lon"] for m in cluster["members"]]))
                assigned = True
                break

        if not assigned:
            clusters.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "members": [row.to_dict()]
            })

    out = []
    next_id = INTERSECTION_ID_START

    for cluster in clusters:
        members = cluster["members"]
        connected_roads = sorted(set(
            [m["road_a"] for m in members] + [m["road_b"] for m in members]
        ))

        out.append({
            "intersection_id": next_id,
            "lat": cluster["lat"],
            "lon": cluster["lon"],
            "name": f"INT_{'_'.join(connected_roads)}",
            "connected_roads": connected_roads,
            "members": members,
        })
        next_id += 1

    return pd.DataFrame(out)


def nearest_row_id_for_chainage(road_df: pd.DataFrame, target_chainage: float) -> int:
    diffs = (road_df["chainage"] - target_chainage).abs().to_numpy()
    idx = int(np.argmin(diffs))
    return int(road_df.iloc[idx]["id"])


def build_insert_map(roads: pd.DataFrame, clustered: pd.DataFrame):
    """
    Returns:
      insert_after[(road, row_id)] = [intersection_row_dict, ...]
    """
    insert_after = defaultdict(list)

    roads_by_name = {
        road_name: g.sort_values(["chainage", "id"]).copy()
        for road_name, g in roads.groupby("road", sort=False)
    }

    for _, c in clustered.iterrows():
        intersection_id = int(c["intersection_id"])
        lat = float(c["lat"])
        lon = float(c["lon"])
        name = str(c["name"])

        # For each member road pair, insert this same intersection into both roads.
        # We deduplicate per road so the same intersection is inserted once per connected road.
        road_to_target_chainages = defaultdict(list)

        for m in c["members"]:
            road_to_target_chainages[m["road_a"]].append(float(m["chainage_a"]))
            road_to_target_chainages[m["road_b"]].append(float(m["chainage_b"]))

        for road_name, chainages in road_to_target_chainages.items():
            if road_name not in roads_by_name:
                continue

            road_df = roads_by_name[road_name]
            target_chainage = float(np.mean(chainages))
            row_id = nearest_row_id_for_chainage(road_df, target_chainage)

            intersection_row = {
                "id": intersection_id,
                "road": road_name,
                "lrp": "",
                "chainage": target_chainage,
                "lat": lat,
                "lon": lon,
                "name": name,
                "model_type": "intersection",
                "length": 0.0,
                "condition": "N/A",
            }

            # preserve optional columns if they exist later
            insert_after[(road_name, row_id)].append(intersection_row)

    # sort multiple intersections after the same row by chainage
    for key in insert_after:
        insert_after[key] = sorted(insert_after[key], key=lambda r: r["chainage"])

    return insert_after


def merge_roads_and_intersections(roads: pd.DataFrame, insert_after) -> pd.DataFrame:
    """
    Write each road row in order, and immediately after a matched row insert
    any intersection rows assigned to that location.
    """
    final_rows = []

    # preserve original column order where possible
    base_cols = list(roads.columns)
    extra_cols = [c for c in ["lrp"] if c not in base_cols]
    all_cols = base_cols + [c for c in extra_cols if c not in base_cols]

    for road_name, g in roads.groupby("road", sort=False):
        g = g.sort_values(["chainage", "id"]).copy()

        for _, row in g.iterrows():
            final_rows.append(row.to_dict())

            key = (road_name, int(row["id"]))
            if key in insert_after:
                for inter_row in insert_after[key]:
                    out_row = {col: np.nan for col in all_cols}
                    out_row.update(inter_row)

                    # fill any missing schema columns safely
                    if "structure_type" in roads.columns and "structure_type" not in out_row:
                        out_row["structure_type"] = ""
                    if "name" not in out_row:
                        out_row["name"] = ""
                    if "condition" not in out_row:
                        out_row["condition"] = "N/A"
                    if "length" not in out_row:
                        out_row["length"] = 0.0

                    final_rows.append(out_row)

    final_df = pd.DataFrame(final_rows)

    # remove exact duplicate intersection rows accidentally inserted twice after same row
    final_df = final_df.drop_duplicates(
        subset=["road", "id", "model_type", "chainage"],
        keep="first"
    ).reset_index(drop=True)

    return final_df


def validate_intersections(final_df: pd.DataFrame):
    inter = final_df[final_df["model_type"].astype(str).str.lower() == "intersection"].copy()
    if inter.empty:
        print("No intersections inserted.")
        return

    counts = inter.groupby("id")["road"].nunique().sort_values()
    print("\nIntersection usage by unique roads:")
    print(counts.to_string())

    bad = counts[counts < 2]
    if not bad.empty:
        print("\nWarning: these intersections appear on fewer than 2 roads:")
        print(bad.to_string())


def main():
    roads, raw_intersections = load_inputs()
    clustered = cluster_raw_matches(raw_intersections)
    insert_after = build_insert_map(roads, clustered)
    final_df = merge_roads_and_intersections(roads, insert_after)

    final_df.to_csv(OUTPUT_FINAL, index=False)

    print(f"Wrote final network CSV to: {OUTPUT_FINAL.resolve()}")
    print(f"Road rows: {len(roads)}")
    print(f"Clustered intersections: {len(clustered)}")
    print(f"Final rows: {len(final_df)}")

    validate_intersections(final_df)

    print("\nSample inserted intersections:")
    sample = final_df[final_df["model_type"].astype(str).str.lower() == "intersection"][
        ["id", "road", "chainage", "lat", "lon", "name", "model_type"]
    ]
    if len(sample) == 0:
        print("No intersections present.")
    else:
        print(sample.head(20).to_string(index=False))


if __name__ == "__main__":
    main()