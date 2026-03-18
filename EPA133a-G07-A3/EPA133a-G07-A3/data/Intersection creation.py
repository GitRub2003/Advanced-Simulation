from pathlib import Path
import itertools
import numpy as np
import pandas as pd

INPUT_ROADS = Path("_roads3.csv")
OUTPUT_INTERSECTIONS = Path("intersections_detected.csv")

from pathlib import Path
import pandas as pd

SIDE_ROADS_FILE = Path("side_road_candidates.csv")

def load_selected_roads() -> list[str]:
    base_roads = ["N1", "N2"]

    if not SIDE_ROADS_FILE.exists():
        raise FileNotFoundError(
            f"{SIDE_ROADS_FILE} not found. Run 'Side roads choosing.py' first."
        )

    df = pd.read_csv(SIDE_ROADS_FILE)

    if "road" not in df.columns or "selected" not in df.columns:
        raise ValueError(
            "side_road_candidates.csv must contain columns 'road' and 'selected'"
        )

    selected_side_roads = (
        df.loc[df["selected"] == True, "road"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    return base_roads + [r for r in selected_side_roads if r not in base_roads]

SELECTED_ROADS = load_selected_roads()
# max distance between two sampled road points to count as an intersection
INTERSECTION_THRESHOLD_M = 120.0

# if multiple detected intersections are very close, merge them into one
CLUSTER_THRESHOLD_M = 80.0

# choose a high ID range that will not overlap with your road-object IDs
INTERSECTION_ID_START = 9_000_000


def haversine_m(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in meters between two coordinates."""
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Raise an error if a required set of columns is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_roads(path: Path) -> pd.DataFrame:
    """Load the road CSV, normalize key columns, and keep only selected roads."""
    roads = pd.read_csv(path)
    _has_cols(roads, ["road", "chainage", "lat", "lon"], "roads")

    roads = roads.copy()
    roads["road"] = roads["road"].astype(str).str.strip().str.upper()
    roads["chainage"] = pd.to_numeric(roads["chainage"], errors="coerce")
    roads["lat"] = pd.to_numeric(roads["lat"], errors="coerce")
    roads["lon"] = pd.to_numeric(roads["lon"], errors="coerce")

    if "name" not in roads.columns:
        roads["name"] = ""
    if "lrp" not in roads.columns:
        roads["lrp"] = ""

    roads["name"] = roads["name"].fillna("").astype(str)
    roads["lrp"] = roads["lrp"].fillna("").astype(str)

    # Drop incomplete rows before filtering so downstream geometry checks can
    # assume each road point has valid coordinates and chainage.
    roads = roads.dropna(subset=["road", "chainage", "lat", "lon"]).copy()
    roads = roads[roads["road"].isin(SELECTED_ROADS)].copy()
    roads = roads.sort_values(["road", "chainage"]).reset_index(drop=True)
    return roads


def get_best_pair_match(a: pd.DataFrame, b: pd.DataFrame) -> dict | None:
    """Find the closest sampled-point pair between two roads."""
    a_pts = a[["lat", "lon"]].to_numpy(dtype=float)
    b_pts = b[["lat", "lon"]].to_numpy(dtype=float)

    best = {
        "dist_m": np.inf,
        "a_idx": None,
        "b_idx": None,
    }

    for i, (alat, alon) in enumerate(a_pts):
        # For each point on road A, compare it to all sampled points on road B
        # and keep only the globally closest pair.
        d = haversine_m(alat, alon, b_pts[:, 0], b_pts[:, 1])
        j = int(np.argmin(d))
        if float(d[j]) < best["dist_m"]:
            best["dist_m"] = float(d[j])
            best["a_idx"] = i
            best["b_idx"] = j

    if best["dist_m"] > INTERSECTION_THRESHOLD_M:
        return None

    a_row = a.iloc[best["a_idx"]]
    b_row = b.iloc[best["b_idx"]]

    # midpoint between closest sampled points
    lat_mid = float((a_row["lat"] + b_row["lat"]) / 2.0)
    lon_mid = float((a_row["lon"] + b_row["lon"]) / 2.0)

    return {
        "road_a": str(a_row["road"]),
        "road_b": str(b_row["road"]),
        "chainage_a": float(a_row["chainage"]),
        "chainage_b": float(b_row["chainage"]),
        "lat_a": float(a_row["lat"]),
        "lon_a": float(a_row["lon"]),
        "lat_b": float(b_row["lat"]),
        "lon_b": float(b_row["lon"]),
        "lat": lat_mid,
        "lon": lon_mid,
        "dist_m": float(best["dist_m"]),
        "name_a": str(a_row.get("name", "")),
        "name_b": str(b_row.get("name", "")),
        "lrp_a": str(a_row.get("lrp", "")),
        "lrp_b": str(b_row.get("lrp", "")),
    }


def cluster_intersections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge near-duplicate intersections into one.
    Simple greedy clustering by centroid distance.
    """
    if df.empty:
        return df.copy()

    clusters = []

    for _, row in df.iterrows():
        assigned = False

        for cluster in clusters:
            d = haversine_m(
                row["lat"], row["lon"],
                cluster["lat"], cluster["lon"]
            )
            if d <= CLUSTER_THRESHOLD_M:
                cluster["members"].append(row.to_dict())

                # Recompute the centroid after each added member so the cluster
                # location tracks the average of all contributing matches.
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

    out_rows = []
    next_id = INTERSECTION_ID_START

    for cluster in clusters:
        members = cluster["members"]

        connected_roads = sorted(
            set([m["road_a"] for m in members] + [m["road_b"] for m in members])
        )

        pair_labels = sorted(
            set(f"{min(m['road_a'], m['road_b'])}-{max(m['road_a'], m['road_b'])}" for m in members)
        )

        out_rows.append({
            "id": next_id,
            "road": connected_roads[0],   # placeholder for model.py compatibility
            "lrp": "",
            "chainage": np.nan,
            "lat": cluster["lat"],
            "lon": cluster["lon"],
            "name": f"INT_{'_'.join(connected_roads)}",
            "model_type": "intersection",
            "structure_type": "",
            "length": 0.0,
            "condition": "N/A",
            "connected_roads": "|".join(connected_roads),
            "road_pairs": "|".join(pair_labels),
            "n_matches": len(members),
            "min_pair_dist_m": float(min(m["dist_m"] for m in members)),
            "mean_pair_dist_m": float(np.mean([m["dist_m"] for m in members])),
        })
        next_id += 1

    return pd.DataFrame(out_rows)


def detect_intersections(roads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect pairwise road intersections and cluster nearby matches."""
    raw_matches = []

    for road_a, road_b in itertools.combinations(SELECTED_ROADS, 2):
        a = roads[roads["road"] == road_a].sort_values("chainage")
        b = roads[roads["road"] == road_b].sort_values("chainage")

        if a.empty or b.empty:
            continue

        match = get_best_pair_match(a, b)
        if match is not None:
            raw_matches.append(match)

    raw_df = pd.DataFrame(raw_matches)
    clustered_df = cluster_intersections(raw_df)

    return raw_df, clustered_df


def main():
    """Run the intersection-detection workflow and write both output CSV files."""
    roads = load_roads(INPUT_ROADS)
    raw_df, clustered_df = detect_intersections(roads)

    raw_out = OUTPUT_INTERSECTIONS.with_name("intersections_raw_matches.csv")
    raw_df.to_csv(raw_out, index=False)
    clustered_df.to_csv(OUTPUT_INTERSECTIONS, index=False)

    print(f"Wrote raw matches to: {raw_out.resolve()}")
    print(f"Wrote clustered intersections to: {OUTPUT_INTERSECTIONS.resolve()}")

    print("\nRaw pair matches:")
    if raw_df.empty:
        print("No candidate intersections found.")
    else:
        print(raw_df[[
            "road_a", "road_b", "chainage_a", "chainage_b", "dist_m", "lat", "lon"
        ]].sort_values("dist_m").to_string(index=False))

    print("\nFinal clustered intersections:")
    if clustered_df.empty:
        print("No clustered intersections found.")
    else:
        print(clustered_df[[
            "id", "name", "connected_roads", "road_pairs",
            "lat", "lon", "min_pair_dist_m", "n_matches"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
