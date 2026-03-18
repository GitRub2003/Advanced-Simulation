from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from road_selection import extract_road_references, load_selected_roads

INPUT_ROADS = Path("_roads3.csv")
OUTPUT_INTERSECTIONS = Path("intersections_detected.csv")

INTERSECTION_THRESHOLD_M = 120.0
PAIR_CLUSTER_THRESHOLD_M = 120.0
CLUSTER_THRESHOLD_M = 80.0
REFERENCE_ENDPOINT_THRESHOLD_M = 1000.0
INTERSECTION_ID_START = 9_000_000


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


def _has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


SELECTED_ROADS = load_selected_roads()


def load_roads(path: Path) -> pd.DataFrame:
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

    roads = roads.dropna(subset=["road", "chainage", "lat", "lon"]).copy()
    roads = roads[roads["road"].isin(SELECTED_ROADS)].copy()
    roads = roads.sort_values(["road", "chainage"]).reset_index(drop=True)
    return roads


def build_match_record(a_row: pd.Series, b_row: pd.Series, dist_m: float) -> dict:
    return {
        "road_a": str(a_row["road"]),
        "road_b": str(b_row["road"]),
        "chainage_a": float(a_row["chainage"]),
        "chainage_b": float(b_row["chainage"]),
        "lat_a": float(a_row["lat"]),
        "lon_a": float(a_row["lon"]),
        "lat_b": float(b_row["lat"]),
        "lon_b": float(b_row["lon"]),
        "lat": float((a_row["lat"] + b_row["lat"]) / 2.0),
        "lon": float((a_row["lon"] + b_row["lon"]) / 2.0),
        "dist_m": float(dist_m),
        "name_a": str(a_row.get("name", "")),
        "name_b": str(b_row.get("name", "")),
        "lrp_a": str(a_row.get("lrp", "")),
        "lrp_b": str(b_row.get("lrp", "")),
    }


def endpoint_reference_match(a: pd.DataFrame, b: pd.DataFrame) -> dict | None:
    """
    Fall back to endpoint metadata when the sampled geometry misses a logical
    connection that is explicitly described in the source data.
    """
    endpoints_a = pd.concat([a.head(1), a.tail(1)]).drop_duplicates().reset_index(drop=True)
    endpoints_b = pd.concat([b.head(1), b.tail(1)]).drop_duplicates().reset_index(drop=True)

    road_a = str(a.iloc[0]["road"])
    road_b = str(b.iloc[0]["road"])

    referenced_pairs: list[tuple[pd.Series, pd.Series]] = []

    for _, a_row in endpoints_a.iterrows():
        if road_b in extract_road_references(a_row.get("name", "")):
            for _, b_row in endpoints_b.iterrows():
                referenced_pairs.append((a_row, b_row))

    for _, b_row in endpoints_b.iterrows():
        if road_a in extract_road_references(b_row.get("name", "")):
            for _, a_row in endpoints_a.iterrows():
                referenced_pairs.append((a_row, b_row))

    best_match = None
    for a_row, b_row in referenced_pairs:
        dist = haversine_m(
            float(a_row["lat"]),
            float(a_row["lon"]),
            float(b_row["lat"]),
            float(b_row["lon"]),
        )
        if dist <= REFERENCE_ENDPOINT_THRESHOLD_M and (
            best_match is None or dist < best_match["dist_m"]
        ):
            best_match = build_match_record(a_row, b_row, float(dist))

    return best_match


def find_all_pair_matches(a: pd.DataFrame, b: pd.DataFrame) -> list[dict]:
    """
    Find all local close approaches between two roads, not just the single best one.
    """
    a = a.sort_values("chainage").reset_index(drop=True)
    b = b.sort_values("chainage").reset_index(drop=True)

    a_pts = a[["lat", "lon"]].to_numpy(dtype=float)
    b_pts = b[["lat", "lon"]].to_numpy(dtype=float)

    candidates = []

    for i, (alat, alon) in enumerate(a_pts):
        d = haversine_m(alat, alon, b_pts[:, 0], b_pts[:, 1])
        j = int(np.argmin(d))
        dist = float(d[j])

        if dist <= INTERSECTION_THRESHOLD_M:
            a_row = a.iloc[i]
            b_row = b.iloc[j]
            candidates.append(build_match_record(a_row, b_row, dist))

    if not candidates:
        fallback = endpoint_reference_match(a, b)
        return [fallback] if fallback is not None else []

    # cluster nearby candidates for this road pair into distinct intersections
    clusters = []
    for row in candidates:
        assigned = False
        for cluster in clusters:
            d = haversine_m(row["lat"], row["lon"], cluster["lat"], cluster["lon"])
            if d <= PAIR_CLUSTER_THRESHOLD_M:
                cluster["members"].append(row)
                cluster["lat"] = float(np.mean([m["lat"] for m in cluster["members"]]))
                cluster["lon"] = float(np.mean([m["lon"] for m in cluster["members"]]))
                assigned = True
                break

        if not assigned:
            clusters.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "members": [row],
            })

    out = []
    for cluster in clusters:
        members = cluster["members"]
        best = min(members, key=lambda m: m["dist_m"])
        best["lat"] = float(np.mean([m["lat"] for m in members]))
        best["lon"] = float(np.mean([m["lon"] for m in members]))
        out.append(best)

    return out


def cluster_intersections(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    clusters = []

    for _, row in df.iterrows():
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

    out_rows = []
    next_id = INTERSECTION_ID_START

    for cluster in clusters:
        members = cluster["members"]
        connected_roads = sorted(set([m["road_a"] for m in members] + [m["road_b"] for m in members]))
        pair_labels = sorted(set(
            f"{min(m['road_a'], m['road_b'])}-{max(m['road_a'], m['road_b'])}"
            for m in members
        ))

        out_rows.append({
            "id": next_id,
            "road": connected_roads[0],
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
    raw_matches = []

    available_roads = sorted(roads["road"].dropna().astype(str).str.upper().unique())

    for road_a, road_b in itertools.combinations(available_roads, 2):
        a = roads[roads["road"] == road_a].sort_values("chainage")
        b = roads[roads["road"] == road_b].sort_values("chainage")

        if a.empty or b.empty:
            continue

        matches = find_all_pair_matches(a, b)
        raw_matches.extend(matches)

    raw_df = pd.DataFrame(raw_matches)
    clustered_df = cluster_intersections(raw_df)

    return raw_df, clustered_df


def main():
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
        ]].sort_values(["road_a", "road_b", "chainage_a"]).to_string(index=False))

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
