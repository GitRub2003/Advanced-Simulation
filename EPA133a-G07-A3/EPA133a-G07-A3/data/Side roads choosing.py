from pathlib import Path
import numpy as np
import pandas as pd

from road_selection import BASE_ROADS, REQUIRED_ROADS, find_connector_roads, normalize_roads

# ---- config ----
INPUT_ROADS = Path("_roads3.csv")
OUTPUT_CANDIDATES = Path("side_road_candidates.csv")

MAIN_ROADS = set(BASE_ROADS)
MIN_SIDE_ROAD_KM = 25.0
MAX_ROAD_DIST_M = 500.0


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
    """Raise an error when a required column is missing from a dataframe."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")


def prepare_roads_df(path: Path) -> pd.DataFrame:
    """Load the road CSV, validate required columns, and normalize key fields."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    roads = pd.read_csv(path)
    _has_cols(roads, ["road", "chainage", "lat", "lon"], "roads")

    roads = roads.copy()
    roads["road"] = roads["road"].astype(str).str.strip().str.upper()
    roads["chainage"] = pd.to_numeric(roads["chainage"], errors="coerce")
    roads["lat"] = pd.to_numeric(roads["lat"], errors="coerce")
    roads["lon"] = pd.to_numeric(roads["lon"], errors="coerce")

    if "name" not in roads.columns:
        roads["name"] = ""
    roads["name"] = roads["name"].fillna("").astype(str)

    # Drop incomplete rows so downstream distance and length calculations
    # can assume numeric coordinates and chainage values are present.
    roads = roads.dropna(subset=["road", "chainage", "lat", "lon"]).copy()
    roads = roads.sort_values(["road", "chainage"]).reset_index(drop=True)
    return roads


def get_road_summary(roads: pd.DataFrame) -> pd.DataFrame:
    """Summarize each road using its first and last chainage point."""
    rows = []

    for road_name, g in roads.groupby("road", sort=True):
        # Duplicate chainage values do not change the road extent, so only the
        # first occurrence is needed for start/end based summary metrics.
        g = g.sort_values("chainage").drop_duplicates(subset=["chainage"], keep="first").reset_index(drop=True)
        if len(g) < 2:
            continue

        first = g.iloc[0]
        last = g.iloc[-1]

        rows.append({
            "road": road_name,
            "length_km": float(last["chainage"] - first["chainage"]),
            "start_lat": float(first["lat"]),
            "start_lon": float(first["lon"]),
            "end_lat": float(last["lat"]),
            "end_lon": float(last["lon"]),
            "start_name": str(first.get("name", "")),
            "end_name": str(last.get("name", "")),
        })

    return pd.DataFrame(rows)


def ensure_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Add columns that may be absent for manually included or connector roads."""
    work = summary.copy()

    if "length_km" not in work.columns:
        work["length_km"] = np.nan
    if "min_dist_to_main_m" not in work.columns:
        work["min_dist_to_main_m"] = np.nan
    if "passes_length" not in work.columns:
        work["passes_length"] = False
    if "connected_by_distance" not in work.columns:
        work["connected_by_distance"] = False

    return work


def build_main_road_reference(roads: pd.DataFrame, main_roads: set[str]) -> np.ndarray:
    """Extract coordinate points for the main roads used as the distance reference."""
    main_df = roads[roads["road"].isin(main_roads)].copy()
    if main_df.empty:
        raise ValueError(f"No rows found for main roads {main_roads}")

    return main_df[["lat", "lon"]].dropna().to_numpy(dtype=float)


def road_min_distance_to_main(road_points: np.ndarray, main_coords: np.ndarray) -> float:
    """Return the minimum distance in meters from one road to the main-road network."""
    min_dist = np.inf

    for lat, lon in road_points:
        d = haversine_m(lat, lon, main_coords[:, 0], main_coords[:, 1])
        local_min = np.min(d)
        if local_min < min_dist:
            min_dist = local_min

    return float(min_dist)


def select_side_roads(roads: pd.DataFrame) -> pd.DataFrame:
    """Filter N-roads to side-road candidates and flag the selected ones."""
    road_summary = get_road_summary(roads)
    main_coords = build_main_road_reference(roads, MAIN_ROADS)

    candidates = road_summary[
        road_summary["road"].str.startswith("N")
        & (~road_summary["road"].isin(MAIN_ROADS))
    ].copy()

    if candidates.empty:
        raise ValueError("No N-road candidates found.")

    candidates["passes_length"] = candidates["length_km"] > MIN_SIDE_ROAD_KM

    # Compute the closest approach to the main roads for each candidate. This
    # keeps the expensive point-to-point distance check in one place.
    road_points_by_name = {
        road_name: group[["lat", "lon"]].dropna().to_numpy(dtype=float)
        for road_name, group in roads.groupby("road", sort=False)
    }
    candidates["min_dist_to_main_m"] = candidates["road"].map(
        lambda road_name: road_min_distance_to_main(road_points_by_name[road_name], main_coords)
    )
    candidates["connected_by_distance"] = candidates["min_dist_to_main_m"] <= MAX_ROAD_DIST_M

    # A road is selected only if it is both long enough and sufficiently close
    # to the N1/N2 corridor to count as connected.
    candidates["selected"] = (
        candidates["passes_length"] & candidates["connected_by_distance"]
    )

    candidates["reason"] = np.where(
        candidates["selected"],
        "length>25km and road comes close to N1/N2",
        "not selected"
    )

    candidates["selection_source"] = np.where(
        candidates["selected"],
        "distance_rule",
        "distance_rule_rejected"
    )

    road_summary_by_name = road_summary.set_index("road", drop=False)

    manual_rows = []
    for road_name in normalize_roads(REQUIRED_ROADS):
        if road_name in MAIN_ROADS or road_name not in road_summary_by_name.index:
            continue

        row = ensure_summary_columns(road_summary_by_name.loc[[road_name]]).iloc[0].to_dict()
        row["passes_length"] = bool(row["length_km"] > MIN_SIDE_ROAD_KM) if pd.notna(row["length_km"]) else False
        row["selected"] = True
        row["reason"] = "required road for the expanded network"
        row["selection_source"] = "required"
        manual_rows.append(row)

    selected_seed_roads = normalize_roads(
        list(MAIN_ROADS)
        + candidates.loc[candidates["selected"], "road"].tolist()
        + [row["road"] for row in manual_rows]
    )
    connector_roads = find_connector_roads(roads, list(REQUIRED_ROADS))

    connector_rows = []
    for road_name in connector_roads:
        if road_name in MAIN_ROADS or road_name in selected_seed_roads:
            continue
        if road_name not in road_summary_by_name.index:
            continue

        row = ensure_summary_columns(road_summary_by_name.loc[[road_name]]).iloc[0].to_dict()
        row["passes_length"] = bool(row["length_km"] > MIN_SIDE_ROAD_KM) if pd.notna(row["length_km"]) else False
        row["selected"] = True
        row["reason"] = "explicit metadata connection to the selected network"
        row["selection_source"] = "connector"
        connector_rows.append(row)

    extra = pd.DataFrame(manual_rows + connector_rows)
    if not extra.empty:
        candidates = pd.concat([candidates, extra], ignore_index=True)
        candidates = candidates.drop_duplicates(subset=["road"], keep="last")

    candidates = candidates.sort_values(
        by=["selected", "selection_source", "length_km", "road"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    return candidates


def main():
    """Run the side-road selection workflow and write the candidate CSV."""
    roads = prepare_roads_df(INPUT_ROADS)
    candidates = select_side_roads(roads)

    candidates.to_csv(OUTPUT_CANDIDATES, index=False)

    print(f"Wrote {OUTPUT_CANDIDATES.resolve()}")
    print("\nSelected side roads:")
    selected = candidates[candidates["selected"]].copy()
    if selected.empty:
        print("No side roads selected with current threshold.")
    else:
        print(selected[[
            "road", "length_km", "min_dist_to_main_m", "selection_source", "reason"
        ]].to_string(index=False))

    print("\nAll candidates:")
    print(candidates[[
        "road", "length_km", "min_dist_to_main_m", "selection_source", "selected"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
