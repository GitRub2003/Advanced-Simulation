from pathlib import Path
import numpy as np
import pandas as pd

# ---- config ----
INPUT_ROADS = Path("_roads3.csv")
OUTPUT_CANDIDATES = Path("side_road_candidates.csv")

MAIN_ROADS = {"N1", "N2"}
MIN_SIDE_ROAD_KM = 25.0
MAX_ROAD_DIST_M = 500.0


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
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")


def prepare_roads_df(path: Path) -> pd.DataFrame:
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

    roads = roads.dropna(subset=["road", "chainage", "lat", "lon"]).copy()
    roads = roads.sort_values(["road", "chainage"]).reset_index(drop=True)
    return roads


def get_road_summary(roads: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for road_name, g in roads.groupby("road", sort=True):
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


def build_main_road_reference(roads: pd.DataFrame, main_roads: set[str]) -> np.ndarray:
    main_df = roads[roads["road"].isin(main_roads)].copy()
    if main_df.empty:
        raise ValueError(f"No rows found for main roads {main_roads}")

    return main_df[["lat", "lon"]].dropna().to_numpy(dtype=float)


def road_min_distance_to_main(road_points: np.ndarray, main_coords: np.ndarray) -> float:
    """
    Minimum distance in meters between any point on a candidate road
    and any point on N1/N2.
    """
    min_dist = np.inf

    for lat, lon in road_points:
        d = haversine_m(lat, lon, main_coords[:, 0], main_coords[:, 1])
        local_min = np.min(d)
        if local_min < min_dist:
            min_dist = local_min

    return float(min_dist)


def select_side_roads(roads: pd.DataFrame) -> pd.DataFrame:
    road_summary = get_road_summary(roads)
    main_coords = build_main_road_reference(roads, MAIN_ROADS)

    candidates = road_summary[
        road_summary["road"].str.startswith("N")
        & (~road_summary["road"].isin(MAIN_ROADS))
    ].copy()

    if candidates.empty:
        raise ValueError("No N-road candidates found.")

    candidates["passes_length"] = candidates["length_km"] > MIN_SIDE_ROAD_KM

    min_dists = []
    for road_name in candidates["road"]:
        g = roads[roads["road"] == road_name].sort_values("chainage")
        road_points = g[["lat", "lon"]].dropna().to_numpy(dtype=float)
        min_dist = road_min_distance_to_main(road_points, main_coords)
        min_dists.append(min_dist)

    candidates["min_dist_to_main_m"] = min_dists
    candidates["connected_by_distance"] = candidates["min_dist_to_main_m"] <= MAX_ROAD_DIST_M

    candidates["selected"] = (
        candidates["passes_length"] & candidates["connected_by_distance"]
    )

    candidates["reason"] = np.where(
        candidates["selected"],
        "length>25km and road comes close to N1/N2",
        "not selected"
    )

    candidates = candidates.sort_values(
        by=["selected", "length_km"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return candidates


def main():
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
            "road", "length_km", "min_dist_to_main_m", "reason"
        ]].to_string(index=False))

    print("\nAll candidates:")
    print(candidates[[
        "road", "length_km", "min_dist_to_main_m", "selected"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()