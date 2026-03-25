from pathlib import Path
import numpy as np
import pandas as pd

from road_selection import load_selected_roads

INPUT_ROADS = Path("_roads3.csv")
BMMS_XLSX = Path("BMMS_overview.xlsx")
RMMS_TRAFFIC_CSV = Path("rmms_traffic_raw.csv")
OUTPUT = Path("road_objects.csv")

SELECTED_ROADS = load_selected_roads()
MAX_CHAINAGE_DIFF_KM = 1
MAX_DIST_M = 500
START_SOURCE_TRAFFIC_SIDE = "L"
END_SOURCE_TRAFFIC_SIDE = "R"


def haversine_m(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in meters between two coordinates."""
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def detect_structure_type(row: pd.Series) -> str:
    """Infer a structure type from the row type and name fields."""
    t = str(row.get("type", "")).lower()
    n = str(row.get("name", "")).lower()

    if "bridge" in t or "bridge" in n:
        return "Bridge"
    if "culvert" in t or "culvert" in n:
        if "box" in n or "box" in t:
            return "Box Culvert"
        return "Culvert"
    return ""


def _has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Raise an error if a required set of columns is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")


def normalize_lrp(s: pd.Series) -> pd.Series:
    """Normalize LRP values so matching is case- and whitespace-insensitive."""
    return s.astype(str).str.strip().str.upper()


def load_rmms_source_truck_lookup(path: Path) -> dict[str, dict[str, float]]:
    """
    Build a per-road lookup of one-direction truck AADT for the first and last links.

    The current network represents each road as one line with a source/sink at each end.
    To avoid counting inbound and outbound traffic together, one carriageway is mapped to
    the road start and the opposite carriageway is mapped to the road end.
    """
    if not path.exists():
        raise FileNotFoundError(f"RMMS traffic CSV not found: {path.resolve()}")

    traffic = pd.read_csv(path).copy()
    required_cols = ["road", "link_no", "total_trucks", "start_chainage", "end_chainage"]
    _has_cols(traffic, required_cols, "rmms_traffic_raw")

    traffic["road"] = traffic["road"].astype(str).str.strip().str.upper()
    traffic["link_no"] = traffic["link_no"].astype(str).str.strip().str.upper()
    traffic["total_trucks"] = pd.to_numeric(traffic["total_trucks"], errors="coerce").fillna(0.0)
    traffic["start_chainage"] = pd.to_numeric(traffic["start_chainage"], errors="coerce")
    traffic["end_chainage"] = pd.to_numeric(traffic["end_chainage"], errors="coerce")
    traffic = traffic.dropna(subset=["start_chainage", "end_chainage"]).copy()

    lookup: dict[str, dict[str, float]] = {}

    for road_name, group in traffic.groupby("road", sort=False):
        group = group.copy()
        group["side"] = group["link_no"].str.extract(r"([LR])$", expand=False)

        start_candidates = group.sort_values(["start_chainage", "end_chainage", "link_no"])
        end_candidates = group.sort_values(["end_chainage", "start_chainage", "link_no"], ascending=[False, False, True])

        start_row = start_candidates[start_candidates["side"] == START_SOURCE_TRAFFIC_SIDE].head(1)
        end_row = end_candidates[end_candidates["side"] == END_SOURCE_TRAFFIC_SIDE].head(1)

        if start_row.empty:
            start_row = start_candidates[start_candidates["side"].isna()].head(1)
        if end_row.empty:
            end_row = end_candidates[end_candidates["side"].isna()].head(1)

        if start_row.empty or end_row.empty:
            raise ValueError(
                f"Could not map RMMS source traffic sides for road {road_name}. "
                f"Expected start side {START_SOURCE_TRAFFIC_SIDE} and end side {END_SOURCE_TRAFFIC_SIDE}, "
                f"or unsuffixed link numbers."
            )

        lookup[road_name] = {
            "start_total_trucks": float(start_row.iloc[0]["total_trucks"]),
            "end_total_trucks": float(end_row.iloc[0]["total_trucks"]),
            "start_link_no": str(start_row.iloc[0]["link_no"]),
            "end_link_no": str(end_row.iloc[0]["link_no"]),
        }

    return lookup


def process_one_road(
    roads_all: pd.DataFrame,
    bmms_all: pd.DataFrame,
    road_name: str,
    rmms_lookup: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build the model-ready road-object rows for one road."""
    roads = roads_all[roads_all["road"] == road_name].copy()
    if roads.empty:
        raise ValueError(f"No rows found for road == {road_name}")

    roads = roads.sort_values("chainage").reset_index(drop=True)

    if len(roads) < 2:
        raise ValueError(f"Not enough points for {road_name}")

    if "name" not in roads.columns:
        roads["name"] = ""
    roads["name"] = roads["name"].fillna("").astype(str)

    chain = roads["chainage"].to_numpy(dtype=float)
    seg_km = np.diff(chain, append=chain[-1])
    seg_m = np.maximum(seg_km * 1000.0, 0.0)
    seg_m[-1] = 0.0

    # Start from the road CSV classification, then enrich bridge rows with BMMS
    # data where a suitable match can be found.
    structure_type = roads.apply(detect_structure_type, axis=1).astype(str).to_numpy()

    model_type = np.array(["link"] * len(roads), dtype=object)
    model_type[0] = "sourcesink"
    model_type[-1] = "sourcesink"

    is_structure = (structure_type != "")
    is_structure[0] = False
    is_structure[-1] = False
    model_type[is_structure] = "bridge"

    bmms = bmms_all[bmms_all["road"] == road_name].copy()

    bmms_groups = {k: g for k, g in bmms.groupby("LRPName", sort=False)}
    bm_lat = pd.to_numeric(bmms["lat"], errors="coerce").to_numpy(dtype=float)
    bm_lon = pd.to_numeric(bmms["lon"], errors="coerce").to_numpy(dtype=float)
    bm_cond = bmms["condition"].astype(str).str.upper().str.strip().to_numpy()
    bm_type = bmms["type"].astype(str).to_numpy()

    condition = np.array(["N/A"] * len(roads), dtype=object)
    structure_type_out = structure_type.copy()

    for idx, row in roads.iterrows():
        if model_type[idx] != "bridge":
            continue

        lrp = str(row["lrp"]).strip().upper()
        ch = float(row["chainage"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        matched_row = None

        if lrp in bmms_groups:
            candidates = bmms_groups[lrp].copy()
            cand_valid = candidates.dropna(subset=["chainage"]).copy()

            # Prefer a same-LRP match with a nearby chainage because this is
            # more reliable than matching only on geographic distance.
            if len(cand_valid) > 0:
                diffs = (cand_valid["chainage"] - ch).abs().to_numpy()
                best_pos = int(np.argmin(diffs))
                if float(diffs[best_pos]) <= MAX_CHAINAGE_DIFF_KM:
                    matched_row = cand_valid.iloc[best_pos]
            else:
                matched_row = candidates.iloc[0]

        if matched_row is None and len(bmms) > 0:
            # Fall back to spatial matching when the LRP name does not produce
            # a confident BMMS match.
            d = haversine_m(lat, lon, bm_lat, bm_lon)
            j = int(np.argmin(d))
            if float(d[j]) <= MAX_DIST_M:
                cond = bm_cond[j]
                condition[idx] = cond if cond in {"A", "B", "C", "D"} else "Unknown"

                bt = str(bm_type[j]).lower()
                if "culvert" in bt:
                    structure_type_out[idx] = "Box Culvert" if "box" in bt else "Culvert"
                elif "bridge" in bt:
                    structure_type_out[idx] = "Bridge"
                else:
                    structure_type_out[idx] = str(bm_type[j]).strip()
                continue

        if matched_row is not None:
            cond = str(matched_row["condition"]).strip().upper()
            condition[idx] = cond if cond in {"A", "B", "C", "D"} else "Unknown"

            bm_type_str = str(matched_row.get("type", "")).strip()
            if bm_type_str:
                bt = bm_type_str.lower()
                if "culvert" in bt:
                    structure_type_out[idx] = "Box Culvert" if "box" in bt else "Culvert"
                elif "bridge" in bt:
                    structure_type_out[idx] = "Bridge"
                else:
                    structure_type_out[idx] = bm_type_str
        else:
            condition[idx] = "Unknown"

    out = pd.DataFrame({
        "road": roads["road"].astype(str),
        "lrp": roads["lrp"].astype(str),
        "chainage": roads["chainage"].astype(float),
        "lat": roads["lat"].astype(float),
        "lon": roads["lon"].astype(float),
        "name": roads["name"].astype(str),
        "model_type": model_type,
        "structure_type": structure_type_out,
        "length": seg_m,
        "condition": condition,
    })

    out["source_total_trucks"] = np.nan
    out["source_rmms_link_no"] = ""

    traffic_info = rmms_lookup.get(road_name)
    if traffic_info is not None:
        out.loc[out.index[0], "source_total_trucks"] = traffic_info["start_total_trucks"]
        out.loc[out.index[0], "source_rmms_link_no"] = traffic_info["start_link_no"]
        out.loc[out.index[-1], "source_total_trucks"] = traffic_info["end_total_trucks"]
        out.loc[out.index[-1], "source_rmms_link_no"] = traffic_info["end_link_no"]

    return out


def main():
    """Prepare road objects for all selected roads and write the merged CSV."""
    roads_all = pd.read_csv(INPUT_ROADS)
    _has_cols(roads_all, ["road", "chainage", "lrp", "lat", "lon"], "roads")

    roads_all = roads_all.copy()
    roads_all["road"] = roads_all["road"].astype(str).str.strip().str.upper()
    roads_all["chainage"] = pd.to_numeric(roads_all["chainage"], errors="coerce")
    roads_all["lat"] = pd.to_numeric(roads_all["lat"], errors="coerce")
    roads_all["lon"] = pd.to_numeric(roads_all["lon"], errors="coerce")
    roads_all["lrp"] = normalize_lrp(roads_all["lrp"])
    roads_all = roads_all.dropna(subset=["road", "chainage", "lat", "lon", "lrp"]).copy()

    bmms_all = pd.read_excel(BMMS_XLSX, sheet_name="BMMS_overview")
    _has_cols(bmms_all, ["road", "LRPName", "condition", "chainage", "type"], "BMMS_overview")

    bmms_all = bmms_all.copy()
    bmms_all["road"] = bmms_all["road"].astype(str).str.strip().str.upper()
    bmms_all["LRPName"] = normalize_lrp(bmms_all["LRPName"])
    bmms_all["condition"] = bmms_all["condition"].astype(str).str.strip().str.upper()
    bmms_all["chainage"] = pd.to_numeric(bmms_all["chainage"], errors="coerce")
    bmms_all["type"] = bmms_all["type"].astype(str).str.strip()
    rmms_lookup = load_rmms_source_truck_lookup(RMMS_TRAFFIC_CSV)

    # Process roads independently so each road keeps its own chainage order and
    # BMMS matching is limited to records from that same road.
    pieces = []
    for road_name in SELECTED_ROADS:
        part = process_one_road(roads_all, bmms_all, road_name, rmms_lookup)
        pieces.append(part)

    out = pd.concat(pieces, ignore_index=True)
    out.insert(0, "id", np.arange(len(out), dtype=int))

    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.resolve()}")
    print(out["model_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
