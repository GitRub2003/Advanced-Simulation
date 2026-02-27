from pathlib import Path
import numpy as np
import pandas as pd

# ---- config ----
INPUT_ROADS = Path("_roads3.csv")              # jouw roads file in data/
BMMS_XLSX = Path("BMMS_overview.xlsx")         # jouw BMMS file in data/
OUTPUT = Path("n1_model.csv")
ROAD_NAME = "N1"
MAX_CHAINAGE_DIFF_KM = 1.0   # max verschil in chainage voor LRP-match (in km)
MAX_DIST_M = 2000            # fallback: max afstand voor lat/lon match (in meters)

#--- Distance Function
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
# ---- detection helpers ----
def detect_structure_type(row: pd.Series) -> str:
    """
    Returns:
      - "Bridge" if row indicates a bridge
      - "Box Culvert" if row indicates a culvert/box culvert
      - "" otherwise (not a structure)
    """
    t = str(row.get("type", "")).lower()
    n = str(row.get("name", "")).lower()

    # bridge keywords
    if "bridge" in t or "bridge" in n:
        return "Bridge"

    # culvert keywords (roads file often has type='Culvert' and name='Box Culvert')
    if "culvert" in t or "culvert" in n:
        # if you want to distinguish box culvert vs other culverts:
        if "box" in n or "box" in t:
            return "Box Culvert"
        return "Culvert"

    return ""

def _has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")

def normalize_lrp(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def main():
    # --- Load roads ---
    if not INPUT_ROADS.exists():
        raise FileNotFoundError(f"Could not find {INPUT_ROADS.resolve()} (put _roads3.csv in data/)")

    roads = pd.read_csv(INPUT_ROADS)
    _has_cols(roads, ["road", "chainage", "lrp", "lat", "lon"], "roads")

    # Normalize + filter N1
    roads = roads.copy()
    roads["road"] = roads["road"].astype(str).str.strip().str.upper()
    roads = roads[roads["road"] == ROAD_NAME].copy()
    if roads.empty:
        raise ValueError(f"No rows found for road == {ROAD_NAME}")

    # numeric cleanup
    roads["chainage"] = pd.to_numeric(roads["chainage"], errors="coerce")
    roads["lat"] = pd.to_numeric(roads["lat"], errors="coerce")
    roads["lon"] = pd.to_numeric(roads["lon"], errors="coerce")
    roads["lrp"] = normalize_lrp(roads["lrp"])

    roads = roads.dropna(subset=["chainage", "lat", "lon"])
    roads = roads.sort_values("chainage").reset_index(drop=True)

    # keep first per chainage (simple, stable route)
    roads = roads.drop_duplicates(subset=["chainage"], keep="first").reset_index(drop=True)

    if len(roads) < 2:
        raise ValueError("Not enough N1 points after cleaning (need at least 2).")

    # name optional
    if "name" not in roads.columns:
        roads["name"] = ""
    roads["name"] = roads["name"].fillna("").astype(str)

    # --- Compute segment lengths (meters) ---
    chain = roads["chainage"].to_numpy(dtype=float)
    seg_km = np.diff(chain, append=chain[-1])
    seg_m = np.maximum(seg_km * 1000.0, 0.0)
    seg_m[-1] = 0.0

    # --- Determine structure_type for each row (Bridge / Box Culvert / Culvert / "") ---
    structure_type = roads.apply(detect_structure_type, axis=1).astype(str).to_numpy()

    # --- Assign model_type ---
    # Default: link; first: source; last: sink; structures: "bridge" (we reuse Bridge component for both bridges & culverts)
    model_type = np.array(["link"] * len(roads), dtype=object)
    model_type[0] = "source"
    model_type[-1] = "sink"

    is_structure = (structure_type != "")
    is_structure[0] = False
    is_structure[-1] = False
    model_type[is_structure] = "bridge"   # <-- bridges + culverts treated as "bridge" component

    # --- Load BMMS and build mapping by (road, LRPName) ---
    if not BMMS_XLSX.exists():
        raise FileNotFoundError(f"Could not find {BMMS_XLSX.resolve()} (put BMMS_overview.xlsx in data/)")

    bmms = pd.read_excel(BMMS_XLSX, sheet_name="BMMS_overview")
    _has_cols(bmms, ["road", "LRPName", "condition", "chainage", "type"], "BMMS_overview")

    bmms = bmms.copy()
    bmms["road"] = bmms["road"].astype(str).str.strip().str.upper()
    bmms = bmms[bmms["road"] == ROAD_NAME].copy()

    bmms["LRPName"] = normalize_lrp(bmms["LRPName"])
    bmms["condition"] = bmms["condition"].astype(str).str.strip().str.upper()
    bmms["chainage"] = pd.to_numeric(bmms["chainage"], errors="coerce")
    bmms["type"] = bmms["type"].astype(str).str.strip()


    # group by LRPName (fast lookup)
    bmms_groups = {k: g for k, g in bmms.groupby("LRPName", sort=False)}
    bm_lat = pd.to_numeric(bmms["lat"], errors="coerce").to_numpy(dtype=float)
    bm_lon = pd.to_numeric(bmms["lon"], errors="coerce").to_numpy(dtype=float)
    bm_cond = bmms["condition"].astype(str).str.upper().str.strip().to_numpy()
    bm_chain = pd.to_numeric(bmms["chainage"], errors="coerce").to_numpy(dtype=float)
    bm_type = bmms["type"].astype(str).to_numpy()
    # --- Assign condition for structures only ---
    # Non-structures: N/A
    # Structures (bridges + culverts): A/B/C/D if match, else Unknown
    condition = np.array(["N/A"] * len(roads), dtype=object)

    # Optional: overwrite structure_type based on BMMS type if we find a match
    # (BMMS knows exact type: Bridge / Box Culvert etc.)
    structure_type_out = structure_type.copy()

    for idx in np.where(model_type == "bridge")[0]:
        lrp = roads.at[idx, "lrp"]
        ch = float(roads.at[idx, "chainage"])
        lat = float(roads.at[idx, "lat"])
        lon = float(roads.at[idx, "lon"])

        matched_row = None

        # --- 1) Primary: LRP match + chainage-nearest (but only if within MAX_CHAINAGE_DIFF_KM) ---
        if lrp in bmms_groups:
            candidates = bmms_groups[lrp].copy()

            # prefer candidates with chainage, choose closest
            cand_valid = candidates.dropna(subset=["chainage"]).copy()
            if len(cand_valid) > 0:
                diffs = (cand_valid["chainage"] - ch).abs().to_numpy()
                best_pos = int(np.argmin(diffs))
                best_row = cand_valid.iloc[best_pos]

                # accept only if chainage difference is not crazy
                if float(diffs[best_pos]) <= MAX_CHAINAGE_DIFF_KM:
                    matched_row = best_row
            else:
                # no chainage info; accept first candidate
                matched_row = candidates.iloc[0]

        # --- 2) Fallback: nearest by distance within MAX_DIST_M ---
        if matched_row is None:
            d = haversine_m(lat, lon, bm_lat, bm_lon)
            j = int(np.argmin(d))
            if float(d[j]) <= MAX_DIST_M:
                # use BMMS row j
                cond = bm_cond[j]
                condition[idx] = cond if cond in {"A", "B", "C", "D"} else "Unknown"

                # structure_type label from BMMS type (for report)
                bt = str(bm_type[j]).lower()
                if "culvert" in bt:
                    structure_type_out[idx] = "Box Culvert" if "box" in bt else "Culvert"
                elif "bridge" in bt:
                    structure_type_out[idx] = "Bridge"
                else:
                    structure_type_out[idx] = str(bm_type[j]).strip()
                continue
            else:
                # no acceptable distance match
                condition[idx] = "Unknown"
                continue

        # --- If primary matched_row found, set condition + structure_type ---
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

    # --- Build output ---
    out = pd.DataFrame({
        "id": np.arange(len(roads), dtype=int),
        "road": roads["road"].astype(str),
        "lrp": roads["lrp"].astype(str),
        "chainage": roads["chainage"].astype(float),
        "lat": roads["lat"].astype(float),
        "lon": roads["lon"].astype(float),
        "name": roads["name"].astype(str),
        "model_type": model_type,
        "structure_type": structure_type_out,   # <-- NEW: Bridge / Box Culvert / ...
        "length": seg_m,
        "condition": condition,                 # <-- NEW: A/B/C/D for structures, N/A for others
    })

    out.to_csv(OUTPUT, index=False)

    # --- Prints for quick validation ---
    print(f"Wrote {OUTPUT.resolve()}")
    print(f"Rows: {len(out)}")

    print("\nmodel_type counts:")
    print(out["model_type"].value_counts().to_string())

    print("\nStructure counts (rows where model_type == 'bridge') by structure_type:")
    print(out.loc[out["model_type"] == "bridge", "structure_type"].value_counts(dropna=False).to_string())

    print("\nCondition counts for structures only:")
    print(out.loc[out["model_type"] == "bridge", "condition"].value_counts(dropna=False).to_string())

    print("\nNon-structure condition check (should be only N/A):")
    print(out.loc[out["model_type"] != "bridge", "condition"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()