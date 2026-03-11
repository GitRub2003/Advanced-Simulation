from pathlib import Path
import numpy as np
import pandas as pd

# ---- config ----
INPUT_ROADS = Path("_roads3.csv")              # Opening the correct files
BMMS_XLSX = Path("BMMS_overview.xlsx")
OUTPUT = Path("n1_model.csv")
ROAD_NAME = "N1"
MAX_CHAINAGE_DIFF_KM = 1.0   # max difference in chainage for LRP-match (in km)
MAX_DIST_M = 2000            # fallback: max dist for lat/lon match (in meters)

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
    Determines whether a row in the dataset represents a bridge or culvert.
    The function inspects the 'type' and 'name' columns of the row and
    searches for keywords that indicate infrastructure structures.
    Returns:
      - "Bridge" if row indicates a bridge
      - "Box Culvert" if row indicates a culvert/box culvert
      - "" otherwise (not a structure)
    """
    # Convert the relevant fields to lowercase strings to allow
    # case-insensitive keyword matching
    t = str(row.get("type", "")).lower()
    n = str(row.get("name", "")).lower()

    # Identify bridges by checking for the keyword "bridge"
    # in either the type or name field
    if "bridge" in t or "bridge" in n:
        return "Bridge"

    # culvert keywords
    if "culvert" in t or "culvert" in n:
        # If the word "box" is present, classify the structure
        # specifically as a box culvert
        if "box" in n or "box" in t:
            return "Box Culvert"
        return "Culvert"

    return ""

def _has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    # Checks whether the required columns exist in the dataframe
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # Raises an error if any required columns are missing
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")

def normalize_lrp(s: pd.Series) -> pd.Series:
    # Standardizes LRP values by converting them to uppercase strings and removing whitespace
    return s.astype(str).str.strip().str.upper()

def main():
    # --- Load roads ---
    # Check if the roads dataset exists before attempting to load it
    if not INPUT_ROADS.exists():
        raise FileNotFoundError(f"Could not find {INPUT_ROADS.resolve()} (put _roads3.csv in data/)")

    # Load the roads CSV file into a dataframe
    roads = pd.read_csv(INPUT_ROADS)

    # Verify that the required columns exist in the dataset
    _has_cols(roads, ["road", "chainage", "lrp", "lat", "lon"], "roads")

    # Normalize road names (remove spaces and convert to uppercase)
    # and filter the dataset to only keep rows belonging to the N1 road
    roads = roads.copy()
    roads["road"] = roads["road"].astype(str).str.strip().str.upper()
    roads = roads[roads["road"] == ROAD_NAME].copy()

    # Ensure that the filtering step produced data
    if roads.empty:
        raise ValueError(f"No rows found for road == {ROAD_NAME}")

    # Convert key columns to numeric values so they can be used
    # for sorting, distance calculations and geographic operations
    roads["chainage"] = pd.to_numeric(roads["chainage"], errors="coerce")
    roads["lat"] = pd.to_numeric(roads["lat"], errors="coerce")
    roads["lon"] = pd.to_numeric(roads["lon"], errors="coerce")

    # Standardize LRP identifiers to a consistent format
    roads["lrp"] = normalize_lrp(roads["lrp"])

    # Remove rows that have missing values in critical fields
    roads = roads.dropna(subset=["chainage", "lat", "lon"])

    # Sort the road points along the road based on chainage distance
    roads = roads.sort_values("chainage").reset_index(drop=True)

    # If multiple rows have the same chainage value,
    # keep only the first occurrence to maintain a stable route representation
    roads = roads.drop_duplicates(subset=["chainage"], keep="first").reset_index(drop=True)

    # Ensure there are enough points to form at least one road segment
    if len(roads) < 2:
        raise ValueError("Not enough N1 points after cleaning (need at least 2).")

    # Ensure the "name" column exists (some rows may not contain it)
    if "name" not in roads.columns:
        roads["name"] = ""

    # Replace missing values in the name column and ensure string format
    roads["name"] = roads["name"].fillna("").astype(str)

    # --- Compute segment lengths (meters) ---
    # Extract chainage values and compute the distance between consecutive points
    chain = roads["chainage"].to_numpy(dtype=float)

    # Calculate segment lengths in kilometers between consecutive chainage values
    seg_km = np.diff(chain, append=chain[-1])

    # Convert segment lengths to meters and prevent negative values
    seg_m = np.maximum(seg_km * 1000.0, 0.0)

    # The final node represents the end of the road and therefore has no outgoing segment
    seg_m[-1] = 0.0

    # --- Determine structure_type for each row (Bridge / Box Culvert / Culvert / "") ---
    # Apply the structure detection function to identify infrastructure structures
    structure_type = roads.apply(detect_structure_type, axis=1).astype(str).to_numpy()

    # --- Assign model_type ---
    # Initialize all rows as "link" elements in the simulation network
    model_type = np.array(["link"] * len(roads), dtype=object)

    # The first node is the traffic source and the last node is the sink
    model_type[0] = "source"
    model_type[-1] = "sink"

    # Identify rows that represent infrastructure structures
    is_structure = (structure_type != "")

    # Prevent the first and last nodes from being classified as structures
    is_structure[0] = False
    is_structure[-1] = False

    # Treat bridges and culverts as "bridge" components in the simulation model
    model_type[is_structure] = "bridge"

    # --- Load BMMS and build mapping by (road, LRPName) ---
    # Check if the BMMS overview file exists
    if not BMMS_XLSX.exists():
        raise FileNotFoundError(f"Could not find {BMMS_XLSX.resolve()} (put BMMS_overview.xlsx in data/)")

    # Load the BMMS infrastructure database from Excel
    bmms = pd.read_excel(BMMS_XLSX, sheet_name="BMMS_overview")

    # Verify that all required columns are present
    _has_cols(bmms, ["road", "LRPName", "condition", "chainage", "type"], "BMMS_overview")

    # Standardize road names and filter for the N1 road
    bmms = bmms.copy()
    bmms["road"] = bmms["road"].astype(str).str.strip().str.upper()
    bmms = bmms[bmms["road"] == ROAD_NAME].copy()

    # Normalize LRP identifiers so they match the format used in the roads dataset
    bmms["LRPName"] = normalize_lrp(bmms["LRPName"])

    # Standardize condition values (A/B/C/D) to uppercase strings
    bmms["condition"] = bmms["condition"].astype(str).str.strip().str.upper()

    # Convert chainage values to numeric format for distance comparison
    bmms["chainage"] = pd.to_numeric(bmms["chainage"], errors="coerce")

    # Clean and standardize structure type information
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