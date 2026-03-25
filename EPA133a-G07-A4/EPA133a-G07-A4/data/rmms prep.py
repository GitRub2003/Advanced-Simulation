from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

from road_selection import load_selected_roads

BASE_DIR = Path(__file__).resolve().parent
RMMS_DIR = Path("RMMS")
OUTPUT_RAW = BASE_DIR / "rmms_traffic_raw.csv"
OUTPUT_MISSING = BASE_DIR / "rmms_missing_files.csv"


def normalize_column_name(name: str) -> str:
    """
    Normalize column names so matching becomes easier.
    """
    name = str(name).strip().lower()
    name = name.replace("(", "").replace(")", "")
    name = name.replace(".", "")
    name = name.replace("-", " ")
    name = name.replace("/", " ")
    name = re.sub(r"\s+", "_", name)
    return name


def find_matching_column(columns: list[str], keywords: list[str]) -> str | None:
    """
    Return the first column that contains all given keywords.
    """
    for col in columns:
        if all(keyword in col for keyword in keywords):
            return col
    return None


def identify_truck_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Identify the heavy, medium, and small truck columns.
    """
    cols = list(df.columns)

    heavy_col = find_matching_column(cols, ["heavy", "truck"])
    medium_col = find_matching_column(cols, ["medium", "truck"])
    small_col = find_matching_column(cols, ["small", "truck"])

    missing = []
    if heavy_col is None:
        missing.append("Heavy Truck")
    if medium_col is None:
        missing.append("Medium Truck")
    if small_col is None:
        missing.append("Small Truck")

    if missing:
        raise ValueError(
            f"Could not find RMMS truck columns: {missing}. Available columns: {cols}"
        )

    return {
        "heavy_truck": heavy_col,
        "medium_truck": medium_col,
        "small_truck": small_col,
    }


def add_combined_truck_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standardized truck columns and one combined truck column:
    total_trucks = heavy_truck + medium_truck + small_truck
    """
    out = df.copy()

    truck_cols = identify_truck_columns(out)

    heavy_col = truck_cols["heavy_truck"]
    medium_col = truck_cols["medium_truck"]
    small_col = truck_cols["small_truck"]

    out[heavy_col] = pd.to_numeric(out[heavy_col], errors="coerce").fillna(0.0)
    out[medium_col] = pd.to_numeric(out[medium_col], errors="coerce").fillna(0.0)
    out[small_col] = pd.to_numeric(out[small_col], errors="coerce").fillna(0.0)

    out["heavy_truck"] = out[heavy_col]
    out["medium_truck"] = out[medium_col]
    out["small_truck"] = out[small_col]
    out["total_trucks"] = out["heavy_truck"] + out["medium_truck"] + out["small_truck"]

    return out


def find_traffic_table(soup: BeautifulSoup):
    """
    Find the RMMS table that contains traffic columns.
    """
    tables = soup.find_all("table")

    for table in tables:
        text = " ".join(table.get_text(" ", strip=True).split())
        if "Heavy Truck" in text and "Medium Truck" in text and "Small Truck" in text:
            return table

    raise ValueError("Could not find the RMMS traffic table in the HTML file.")


def extract_rows_from_table(table) -> list[list[str]]:
    """
    Extract all rows as lists of cell text.
    """
    rows = []

    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    return rows


def build_column_names(header_row_1: list[str], header_row_2: list[str]) -> list[str]:
    """
    Build useful column names from the two RMMS header rows.
    """
    # This RMMS structure is stable enough to define explicitly.
    columns = [
        "link_no",
        "name",
        "start_lrp",
        "start_offset",
        "start_chainage",
        "end_lrp",
        "end_offset",
        "end_chainage",
        "length_km",
        "heavy_truck",
        "medium_truck",
        "small_truck",
        "large_bus",
        "medium_bus",
        "micro_bus",
        "utility",
        "car",
        "auto_rickshaw",
        "motor_cycle",
        "bi_cycle",
        "cycle_rickshaw",
        "cart",
        "motorized",
        "non_motorized",
        "total_aadt",
        "total_traffic_aadt",
    ]

    return [normalize_column_name(col) for col in columns]


def load_one_rmms_file(road: str, rmms_dir: Path = RMMS_DIR, debug: bool = False) -> pd.DataFrame:
    """
    Load one RMMS traffic HTML file by manually parsing the table with BeautifulSoup.
    """
    file_path = rmms_dir / f"{road}.traffic.htm"

    if not file_path.exists():
        raise FileNotFoundError(f"RMMS file not found for {road}: {file_path}")

    html = file_path.read_text(encoding="latin1", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    table = find_traffic_table(soup)
    rows = extract_rows_from_table(table)

    if debug:
        print(f"\nDEBUG FOR {road}")
        print(f"Number of extracted rows: {len(rows)}")
        for i, row in enumerate(rows[:8]):
            print(f"row {i}: {row}")

    if len(rows) < 7:
        raise ValueError(f"Unexpected RMMS table structure in {file_path}")

    # Based on your inspection:
    # row 4 = grouped header row
    # row 5 = actual detailed header row
    # row 6 onward = data
    header_row_1 = rows[4]
    header_row_2 = rows[5]
    data_rows = rows[6:]

    columns = build_column_names(header_row_1, header_row_2)

    cleaned_rows = []
    for row in data_rows:
        # keep only rows that look like actual data rows
        if len(row) >= 9 and str(row[0]).strip():
            cleaned_rows.append(row[:len(columns)])

    df = pd.DataFrame(cleaned_rows, columns=columns[:len(cleaned_rows[0])] if cleaned_rows else columns)

    # Make sure all expected columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[columns]
    df["road"] = road

    return df


def load_rmms_for_selected_roads(
    rmms_dir: Path = RMMS_DIR,
    include_support_roads: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RMMS traffic data for all selected roads.
    """
    roads = load_selected_roads(include_support_roads=include_support_roads)

    all_frames = []
    missing_rows = []

    for road in roads:
        file_path = rmms_dir / f"{road}.traffic.htm"

        if not file_path.exists():
            missing_rows.append({
                "road": road,
                "expected_file": str(file_path),
                "status": "missing",
            })
            continue

        try:
            df = load_one_rmms_file(road, rmms_dir=rmms_dir, debug=False)
            df = add_combined_truck_column(df)
            df["source_file"] = file_path.name
            all_frames.append(df)
            print(f"[OK] {road}: loaded {len(df)} rows")
        except Exception as e:
            missing_rows.append({
                "road": road,
                "expected_file": str(file_path),
                "status": f"error: {e}",
            })
            print(f"[ERROR] {road}: {e}")

    if all_frames:
        traffic_df = pd.concat(all_frames, ignore_index=True)
    else:
        traffic_df = pd.DataFrame()

    missing_df = pd.DataFrame(missing_rows)

    return traffic_df, missing_df


def inspect_one_file(road: str, rmms_dir: Path = RMMS_DIR) -> None:
    """
    Inspect one RMMS file and print a preview.
    """
    df = load_one_rmms_file(road, rmms_dir=rmms_dir, debug=True)
    df = add_combined_truck_column(df)

    print("\nColumns:")
    print(df.columns.tolist())

    preview_cols = [
        "road",
        "link_no",
        "name",
        "start_lrp",
        "start_chainage",
        "end_lrp",
        "end_chainage",
        "length_km",
        "heavy_truck",
        "medium_truck",
        "small_truck",
        "total_trucks",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]

    print("\nPreview:")
    print(df[preview_cols].head(10).to_string(index=False))


def main() -> None:
    traffic_df, missing_df = load_rmms_for_selected_roads()

    if not traffic_df.empty:
        traffic_df.to_csv(OUTPUT_RAW, index=False)
        print(f"\nWrote traffic data to: {OUTPUT_RAW.resolve()}")
        print(f"Rows: {len(traffic_df)}")

        preview_cols = [
            "road",
            "link_no",
            "heavy_truck",
            "medium_truck",
            "small_truck",
            "total_trucks",
        ]
        preview_cols = [col for col in preview_cols if col in traffic_df.columns]

        print("\nPreview:")
        print(traffic_df[preview_cols].head(10).to_string(index=False))
    else:
        print("\nNo RMMS traffic data loaded.")

    if not missing_df.empty:
        missing_df.to_csv(OUTPUT_MISSING, index=False)
        print(f"\nWrote missing/error overview to: {OUTPUT_MISSING.resolve()}")
        print(missing_df.to_string(index=False))


if __name__ == "__main__":
    main()

    # When that works, comment the line above and use:
    # main()