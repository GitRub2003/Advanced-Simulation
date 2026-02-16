import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

# --- CONFIGURATION ---
DATA_PATH = Path(r"C:\Users\larsg\data\raw\WBSIM_Lab1_2024\infrastructure")
BMMS_FILE = DATA_PATH / "BMMS_overview - Copy.xlsx"
CORRECTED_ROADS_FILE = DATA_PATH / "_roads.tsv"          # produced by your cleaning pipeline
ROADS_TO_ANALYZE = None                                   # None = all roads

# Outlier parameters (kept for optional validation)
K_MAD = 3
ABS_THRESHOLD_METERS = 2000
ABS_THRESHOLD_KM = ABS_THRESHOLD_METERS / 1000.0
MIN_RESIDUAL_PCT_POS = 0.10
MIN_RESIDUAL_PCT_NEG = 0.10

# --- UTILITIES ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in km."""
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def ratio_outlier(dist_coords, delta_chainage, min_ratio=0.2, max_ratio=1.80, max_abs_diff_km=0.3):
    """Mark outliers based on ratio, allowing for small absolute deviations."""
    if delta_chainage <= 0:
        return True
    ratio = dist_coords / delta_chainage
    abs_diff = abs(dist_coords - delta_chainage)
    ratio_outside = ratio < min_ratio or ratio > max_ratio
    return ratio_outside and abs_diff > max_abs_diff_km

def fit_lowess(group, frac=0.2, it=3):
    """Fit LOWESS smoother to a group of data."""
    fitted = lowess(
        group["dist_chainage"],
        group["dist_coords"],
        frac=frac,
        it=it,
        return_sorted=True,
    )
    return fitted[:, 1]

# --- CORRECTED ROADS LOADING ---

def wide_to_long_corrected(wide_df):
    """Convert the wide corrected roads format (from your pipeline) to long."""
    rows = []
    for _, row in wide_df.iterrows():
        road = row.iloc[0]
        seq = 0
        for j in range(1, len(row) - 2, 3):
            lrp = row.iloc[j]
            lat = row.iloc[j+1]
            lon = row.iloc[j+2]
            if pd.isna(lrp) or pd.isna(lat) or pd.isna(lon):
                continue
            rows.append({
                "road": road,
                "lrp": str(lrp).strip(),
                "lat_corrected": float(lat),
                "lon_corrected": float(lon),
                "seq": seq
            })
            seq += 1
    return pd.DataFrame(rows)

# --- DATA PROCESSING (with corrected coordinates) ---

def load_and_preprocess(file_path, roads, corrected_long):
    """Load BMMS, merge corrected coordinates, and prepare for analysis."""
    df = pd.read_excel(file_path)
    # Rename LRPName to lrp for merging
    df.rename(columns={"LRPName": "lrp"}, inplace=True)

    if roads is not None:
        df = df[df["road"].isin(roads)].copy()
    else:
        df = df.copy()

    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df = df.dropna(subset=["chainage", "lat", "lon"])

    # Merge corrected coordinates
    df = df.merge(
        corrected_long[["road", "lrp", "lat_corrected", "lon_corrected"]],
        on=["road", "lrp"],
        how="left"
    )
    # Use corrected where available, otherwise keep original
    df["lat"] = df["lat_corrected"].fillna(df["lat"])
    df["lon"] = df["lon_corrected"].fillna(df["lon"])
    df.drop(columns=["lat_corrected", "lon_corrected"], inplace=True)

    df = df.sort_values(["road", "chainage"]).reset_index(drop=True)
    return df

def analyze_segments(df):
    """Perform segment‑by‑segment analysis."""
    segment_rows = []
    for road, group in df.groupby("road"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            a = group.iloc[i - 1]
            b = group.iloc[i]

            delta_chainage = b["chainage"] - a["chainage"]
            dist_coords = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            diff = dist_coords - delta_chainage
            rel_err = diff / delta_chainage if delta_chainage else float("nan")

            segment_rows.append({
                "road": road,
                "asset1": a["lrp"],
                "asset2": b["lrp"],
                "chainage1": a["chainage"],
                "chainage2": b["chainage"],
                "lat1": a["lat"],
                "lon1": a["lon"],
                "lat2": b["lat"],
                "lon2": b["lon"],
                "delta_chainage": delta_chainage,
                "dist_coords": dist_coords,
                "diff": diff,
                "rel_err": rel_err,
            })

    seg_df = pd.DataFrame(segment_rows)
    seg_df = seg_df[seg_df["delta_chainage"] > 0].copy()
    seg_df["ratio"] = seg_df["dist_coords"] / seg_df["delta_chainage"]
    seg_df["outlier"] = seg_df.apply(
        lambda row: ratio_outlier(row["dist_coords"], row["delta_chainage"]), axis=1
    )
    return seg_df

def analyze_from_start(df):
    """Perform analysis from the start point of each road using LOWESS."""
    start_rows = []
    for road, group in df.groupby("road"):
        group = group.reset_index(drop=True)
        if group.empty:
            continue
        start = group.iloc[0]
        for _, row in group.iterrows():
            dist_chainage = row["chainage"] - start["chainage"]
            dist_coords = haversine(start["lat"], start["lon"], row["lat"], row["lon"])
            start_rows.append({
                "road": road,
                "dist_chainage": dist_chainage,
                "dist_coords": dist_coords,
            })

    start_df = pd.DataFrame(start_rows)
    start_df = start_df.sort_values(["road", "dist_coords"]).reset_index(drop=True)

    start_df["lowess_fit"] = np.nan
    for road, group in start_df.groupby("road"):
        fitted = fit_lowess(group)
        start_df.loc[group.index, "lowess_fit"] = fitted

    start_df["residual_lowess"] = start_df["dist_chainage"] - start_df["lowess_fit"]
    start_df["residual_pct"] = start_df["residual_lowess"] / start_df["lowess_fit"].replace(0, np.nan)

    start_df["outlier"] = False
    for road, group in start_df.groupby("road"):
        pct = group["residual_pct"].dropna()
        if pct.empty:
            continue
        median_pct = np.median(pct)
        mad_pct = np.median(np.abs(pct - median_pct))
        pct_threshold_pos = max(MIN_RESIDUAL_PCT_POS, median_pct + K_MAD * mad_pct)
        pct_threshold_neg = min(-MIN_RESIDUAL_PCT_NEG, median_pct - K_MAD * mad_pct)

        pos_outliers = (group["residual_lowess"] > ABS_THRESHOLD_KM) & (group["residual_pct"] > pct_threshold_pos)
        neg_outliers = (group["residual_lowess"] < -ABS_THRESHOLD_KM) & (group["residual_pct"] < pct_threshold_neg)

        outliers = pos_outliers | neg_outliers
        if not group.empty:
            outliers.iloc[0] = False   # first point is always (0,0)

        start_df.loc[group.index, "outlier"] = outliers

    return start_df

# --- PLOTTING ---

def plot_segment_analysis(seg_df, output_file="route_analysis_outliers.png"):
    """Plot segment analysis results."""
    plt.figure(figsize=(10, 6))
    for road in seg_df["road"].unique():
        subset = seg_df[seg_df["road"] == road]
        normal = subset[~subset["outlier"]]
        outliers = subset[subset["outlier"]]

        plt.scatter(normal["dist_coords"], normal["delta_chainage"], label=f"Weg {road}", alpha=0.6)
        if not outliers.empty:
            plt.scatter(
                outliers["dist_coords"],
                outliers["delta_chainage"],
                color="red", edgecolor="black", s=70, label=f"Weg {road} outliers"
            )

    max_val = max(seg_df["delta_chainage"].max(), seg_df["dist_coords"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Ideaal (y=x)")

    plt.xlabel("Afstand op basis van Coordinaten (km, segment)")
    plt.ylabel("Chainage (km, segment)")
    plt.title("Segment-analyse: Chainage vs Coordinaten (gecorrigeerde wegen)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nSegment scatterplot opgeslagen als: {output_file}")
    plt.show()

def plot_start_analysis(start_df, output_file="route_analysis_start.png"):
    """Plot start-point analysis results with LOWESS fit."""
    plt.figure(figsize=(10, 6))
    for road in start_df["road"].unique():
        subset = start_df[start_df["road"] == road]
        plt.scatter(subset["dist_coords"], subset["dist_chainage"], label=f"Weg {road}", alpha=0.6)

        outliers = subset[subset["outlier"]]
        if not outliers.empty:
            plt.scatter(
                outliers["dist_coords"], outliers["dist_chainage"],
                color="red", edgecolor="black", s=70, label=f"Weg {road} outliers"
            )

        plt.plot(
            subset["dist_coords"], subset["lowess_fit"],
            color="black", linewidth=1.5, alpha=0.8, label=f"Weg {road} fit"
        )

    max_val = max(start_df["dist_chainage"].max(), start_df["dist_coords"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Ideaal (y=x)")

    plt.xlabel("Afstand op basis van Coordinaten (km, vanaf start)")
    plt.ylabel("Chainage (km, vanaf start)")
    plt.title("Start-analyse: Chainage vs Coordinaten (gecorrigeerde wegen)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nStart-analyse scatterplot opgeslagen als: {output_file}")
    plt.show()

# --- EXPORT CORRECTED BMMS (exact same format) ---

def export_corrected_bmms(original_path, corrected_long, output_path):
    """
    Load the original BMMS Excel, replace lat/lon with corrected values,
    and save as a new Excel file with all original columns (same order),
    using the sheet name 'sheet1' as expected by the Java software.
    """
    df = pd.read_excel(original_path)
    df['LRPName'] = df['LRPName'].astype(str).str.strip()
    corrected_long['lrp'] = corrected_long['lrp'].astype(str).str.strip()

    merged = df.merge(
        corrected_long[['road', 'lrp', 'lat_corrected', 'lon_corrected']],
        left_on=['road', 'LRPName'],
        right_on=['road', 'lrp'],
        how='left'
    )

    merged['lat'] = merged['lat_corrected'].fillna(merged['lat'])
    merged['lon'] = merged['lon_corrected'].fillna(merged['lon'])
    merged.drop(columns=['lrp', 'lat_corrected', 'lon_corrected'], inplace=True)

    # Restore original column order
    original_columns = pd.read_excel(original_path, nrows=0).columns.tolist()
    merged = merged[original_columns]

    # Save with the required sheet name
    merged.to_excel(output_path, sheet_name='BMMS_overview', index=False)
    print(f"✅ Corrected BMMS saved to: {output_path} (sheet name: 'BMMS_overview')")

# --- MAIN ---

def main():
    # 1. Load corrected roads (from your previous pipeline)
    print("Loading corrected roads...")
    corrected_wide = pd.read_csv(CORRECTED_ROADS_FILE, sep="\t")
    corrected_long = wide_to_long_corrected(corrected_wide)
    print(f"Loaded {len(corrected_long)} corrected points.")

    # 2. Load BMMS and merge corrected coordinates (for analysis)
    print("Loading BMMS and merging corrected coordinates...")
    df_filtered = load_and_preprocess(BMMS_FILE, ROADS_TO_ANALYZE, corrected_long)
    print(f"Total points after merge: {len(df_filtered)}")

    # 3. Segment-analyse (now using corrected coordinates) – optional, can be skipped
    print("\nPerforming segment analysis...")
    seg_df = analyze_segments(df_filtered)
    print("\n--- Grootste afwijkingen (segmenten) ---")
    print(seg_df.sort_values("diff", ascending=False).head(10))
    print("\n--- Outliers (ratio) ---")
    print(seg_df[seg_df["outlier"]].sort_values("diff", ascending=False).head(10))
    outlier_pct = seg_df["outlier"].mean() * 100
    print(f"\nOutlier percentage (segmenten): {outlier_pct:.2f}%")

    plot_segment_analysis(seg_df)

    # 4. Start-analyse (LOWESS) – optional
    print("\nPerforming start analysis...")
    start_df = analyze_from_start(df_filtered)
    print("\n--- Start-analyse outliers (lowess, boven fit) ---")
    print(
        start_df[start_df["outlier"]]
        .sort_values("residual_lowess", ascending=False)
        .head(10)
    )

    plot_start_analysis(start_df)

    # 5. Export corrected BMMS file (exact same format as original)
    output_excel = DATA_PATH / "BMMS_overview.xlsx"
    export_corrected_bmms(BMMS_FILE, corrected_long, output_excel)

if __name__ == "__main__":
    main()