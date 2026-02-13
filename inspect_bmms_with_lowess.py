import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

# --- CONFIGURATION ---
DATA_PATH = Path(r"C:\Users\larsg\data\raw\WBSIM_Lab1_2024\infrastructure")
BMMS_FILE = DATA_PATH / "BMMS_overview.xlsx"
ROADS_TO_ANALYZE = ["N3"]  # Current filter in the original script

# Outlier parameters
K_MAD = 3
ABS_THRESHOLD_METERS = 250
ABS_THRESHOLD_KM = ABS_THRESHOLD_METERS / 1000.0
MIN_RESIDUAL_PCT_POS = 0.10
MIN_RESIDUAL_PCT_NEG = 0.05


# --- UTILITIES ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in km."""
    R = 6371  # Aardstraal in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def ratio_outlier(
    dist_coords,
    delta_chainage,
    min_ratio=0.2,
    max_ratio=1.80,
    max_abs_diff_km=0.3,
):
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
    # lowess returns sorted x; group is already sorted by dist_coords.
    return fitted[:, 1]


# --- DATA PROCESSING ---

def load_and_preprocess(file_path, roads):
    """Load and clean BMMS data for specific roads."""
    df = pd.read_excel(file_path)
    print("Kolommen in BMMS:", df.columns.tolist())
    
    df_filtered = df[df["road"].isin(roads)].copy()
    df_filtered["chainage"] = pd.to_numeric(df_filtered["chainage"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["chainage", "lat", "lon"])
    df_filtered = df_filtered.sort_values(["road", "chainage"])
    
    return df_filtered


def analyze_segments(df):
    """Perform segment-by-segment analysis."""
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
                "asset1": a["LRPName"],
                "asset2": b["LRPName"],
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

        # Skip the first point (index 0 in the group) as it is always (0,0) and LOWESS 
        # fits at the boundaries can be unstable, leading to false positives.
        outliers = pos_outliers | neg_outliers
        if not group.empty:
            outliers.iloc[0] = False

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
    plt.title("Segment-analyse: Chainage vs Coordinaten (N1 & N2)")
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
    plt.title("Start-analyse: Chainage vs Coordinaten (N1 & N2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nStart-analyse scatterplot opgeslagen als: {output_file}")
    plt.show()


# --- MAIN ---

def main():
    # 1. Data laden
    df_filtered = load_and_preprocess(BMMS_FILE, ROADS_TO_ANALYZE)
    
    # 2. Segment-analyse
    seg_df = analyze_segments(df_filtered)
    print("\n--- Grootste afwijkingen (segmenten) ---")
    print(seg_df.sort_values("diff", ascending=False).head(10))
    print("\n--- Outliers (ratio) ---")
    print(seg_df[seg_df["outlier"]].sort_values("diff", ascending=False).head(10))
    outlier_pct = seg_df["outlier"].mean() * 100
    print(f"\nOutlier percentage (segmenten): {outlier_pct:.2f}%")
    
    plot_segment_analysis(seg_df)
    
    # 3. Start-analyse (LOWESS)
    start_df = analyze_from_start(df_filtered)
    
    print("\n--- Start-analyse outliers (lowess, boven fit) ---")
    print(
        start_df[start_df["outlier"]]
        .sort_values("residual_lowess", ascending=False)
        .head(10)
    )
    
    plot_start_analysis(start_df)


if __name__ == "__main__":
    main()
