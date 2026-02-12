import matplotlib.pyplot as plt
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# Paden instellen
DATA_PATH = Path(r"C:\Users\larsg\data\raw\WBSIM_Lab1_2024\infrastructure")
BMMS_FILE = DATA_PATH / "BMMS_overview.xlsx"


def haversine(lat1, lon1, lat2, lon2):
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
    # Markeer outliers op basis van ratio, maar sta kleine absolute afwijkingen toe.
    # Verwacht: dist_coords / delta_chainage < 1, maar niet te klein.
    if delta_chainage <= 0:
        return True
    ratio = dist_coords / delta_chainage
    abs_diff = abs(dist_coords - delta_chainage)
    ratio_outside = ratio < min_ratio or ratio > max_ratio
    return ratio_outside and abs_diff > max_abs_diff_km


# Data laden
df = pd.read_excel(BMMS_FILE)

# Relevante kolommen selecteren en opschonen
print("Kolommen in BMMS:", df.columns.tolist())
print(df.head())

# Filter op N1 en N2
#df_filtered = df[df["road"].isin(["N1", "N2"])].copy()
df_filtered = df[df["road"].isin(["N1"])].copy()
# Sorteren op weg en chainage
df_filtered["chainage"] = pd.to_numeric(df_filtered["chainage"], errors="coerce")
df_filtered = df_filtered.dropna(subset=["chainage", "lat", "lon"])
df_filtered = df_filtered.sort_values(["road", "chainage"])

segment_rows = []

for road, group in df_filtered.groupby("road"):
    group = group.reset_index(drop=True)
    if len(group) < 2:
        continue

    for i in range(1, len(group)):
        a = group.iloc[i - 1]
        b = group.iloc[i]

        # Segmentafstand tussen opeenvolgende punten (in km)
        delta_chainage = b["chainage"] - a["chainage"]
        dist_coords = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        diff = dist_coords - delta_chainage
        rel_err = diff / delta_chainage if delta_chainage else float("nan")

        segment_rows.append(
            {
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
            }
        )

seg_df = pd.DataFrame(segment_rows)
seg_df = seg_df[seg_df["delta_chainage"] > 0].copy()

seg_df["ratio"] = seg_df["dist_coords"] / seg_df["delta_chainage"]
seg_df["outlier"] = seg_df.apply(
    lambda row: ratio_outlier(row["dist_coords"], row["delta_chainage"]), axis=1
)

print("--- Grootste afwijkingen (segmenten) ---")
print(seg_df.sort_values("diff", ascending=False).head(10))
print("\n--- Grootste relatieve afwijkingen ---")
print(seg_df.sort_values("rel_err", ascending=False).head(10))
print("\n--- Outliers (ratio) ---")
print(seg_df[seg_df["outlier"]].sort_values("diff", ascending=False).head(10))
outlier_pct = seg_df["outlier"].mean() * 100
print(f"\nOutlier percentage: {outlier_pct:.2f}%")

# Scatterplot maken (x = coords, y = chainage-delta)
plt.figure(figsize=(10, 6))

for road in seg_df["road"].unique():
    subset = seg_df[seg_df["road"] == road]
    normal = subset[~subset["outlier"]]
    outliers = subset[subset["outlier"]]

    plt.scatter(
        normal["dist_coords"],
        normal["delta_chainage"],
        label=f"Weg {road}",
        alpha=0.6,
    )
    if not outliers.empty:
        plt.scatter(
            outliers["dist_coords"],
            outliers["delta_chainage"],
            color="red",
            edgecolor="black",
            s=70,
            label=f"Weg {road} outliers",
        )

# Referentielijn y=x
max_val = max(seg_df["delta_chainage"].max(), seg_df["dist_coords"].max())
plt.plot([0, max_val], [0, max_val], "r--", label="Ideaal (y=x)")

plt.xlabel("Afstand op basis van Coordinaten (km, segment)")
plt.ylabel("Chainage (km, segment)")
plt.title("Segment-analyse: Chainage vs Coordinaten (N1 & N2)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.xlim(xmin=0, xmax=25)
plt.ylim(ymin=0, ymax=25)

output_plot = "route_analysis_outliers.png"
plt.savefig(output_plot, dpi=150)
print(f"\nScatterplot opgeslagen als: {output_plot}")
plt.show()
