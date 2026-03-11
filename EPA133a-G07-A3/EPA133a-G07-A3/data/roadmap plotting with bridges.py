from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("network_model.csv")
OUTPUT = Path("network_map.png")

def main():
    df = pd.read_csv(INPUT)

    # Basic cleanup
    df["road"] = df["road"].astype(str).str.strip().str.upper()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()
    df["name"] = df.get("name", "").fillna("").astype(str)

    df = df.dropna(subset=["road", "lat", "lon"]).copy()

    fig, ax = plt.subplots(figsize=(11, 11))

    # Plot each road as a line through all non-intersection objects
    road_df = df[df["model_type"] != "intersection"].copy()
    if "chainage" in road_df.columns:
        road_df = road_df.sort_values(["road", "chainage", "id"])
    else:
        road_df = road_df.sort_values(["road", "id"])

    for road_name, g in road_df.groupby("road", sort=False):
        ax.plot(g["lon"], g["lat"], linewidth=1.6, alpha=0.9, label=road_name)

    # Highlight bridges
    bridges = df[df["model_type"] == "bridge"].copy()
    if not bridges.empty:
        ax.scatter(
            bridges["lon"],
            bridges["lat"],
            s=10,
            marker="s",
            color="yellow",
            zorder=4,
            label="Bridge"
        )

    # Highlight intersections
    intersections = df[df["model_type"] == "intersection"].copy()

    # avoid duplicate labels/markers for same intersection id
    if "id" in intersections.columns:
        intersections_plot = intersections.drop_duplicates(subset=["id"]).copy()
    else:
        intersections_plot = intersections.drop_duplicates(subset=["lat", "lon"]).copy()

    if not intersections_plot.empty:
        ax.scatter(
            intersections_plot["lon"],
            intersections_plot["lat"],
            s=120,
            marker="x",
            linewidths=2.2,
            zorder=5,
            label="Intersection"
        )

        for _, row in intersections_plot.iterrows():
            label = row["name"].strip() if row["name"].strip() else f'INT_{row["id"]}'
            ax.annotate(
                label,
                (row["lon"], row["lat"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8
            )

    # Optional: mark sourcesinks
    sourcesinks = df[df["model_type"] == "sourcesink"].copy()
    if not sourcesinks.empty:
        ax.scatter(
            sourcesinks["lon"],
            sourcesinks["lat"],
            s=40,
            marker="o",
            zorder=4,
            label="SourceSink"
        )

    ax.set_title("Network map from network_model.csv")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # cleaner legend: unique labels only
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.show()

    print(f"Saved map to {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
