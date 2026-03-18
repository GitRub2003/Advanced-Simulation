from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("network_model.csv")
OUTPUT = Path("network_map.png")


def prepare_network_df(path: Path) -> pd.DataFrame:
    """Load the network CSV and normalize the columns used for plotting."""
    df = pd.read_csv(path)
    df["road"] = df["road"].astype(str).str.strip().str.upper()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()
    df["name"] = df.get("name", "").fillna("").astype(str)

    # Rows without coordinates cannot be placed on the map.
    return df.dropna(subset=["road", "lat", "lon"]).copy()


def get_intersections_to_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per plotted intersection to avoid duplicate markers."""
    intersections = df[df["model_type"] == "intersection"].copy()

    if "id" in intersections.columns:
        return intersections.drop_duplicates(subset=["id"]).copy()

    return intersections.drop_duplicates(subset=["lat", "lon"]).copy()


def make_intersection_label(row: pd.Series) -> str:
    """Build a readable label for an intersection annotation."""
    name = row["name"].strip()
    if name:
        return name

    if "id" in row.index:
        return f'INT_{row["id"]}'

    return "Intersection"


def add_unique_legend(ax) -> None:
    """Add a legend while removing duplicate labels created by repeated plotting calls."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []

    for handle, label in zip(handles, labels):
        if label not in seen:
            uniq_handles.append(handle)
            uniq_labels.append(label)
            seen.add(label)

    ax.legend(uniq_handles, uniq_labels, loc="best", fontsize=8, ncol=2)


def main():
    """Create a map of the road network and highlight bridges and intersections."""
    df = prepare_network_df(INPUT)

    fig, ax = plt.subplots(figsize=(11, 11))

    # Plot each road as a continuous line using all non-intersection points.
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

    intersections_plot = get_intersections_to_plot(df)
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
            ax.annotate(
                make_intersection_label(row),
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

    add_unique_legend(ax)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.show()

    print(f"Saved map to {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
