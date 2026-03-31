from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
NETWORK_INPUT = BASE_DIR / "data" / "network_model.csv"
OUTPUT = BASE_DIR / "Experiments" / "plots" / "network_map_without_bridges.png"


def load_network(path: Path) -> pd.DataFrame:
    """Load and normalize the network CSV for plotting."""
    df = pd.read_csv(path).copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["road"] = df["road"].astype(str).str.strip().str.upper()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["road", "lat", "lon"]).copy()


def add_unique_legend(ax) -> None:
    """Add a legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []

    for handle, label in zip(handles, labels):
        if label not in seen:
            uniq_handles.append(handle)
            uniq_labels.append(label)
            seen.add(label)

    ax.legend(uniq_handles, uniq_labels, loc="best", fontsize=8, ncol=2)


def main() -> None:
    """Plot the network lines only, without bridge markers."""
    df = load_network(NETWORK_INPUT)

    fig, ax = plt.subplots(figsize=(11, 11))

    road_df = df[df["model_type"] != "intersection"].copy()
    road_df = road_df.sort_values(["road", "chainage", "id"])

    for road_name, group in road_df.groupby("road", sort=False):
        ax.plot(group["lon"], group["lat"], linewidth=1.8, alpha=0.95, label=road_name)

    ax.set_title("Network map without bridge markers")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    add_unique_legend(ax)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved network map to: {OUTPUT}")


if __name__ == "__main__":
    main()
