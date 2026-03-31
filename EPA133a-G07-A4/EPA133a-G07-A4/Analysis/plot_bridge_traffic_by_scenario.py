from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
NETWORK_INPUT = BASE_DIR / "data" / "network_model.csv"
IMPORTANCE_INPUT = BASE_DIR / "Experiments" / "plots" / "infrastructure_importance_by_scenario.csv"
OUTPUT_DIR = BASE_DIR / "Experiments" / "plots"
OUTPUT_PATTERN = "bridge_traffic_map_scenario_{scenario}.png"


def load_network(path: Path) -> pd.DataFrame:
    """Load the network geometry used for plotting."""
    df = pd.read_csv(path).copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["road"] = df["road"].astype(str).str.strip().str.upper()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["lrp"] = df.get("lrp", "").fillna("").astype(str)
    return df.dropna(subset=["lat", "lon"]).copy()


def load_bridge_importance(path: Path) -> pd.DataFrame:
    """Load per-scenario bridge crossing summaries."""
    df = pd.read_csv(path).copy()
    required = {"scenario", "infra_id", "infra_type", "mean_crossings"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Importance file is missing required columns: {sorted(missing)}")

    df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    df["infra_id"] = pd.to_numeric(df["infra_id"], errors="coerce").astype("Int64")
    df["infra_type"] = df["infra_type"].astype(str).str.strip().str.lower()
    df["mean_crossings"] = pd.to_numeric(df["mean_crossings"], errors="coerce")
    df = df.dropna(subset=["scenario", "infra_id", "mean_crossings"]).copy()
    return df[df["infra_type"] == "bridge"].copy()


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


def make_bridge_label(row: pd.Series) -> str:
    """Build a short readable bridge label."""
    name = str(row.get("name", "")).strip()
    lrp = str(row.get("lrp", "")).strip()
    if name and lrp:
        return f"{name} ({lrp})"
    if name:
        return name
    if lrp:
        return lrp
    return f'Bridge {int(row["id"])}'


def plot_scenario_map(network_df: pd.DataFrame, scenario_bridges: pd.DataFrame, scenario_id: int) -> Path:
    """Plot one scenario-specific bridge traffic map."""
    fig, ax = plt.subplots(figsize=(12, 12))

    road_df = network_df[network_df["model_type"] != "intersection"].copy()
    road_df = road_df.sort_values(["road", "chainage", "id"])
    for road_name, group in road_df.groupby("road", sort=False):
        ax.plot(group["lon"], group["lat"], linewidth=1.4, alpha=0.5, color="#94a3b8")

    bridges = network_df[network_df["model_type"] == "bridge"].copy()
    bridges = bridges.merge(
        scenario_bridges[["infra_id", "mean_crossings"]],
        left_on="id",
        right_on="infra_id",
        how="left",
    )

    low_traffic = bridges[bridges["mean_crossings"].isna()].copy()
    if not low_traffic.empty:
        ax.scatter(
            low_traffic["lon"],
            low_traffic["lat"],
            s=20,
            marker="s",
            color="#cbd5e1",
            zorder=4,
            label="Bridge without scenario data",
        )

    with_traffic = bridges[bridges["mean_crossings"].notna()].copy()
    if with_traffic.empty:
        raise ValueError(f"No bridge traffic data found for scenario {scenario_id}.")

    vmin = float(with_traffic["mean_crossings"].min())
    vmax = float(with_traffic["mean_crossings"].max())
    if vmin == vmax:
        vmax = vmin + 1.0

    cmap = plt.get_cmap("RdYlGn_r")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    scatter = ax.scatter(
        with_traffic["lon"],
        with_traffic["lat"],
        c=with_traffic["mean_crossings"],
        cmap=cmap,
        norm=norm,
        s=55,
        marker="s",
        edgecolors="black",
        linewidths=0.25,
        zorder=5,
        label="Bridge traffic level",
    )

    top_bridges = with_traffic.nlargest(5, "mean_crossings").copy()
    for _, row in top_bridges.iterrows():
        ax.annotate(
            make_bridge_label(row),
            (row["lon"], row["lat"]),
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=8,
            color="#7f1d1d",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label("Mean bridge crossings")

    ax.set_title(f"Bridge traffic map - scenario {scenario_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    add_unique_legend(ax)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_PATTERN.format(scenario=scenario_id)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Create one bridge-traffic map per scenario."""
    network_df = load_network(NETWORK_INPUT)
    bridge_importance = load_bridge_importance(IMPORTANCE_INPUT)

    scenarios = sorted(bridge_importance["scenario"].dropna().astype(int).unique().tolist())
    output_paths: list[Path] = []
    for scenario_id in scenarios:
        scenario_bridges = bridge_importance[bridge_importance["scenario"].astype(int) == scenario_id].copy()
        output_paths.append(plot_scenario_map(network_df, scenario_bridges, scenario_id))

    print("Generated bridge traffic maps:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
