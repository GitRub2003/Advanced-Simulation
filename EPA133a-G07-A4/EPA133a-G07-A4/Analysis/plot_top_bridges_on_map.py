from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NETWORK_INPUT = Path(__file__).resolve().parents[1] / "data" / "network_model.csv"
IMPORTANCE_INPUT = Path(__file__).resolve().parents[1] / "Experiments" / "plots" / "infrastructure_importance_by_scenario.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "Experiments" / "network_map_top_bridges.png"
TOP_N = 4


def prepare_network_df(path: Path) -> pd.DataFrame:
    """Load the network CSV and normalize the columns used for plotting."""
    df = pd.read_csv(path)
    df["road"] = df["road"].astype(str).str.strip().str.upper()
    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["lrp"] = df.get("lrp", "").fillna("").astype(str)

    return df.dropna(subset=["road", "lat", "lon"]).copy()


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


def load_consistent_top_bridges(path: Path, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Return bridges that appear in the top-N by mean crossings for every scenario.
    """
    importance = pd.read_csv(path).copy()
    required = {"scenario", "infra_id", "infra_label", "infra_type", "road_name", "mean_crossings"}
    missing = required - set(importance.columns)
    if missing:
        raise ValueError(f"Importance file is missing required columns: {sorted(missing)}")

    importance["infra_type"] = importance["infra_type"].astype(str).str.strip().str.lower()
    importance["scenario"] = pd.to_numeric(importance["scenario"], errors="coerce")
    importance["infra_id"] = pd.to_numeric(importance["infra_id"], errors="coerce")
    importance["mean_crossings"] = pd.to_numeric(importance["mean_crossings"], errors="coerce")

    bridge_rows = importance.dropna(subset=["scenario", "infra_id", "mean_crossings"]).copy()
    bridge_rows = bridge_rows[bridge_rows["infra_type"] == "bridge"].copy()
    if bridge_rows.empty:
        raise ValueError("No bridge rows found in infrastructure importance summary.")

    scenarios = sorted(bridge_rows["scenario"].astype(int).unique().tolist())
    top_ids_by_scenario: list[set[int]] = []
    for scenario in scenarios:
        top_rows = (
            bridge_rows[bridge_rows["scenario"].astype(int) == scenario]
            .sort_values(["mean_crossings", "infra_id"], ascending=[False, True])
            .head(top_n)
        )
        top_ids_by_scenario.append(set(top_rows["infra_id"].astype(int).tolist()))

    common_ids = set.intersection(*top_ids_by_scenario) if top_ids_by_scenario else set()
    if not common_ids:
        raise ValueError(f"No bridges appear in the top {top_n} for every scenario.")

    common_bridges = (
        bridge_rows[bridge_rows["infra_id"].astype(int).isin(common_ids)]
        .sort_values(["infra_id", "scenario"])
        .drop_duplicates(subset=["infra_id"])
        .loc[:, ["infra_id", "infra_label", "road_name"]]
        .copy()
    )
    common_bridges["infra_id"] = common_bridges["infra_id"].astype(int)
    return common_bridges


def make_bridge_label(row: pd.Series) -> str:
    """Build a readable label for a highlighted bridge marker."""
    name = str(row.get("name", "")).strip()
    lrp = str(row.get("lrp", "")).strip()

    if name and lrp:
        return f"{name} ({lrp})"
    if name:
        return name
    if lrp:
        return lrp
    return f'Bridge {int(row["id"])}'


def plot_top_bridges(network_df: pd.DataFrame, top_bridges: pd.DataFrame, output_path: Path) -> None:
    """Plot the network and highlight the bridges that are consistently top-ranked."""
    fig, ax = plt.subplots(figsize=(11, 11))

    road_df = network_df[network_df["model_type"] != "intersection"].copy()
    road_df = road_df.sort_values(["road", "chainage", "id"])

    for road_name, group in road_df.groupby("road", sort=False):
        ax.plot(group["lon"], group["lat"], linewidth=1.6, alpha=0.9, label=road_name)

    bridges = network_df[network_df["model_type"] == "bridge"].copy()
    if not bridges.empty:
        ax.scatter(
            bridges["lon"],
            bridges["lat"],
            s=10,
            marker="s",
            color="gold",
            zorder=4,
            label="Bridge",
        )

    highlighted = network_df[
        (network_df["model_type"] == "bridge")
        & (network_df["id"].isin(top_bridges["infra_id"]))
    ].copy()
    highlighted = highlighted.merge(
        top_bridges.rename(columns={"infra_id": "id"}),
        on="id",
        how="left",
    )

    ax.scatter(
        highlighted["lon"],
        highlighted["lat"],
        s=110,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=2.2,
        zorder=6,
        label=f"Top {TOP_N} bridges in every scenario",
    )

    for _, row in highlighted.iterrows():
        ax.annotate(
            make_bridge_label(row),
            (row["lon"], row["lat"]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
            color="darkred",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    ax.set_title("Network map with consistently busiest bridges highlighted")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    add_unique_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create the bridge-highlight map from experiment rankings and network geometry."""
    network_df = prepare_network_df(NETWORK_INPUT)
    top_bridges = load_consistent_top_bridges(IMPORTANCE_INPUT, top_n=TOP_N)
    plot_top_bridges(network_df, top_bridges, OUTPUT)

    labels = ", ".join(
        f'{row["infra_label"]} [{int(row["infra_id"])}]'
        for _, row in top_bridges.iterrows()
    )
    print(f"Saved highlighted bridge map to: {OUTPUT}")
    print(f"Highlighted bridges: {labels}")


if __name__ == "__main__":
    main()
