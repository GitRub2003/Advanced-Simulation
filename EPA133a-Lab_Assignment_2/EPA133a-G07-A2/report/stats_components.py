"""
Simple experiment analysis script.

Run:
    python stats_components.py

What it does:
    - Reads truck and bridge CSV files from the Experiments folder
    - Aggregates scenario/replication metrics
    - Saves plots as PNG in ./img
    - Saves aggregated table as ./img/aggregated_metrics.csv
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------- Config -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "img"
EXPERIMENTS_DIR = PROJECT_ROOT / "Experiments"

EXPECTED_SCENARIOS = 9
EXPECTED_REPLICATIONS = 10
TOP_N_WORST_BRIDGES = 5
TOP_K_PARETO = 20

TRUCK_FILE_RE = re.compile(r"^truck_driving_times_scenario_(\d+)_replicate_(\d+)\.csv$", re.IGNORECASE)
BRIDGE_FILE_RE = re.compile(r"^bridge_total_wait_times_scenario_(\d+)_replicate_(\d+)\.csv$", re.IGNORECASE)

# Road selection for analysis:
# - Set to "ALL" to use all roads.
# - Set to a specific road like "N1" to focus analysis.
ROAD_SELECTION = "ALL"
LENGTH_CATEGORY_ORDER = ["Over 200 m", "50-200 m", "10-50 m", "Under 10 m"]


def discover_files():
    """
    Return two dictionaries:
        truck_files[(scenario_id, replication_id)] = path
        bridge_files[(scenario_id, replication_id)] = path
    """
    truck_files = {}
    bridge_files = {}
    stats = {"csv_found": 0, "truck_files": 0, "bridge_files": 0, "warnings": []}

    if not EXPERIMENTS_DIR.exists():
        stats["warnings"].append(f"Experiments folder not found: {EXPERIMENTS_DIR}")
        return truck_files, bridge_files, stats

    # File names encode scenario/replication IDs; regex classification converts
    # a raw folder scan into structured experiment keys.
    for path in EXPERIMENTS_DIR.rglob("*.csv"):
        stats["csv_found"] += 1
        truck_match = TRUCK_FILE_RE.match(path.name)
        bridge_match = BRIDGE_FILE_RE.match(path.name)

        if truck_match:
            key = (int(truck_match.group(1)), int(truck_match.group(2)))
            truck_files[key] = path
            stats["truck_files"] += 1
        elif bridge_match:
            key = (int(bridge_match.group(1)), int(bridge_match.group(2)))
            bridge_files[key] = path
            stats["bridge_files"] += 1

    if not truck_files and not bridge_files:
        stats["warnings"].append(f"No matching CSV files found in {EXPERIMENTS_DIR}")

    return truck_files, bridge_files, stats


def road_scope_label():
    """Return a human-readable road scope label for titles and summaries."""
    selected = str(ROAD_SELECTION).strip()
    if selected.upper() == "ALL":
        return "All Roads"
    return selected.upper()


def scoped_title(base_title):
    """Append road scope to a chart title."""
    return f"{base_title} ({road_scope_label()})"


def filter_by_road(df, stats, df_name):
    """
    Filter a dataframe by ROAD_SELECTION if possible.
    Accepts common road column names: road_name, road, road_id.
    """
    if df.empty:
        return df

    selected = str(ROAD_SELECTION).strip().upper()
    if selected == "ALL":
        return df

    road_col = "road_name",

    if road_col is None:
        stats["warnings"].append(
            f"{df_name}: ROAD_SELECTION={selected}, but no road column found; data kept unfiltered."
        )
        return df

    filtered = df[df[road_col].astype(str).str.strip().str.upper() == selected].copy()
    if filtered.empty:
        stats["warnings"].append(f"{df_name}: no rows found for ROAD_SELECTION={selected}.")
    return filtered


def build_length_category(length_series):
    """Group numeric bridge lengths (m) into report categories."""
    length_num = pd.to_numeric(length_series, errors="coerce")
    bins = [0, 10, 50, 200, np.inf]
    labels = ["Under 10 m", "10-50 m", "50-200 m", "Over 200 m"]
    categories = pd.cut(length_num, bins=bins, labels=labels, right=False, include_lowest=True)
    return pd.Series(categories, index=length_series.index).astype("object")


def load_truck_driving(truck_files, stats):
    """Load truck driving-time CSVs and return a standardized long-form table."""
    rows = []
    for (scenario_id, replication_id), path in truck_files.items():
        df = pd.read_csv(path)
        if "total_driving_time" not in df.columns:
            stats["warnings"].append(f"Missing total_driving_time in {path}")
            continue

        tmp = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "replication_id": replication_id,
                "source_file": str(path),
                "total_driving_time": pd.to_numeric(df["total_driving_time"], errors="coerce"),
            }
        ).dropna(subset=["total_driving_time"])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["scenario_id", "replication_id", "source_file", "total_driving_time"])

    out = pd.concat(rows, ignore_index=True)
    stats["truck_rows"] = len(out)
    return out


def load_bridge_wait_times(bridge_files, stats):
    """Load bridge wait-time CSVs and harmonize key bridge attributes."""
    needed = ["bridge_id", "bridge_name", "road_name", "condition", "total_wait_time", "length"]
    rows = []

    for (scenario_id, replication_id), path in bridge_files.items():
        df = pd.read_csv(path)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            stats["warnings"].append(f"Missing columns {missing} in {path}")
            continue

        tmp = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "replication_id": replication_id,
                "source_file": str(path),
                "bridge_id": df["bridge_id"].astype(str).str.strip(),
                "bridge_name": df["bridge_name"].astype(str).str.strip(),
                "road_name": df["road_name"].astype(str).str.strip(),
                "condition_category": df["condition"].astype(str).str.strip().str.upper(),
                "total_wait_time": pd.to_numeric(df["total_wait_time"], errors="coerce"),
                "length_category": build_length_category(df["length"]),
            }
        ).dropna(subset=["total_wait_time"])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "replication_id",
                "source_file",
                "bridge_id",
                "bridge_name",
                "road_name",
                "condition_category",
                "total_wait_time",
                "length_category",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    stats["bridge_rows"] = len(out)
    return out


def aggregate_metrics(truck_df, bridge_df):
    """Aggregate per scenario-replication statistics for driving and wait metrics."""
    parts = []

    if not truck_df.empty:
        d = truck_df.groupby(["scenario_id", "replication_id"])["total_driving_time"].agg(
            rows="size", sum_value="sum", mean_value="mean", median_value="median"
        )
        d = d.reset_index()
        d["metric_group"] = "driving_time"
        parts.append(d)

    if not bridge_df.empty:
        b = bridge_df.groupby(["scenario_id", "replication_id"])["total_wait_time"].agg(
            rows="size", sum_value="sum", mean_value="mean", median_value="median"
        )
        b = b.reset_index()
        b["metric_group"] = "bridge_wait_time"
        parts.append(b)

    if not parts:
        return pd.DataFrame(
            columns=["scenario_id", "replication_id", "rows", "sum_value", "mean_value", "median_value", "metric_group"]
        )
    return pd.concat(parts, ignore_index=True)


def save_heatmap(table, title, xlabel, ylabel, out_path, colorbar_label="Percent"):
    """Render a numeric matrix as a labeled heatmap and save it to disk."""
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(table.shape[1]))
    xlabels = [str(x) for x in table.columns.tolist()]
    ylabels = [str(y) for y in table.index.tolist()]
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def percent_table(df, category_col, value_col=None):
    """
    Build a scenario x category percentage table.
    - If value_col is given: percentage share of summed value_col.
    - If value_col is None: percentage share of event counts.
    """
    if value_col is None:
        base = df.groupby(["scenario_id", category_col], as_index=False).size()
        value_name = "value"
        base = base.rename(columns={"size": value_name})
    else:
        value_name = "value"
        base = (
            df.groupby(["scenario_id", category_col], as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: value_name})
        )

    table = base.pivot(index="scenario_id", columns=category_col, values=value_name).fillna(0)
    table = table.reindex(range(EXPECTED_SCENARIOS), fill_value=0).sort_index()
    # Normalize by scenario so each row represents a percentage composition.
    # Zero-total rows are converted to NaN temporarily to avoid divide-by-zero.
    row_total = table.sum(axis=1).replace(0, np.nan)
    pct = table.div(row_total, axis=0) * 100
    pct = pct.fillna(0)
    return pct


def plot_length_heatmaps(bridge_df, stats):
    """Create heatmaps for wait-share and event-share by bridge length class."""
    d = bridge_df.dropna(subset=["scenario_id", "length_category"]).copy()
    if d.empty:
        stats["warnings"].append("Skipping length heatmaps: no length_category data.")
        return

    wait_share = percent_table(d, category_col="length_category", value_col="total_wait_time")
    wait_share = wait_share.reindex(columns=LENGTH_CATEGORY_ORDER, fill_value=0)
    save_heatmap(
        wait_share,
        scoped_title("Share of Total Wait Time by Bridge Length Category and Scenario"),
        "Bridge Length Category",
        "Scenario",
        OUTPUT_DIR / "all_scenarios_total_wait_time_length_category_heatmap_wait_share.png",
    )

    event_share = percent_table(d, category_col="length_category", value_col=None)
    event_share = event_share.reindex(columns=LENGTH_CATEGORY_ORDER, fill_value=0)
    save_heatmap(
        event_share,
        scoped_title("Share of Bridge Wait Events by Length Category and Scenario"),
        "Bridge Length Category",
        "Scenario",
        OUTPUT_DIR / "all_scenarios_total_wait_time_length_category_heatmap_event_share.png",
    )


def plot_length_line(bridge_df, stats):
    """Plot mean wait time across length categories for each scenario."""
    d = bridge_df.dropna(subset=["scenario_id", "length_category"]).copy()
    d = d[d["length_category"].astype(str).str.lower() != "nan"]
    if d.empty:
        stats["warnings"].append("Skipping length line plot: no length_category data.")
        return

    summary = d.groupby(["scenario_id", "length_category"])["total_wait_time"].mean().reset_index()
    categories = LENGTH_CATEGORY_ORDER

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in range(EXPECTED_SCENARIOS):
        s = summary[summary["scenario_id"] == scenario].set_index("length_category").reindex(categories)
        if s["total_wait_time"].notna().any():
            ax.plot(categories, s["total_wait_time"], marker="o", label=f"Scenario {int(scenario)}")

    ax.set_title(scoped_title("Average Bridge Wait Time by Length Category"))
    ax.set_xlabel("Bridge Length Category")
    ax.set_ylabel("Average Total Wait Time")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_wait_time_length_category_line_mean.png", dpi=180)
    plt.close(fig)


def plot_condition_heatmap(bridge_df, stats):
    """Create a heatmap of wait-time share by condition category (A-D)."""
    d = bridge_df.dropna(subset=["scenario_id", "condition_category"]).copy()
    d = d[d["condition_category"].isin(["A", "B", "C", "D"])]
    if d.empty:
        stats["warnings"].append("Skipping condition heatmap: no A-D condition data.")
        return

    table = percent_table(d, category_col="condition_category", value_col="total_wait_time")
    table = table.reindex(columns=["A", "B", "C", "D"], fill_value=0)
    save_heatmap(
        table,
        scoped_title("Share of Total Wait Time by Condition Category and Scenario"),
        "Condition Category",
        "Scenario",
        OUTPUT_DIR / "all_scenarios_total_wait_time_condition_heatmap_wait_share.png",
    )


def plot_condition_line(bridge_df, stats):
    """Plot mean wait time by condition category for each scenario."""
    d = bridge_df.dropna(subset=["scenario_id", "condition_category"]).copy()
    d = d[d["condition_category"].isin(["A", "B", "C", "D"])]
    if d.empty:
        stats["warnings"].append("Skipping condition line plot: no A-D condition data.")
        return

    summary = d.groupby(["scenario_id", "condition_category"])["total_wait_time"].mean().reset_index()
    categories = ["A", "B", "C", "D"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in range(EXPECTED_SCENARIOS):
        s = summary[summary["scenario_id"] == scenario].set_index("condition_category").reindex(categories)
        if s["total_wait_time"].notna().any():
            ax.plot(categories, s["total_wait_time"], marker="o", label=f"Scenario {int(scenario)}")

    ax.set_title(scoped_title("Average Bridge Wait Time by Condition Category"))
    ax.set_xlabel("Condition Category")
    ax.set_ylabel("Average Total Wait Time")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_wait_time_condition_line_mean.png", dpi=180)
    plt.close(fig)


def plot_driving_boxplot(truck_df, stats):
    """Compare replication-level driving-time distributions across scenarios."""
    if truck_df.empty:
        stats["warnings"].append("Skipping driving boxplot: no truck data.")
        return

    rep = truck_df.groupby(["scenario_id", "replication_id"])["total_driving_time"].sum().reset_index()
    scenarios = sorted(rep["scenario_id"].unique())
    values = [rep.loc[rep["scenario_id"] == s, "total_driving_time"].values for s in scenarios]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(values, labels=[str(int(s)) for s in scenarios], showmeans=True)
    ax.set_title(scoped_title("Travel Time Distribution per Scenario (10 Replications Expected)"))
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Driving Time per Replication (sum)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_driving_time_boxplot_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_worst_bridges_dot(bridge_df, stats):
    """Plot top-N worst bridges by mean wait time with uncertainty bars."""
    d = bridge_df.dropna(subset=["scenario_id", "bridge_id"]).copy()
    d = d[d["scenario_id"].between(1, 7)]
    if d.empty:
        stats["warnings"].append("Skipping worst-bridges dot plot: no scenario 1-7 bridge data.")
        return

    worst = d.groupby("bridge_id")["total_wait_time"].agg(mean_wait="mean", std_wait="std")
    worst = worst.sort_values("mean_wait", ascending=False).head(TOP_N_WORST_BRIDGES).sort_values("mean_wait")

    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(worst))
    ax.errorbar(worst["mean_wait"], y, xerr=worst["std_wait"].fillna(0), fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(worst.index.astype(str))
    ax.set_title(scoped_title(f"Top {len(worst)} Worst Bridges by Mean Wait Time"))
    ax.set_xlabel("Mean Total Wait Time (Scenarios 1-7)")
    ax.set_ylabel("Bridge")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scenario_1_7_total_wait_time_worst_bridges_dotplot.png", dpi=180)
    plt.close(fig)


def plot_replication_heatmap(truck_df, stats):
    """Show total driving time per scenario-replication pair as a heatmap."""
    if truck_df.empty:
        return
    table = truck_df.groupby(["scenario_id", "replication_id"])["total_driving_time"].sum().unstack()
    if table.empty:
        return
    save_heatmap(
        table,
        scoped_title("Total Driving Time per Scenario and Replication"),
        "Replication",
        "Scenario",
        OUTPUT_DIR / "all_scenarios_total_driving_time_replication_heatmap.png",
        colorbar_label="Total Driving Time (minutes)",
    )


def plot_wait_ecdf(bridge_df, stats):
    """Plot empirical CDFs of bridge wait times for scenario-wise comparison."""
    d = bridge_df.dropna(subset=["scenario_id", "total_wait_time"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in sorted(d["scenario_id"].unique()):
        values = np.sort(d.loc[d["scenario_id"] == scenario, "total_wait_time"].to_numpy())
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=f"Scenario {int(scenario)}", alpha=0.9)

    ax.set_title(scoped_title("ECDF of Bridge Wait Times by Scenario"))
    ax.set_xlabel("Total Wait Time")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_wait_time_ecdf_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_pareto(bridge_df, stats):
    """Plot cumulative top-K bridge contributions to total scenario wait time."""
    d = bridge_df.dropna(subset=["scenario_id", "bridge_id"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in sorted(d["scenario_id"].unique()):
        totals = d[d["scenario_id"] == scenario].groupby("bridge_id")["total_wait_time"].sum().sort_values(ascending=False)
        if totals.empty:
            continue
        top = totals.head(TOP_K_PARETO)
        cum = top.cumsum() / totals.sum() * 100 if totals.sum() > 0 else top * 0
        ax.plot(np.arange(1, len(cum) + 1), cum.values, marker="o", label=f"Scenario {int(scenario)}")

    ax.set_title(scoped_title("Pareto Concentration of Bridge Wait Time by Scenario"))
    ax.set_xlabel(f"Top-K Bridges (K <= {TOP_K_PARETO})")
    ax.set_ylabel("Cumulative Share of Total Wait Time (%)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_wait_time_pareto_topk_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_driving_vs_wait(agg_df, stats):
    """Scatter total driving time against total wait time per replication."""
    if agg_df.empty:
        return

    drive = agg_df[agg_df["metric_group"] == "driving_time"][["scenario_id", "replication_id", "sum_value"]]
    wait = agg_df[agg_df["metric_group"] == "bridge_wait_time"][["scenario_id", "replication_id", "sum_value"]]
    drive = drive.rename(columns={"sum_value": "total_driving_time_sum"})
    wait = wait.rename(columns={"sum_value": "total_wait_time_sum"})
    # Inner join keeps only matched scenario-replication runs so both axes
    # describe the same simulation realization.
    merged = drive.merge(wait, on=["scenario_id", "replication_id"], how="inner")

    if merged.empty:
        stats["warnings"].append("Skipping driving-vs-wait scatter: no matched replication pairs.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in sorted(merged["scenario_id"].unique()):
        s = merged[merged["scenario_id"] == scenario]
        ax.scatter(s["total_wait_time_sum"], s["total_driving_time_sum"], label=f"Scenario {int(scenario)}", alpha=0.8)

    ax.set_title(scoped_title("Driving Time vs Wait Time by Scenario/Replication"))
    ax.set_xlabel("Total Wait Time per Replication (sum)")
    ax.set_ylabel("Total Driving Time per Replication (sum)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_scenarios_total_driving_vs_total_wait_scatter.png", dpi=180)
    plt.close(fig)


def report_missing_pairs(df, kind, stats):
    """Track missing expected scenario-replication combinations."""
    if df.empty:
        stats["warnings"].append(f"No rows loaded for {kind} data.")
        return

    known = set(tuple(x) for x in df[["scenario_id", "replication_id"]].dropna().astype(int).drop_duplicates().to_numpy())
    scenario_ids = sorted(df["scenario_id"].dropna().astype(int).unique())
    expected = {(s, r) for s in range(EXPECTED_SCENARIOS) for r in range(1, EXPECTED_REPLICATIONS + 1)}
    missing = expected - known
    if missing:
        stats["warnings"].append(f"{kind}: missing {len(missing)} expected scenario-replication pairs.")


def main():
    """Run the full analysis workflow and export plots plus summary metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    truck_files, bridge_files, stats = discover_files()
    truck_df = load_truck_driving(truck_files, stats)
    bridge_df = load_bridge_wait_times(bridge_files, stats)
    truck_df = filter_by_road(truck_df, stats, "truck")
    bridge_df = filter_by_road(bridge_df, stats, "bridge")

    report_missing_pairs(truck_df, "truck", stats)
    report_missing_pairs(bridge_df, "bridge", stats)

    # Aggregated metrics are the common analytical base for tabular export and
    # cross-metric comparisons.
    agg_df = aggregate_metrics(truck_df, bridge_df)
    agg_path = OUTPUT_DIR / "aggregated_metrics.csv"
    agg_df.to_csv(agg_path, index=False)

    plot_length_heatmaps(bridge_df, stats)
    plot_length_line(bridge_df, stats)
    plot_condition_heatmap(bridge_df, stats)
    plot_condition_line(bridge_df, stats)
    plot_driving_boxplot(truck_df, stats)
    plot_worst_bridges_dot(bridge_df, stats)
    plot_replication_heatmap(truck_df, stats)
    plot_wait_ecdf(bridge_df, stats)
    plot_pareto(bridge_df, stats)
    plot_driving_vs_wait(agg_df, stats)

    print("\n=== Compact Summary ===")
    print(f"road_selection: {road_scope_label()}")
    print(f"csv_found: {stats.get('csv_found', 0)}")
    print(f"truck_files_classified: {len(truck_files)}")
    print(f"bridge_files_classified: {len(bridge_files)}")
    print(f"truck_scenarios_found: {sorted({k[0] for k in truck_files.keys()})}")
    print(f"bridge_scenarios_found: {sorted({k[0] for k in bridge_files.keys()})}")
    print(f"truck_rows_loaded: {stats.get('truck_rows', 0)}")
    print(f"bridge_rows_loaded: {stats.get('bridge_rows', 0)}")
    print(f"warnings_count: {len(stats.get('warnings', []))}")
    print(f"aggregated_csv: {agg_path}")
    print(f"img_dir: {OUTPUT_DIR}")
    if stats["warnings"]:
        print("\nWarnings:")
        for w in stats["warnings"][:60]:
            print(f"- {w}")
        if len(stats["warnings"]) > 60:
            print(f"- ... {len(stats['warnings']) - 60} more warnings omitted.")


if __name__ == "__main__":
    main()
