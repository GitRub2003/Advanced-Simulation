from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = BASE_DIR / "Experiments"
OUTPUT_DIR = BASE_DIR / "img"
TOP_N = 10


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir() -> None:
    """Create the output directory if it does not exist yet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_scenario_replicate(file_path: Path):
    """
    Extract scenario and replicate from filenames like:
    truck_driving_times_scenario_3_replicate_2.csv
    infrastructure_crossings_scenario_4_replicate_1.csv
    """
    match = re.search(r"scenario_(\d+)_replicate_(\d+)", file_path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def find_files(prefix: str):
    """
    Find all csv files with the given prefix.
    Example prefix:
    - 'truck_driving_times'
    - 'infrastructure_crossings'
    """
    return sorted(EXPERIMENTS_DIR.glob(f"{prefix}_scenario_*_replicate_*.csv"))


# ============================================================
# LOAD TRAVEL TIME DATA
# ============================================================

def load_travel_time_data() -> pd.DataFrame:
    """Load all travel-time CSVs and attach scenario and replicate identifiers."""
    files = find_files("truck_driving_times")
    all_dfs = []

    for file_path in files:
        scenario, replicate = extract_scenario_replicate(file_path)
        if scenario is None:
            continue

        df = pd.read_csv(file_path)
        df["scenario"] = scenario
        df["replicate"] = replicate

        # Safety: if total_driving_time is missing, compute it
        if "total_driving_time" not in df.columns:
            if {"generated_at_step", "removed_at_step"}.issubset(df.columns):
                df["total_driving_time"] = df["removed_at_step"] - df["generated_at_step"]
            else:
                raise ValueError(
                    f"{file_path.name} has no 'total_driving_time' column and cannot compute it."
                )

        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No truck_driving_times_scenario_*_replicate_*.csv files found.")

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# LOAD INFRASTRUCTURE CROSSINGS
# ============================================================

def load_crossing_data() -> pd.DataFrame:
    """Load all infrastructure-crossing CSVs and attach scenario and replicate identifiers."""
    files = find_files("infrastructure_crossings")
    all_dfs = []

    for file_path in files:
        scenario, replicate = extract_scenario_replicate(file_path)
        if scenario is None:
            continue

        df = pd.read_csv(file_path)
        df["scenario"] = scenario
        df["replicate"] = replicate
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No infrastructure_crossings_scenario_*_replicate_*.csv files found.")

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# SUMMARIES
# ============================================================

def build_travel_time_summary(travel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate travel-time metrics to one row per scenario and replication."""
    summary = (
        travel_df.groupby(["scenario", "replicate"])
        .agg(
            mean_travel_time=("total_driving_time", "mean"),
            median_travel_time=("total_driving_time", "median"),
            p95_travel_time=("total_driving_time", lambda s: s.quantile(0.95)),
            total_travel_time=("total_driving_time", "sum"),
            completed_trucks=("truck_id", "count"),
            mean_infra_crossings=("infra_crossing_count", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "replicate"])
    )
    return summary


def build_crossing_summary(cross_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw crossing events to infrastructure totals per scenario and replication."""
    summary = (
        cross_df.groupby(
            ["scenario", "replicate", "infra_id", "infra_label", "infra_type", "road_name"],
            dropna=False
        )
        .agg(
            crossing_count=("crossing_count", "sum"),
            unique_truck_count=("unique_truck_count", "sum"),
        )
        .reset_index()
    )
    return summary


def build_infrastructure_importance(cross_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across replications so you can rank bridges/links by usage per scenario.
    """
    importance = (
        cross_summary_df.groupby(
            ["scenario", "infra_id", "infra_label", "infra_type", "road_name"],
            dropna=False
        )
        .agg(
            mean_crossings=("crossing_count", "mean"),
            std_crossings=("crossing_count", "std"),
            mean_unique_trucks=("unique_truck_count", "mean"),
            n_replications=("replicate", "nunique"),
        )
        .reset_index()
    )

    importance["std_crossings"] = importance["std_crossings"].fillna(0)
    return importance


def build_baseline_comparison(importance_df: pd.DataFrame, baseline_scenario: int = 0) -> pd.DataFrame:
    """
    Compare each infrastructure's mean crossings to the baseline scenario.
    Positive delta = more traffic than baseline.
    Negative delta = less traffic than baseline.
    """
    baseline = importance_df[importance_df["scenario"] == baseline_scenario].copy()
    baseline = baseline[
        ["infra_id", "mean_crossings"]
    ].rename(columns={"mean_crossings": "baseline_mean_crossings"})

    merged = importance_df.merge(baseline, on="infra_id", how="left")
    merged["baseline_mean_crossings"] = merged["baseline_mean_crossings"].fillna(0)
    merged["delta_vs_baseline"] = merged["mean_crossings"] - merged["baseline_mean_crossings"]
    return merged


# ============================================================
# PLOTS: TRAVEL TIMES
# ============================================================

def plot_mean_travel_time_boxplot(summary_df: pd.DataFrame) -> None:
    """Plot the distribution of replicate mean travel times for each scenario."""
    scenarios = sorted(summary_df["scenario"].unique())
    data = [
        summary_df.loc[summary_df["scenario"] == s, "mean_travel_time"].dropna().tolist()
        for s in scenarios
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[f"Scenario {s}" for s in scenarios])
    plt.ylabel("Mean travel time per replication")
    plt.title("Distribution of mean truck travel time by scenario")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_mean_travel_time_by_scenario.png", dpi=300)
    plt.close()


def plot_total_travel_time_boxplot(summary_df: pd.DataFrame) -> None:
    """Plot the distribution of total completed-truck travel time per scenario."""
    scenarios = sorted(summary_df["scenario"].unique())
    data = [
        summary_df.loc[summary_df["scenario"] == s, "total_travel_time"].dropna().tolist()
        for s in scenarios
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[f"Scenario {s}" for s in scenarios])
    plt.ylabel("Total travel time across completed trucks")
    plt.title("Distribution of total travel time by scenario")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_total_travel_time_by_scenario.png", dpi=300)
    plt.close()


def plot_completed_trucks_boxplot(summary_df: pd.DataFrame) -> None:
    """Plot how many trucks finish per replication in each scenario."""
    scenarios = sorted(summary_df["scenario"].unique())
    data = [
        summary_df.loc[summary_df["scenario"] == s, "completed_trucks"].dropna().tolist()
        for s in scenarios
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[f"Scenario {s}" for s in scenarios])
    plt.ylabel("Completed trucks per replication")
    plt.title("Distribution of completed trucks by scenario")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_completed_trucks_by_scenario.png", dpi=300)
    plt.close()


def plot_mean_travel_time_bar(summary_df: pd.DataFrame) -> None:
    """Plot average mean travel time per scenario with replicate-level variation."""
    agg = (
        summary_df.groupby("scenario")
        .agg(
            mean_travel_time=("mean_travel_time", "mean"),
            std_travel_time=("mean_travel_time", "std"),
        )
        .reset_index()
        .sort_values("scenario")
    )

    agg["std_travel_time"] = agg["std_travel_time"].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.bar(agg["scenario"].astype(str), agg["mean_travel_time"], yerr=agg["std_travel_time"])
    plt.xlabel("Scenario")
    plt.ylabel("Average mean travel time")
    plt.title("Average travel time by scenario")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bar_mean_travel_time_by_scenario.png", dpi=300)
    plt.close()


# ============================================================
# PLOTS: INFRASTRUCTURE IMPORTANCE / CRITICALITY SUPPORT
# ============================================================

def plot_top_infrastructure_by_scenario(importance_df: pd.DataFrame, infra_type: str = None) -> None:
    """Plot the busiest infrastructure items in each scenario for the requested type."""
    scenarios = sorted(importance_df["scenario"].unique())

    for scenario in scenarios:
        df = importance_df[importance_df["scenario"] == scenario].copy()

        if infra_type is not None:
            df = df[df["infra_type"].str.lower() == infra_type.lower()].copy()

        if df.empty:
            continue

        top_df = df.nlargest(TOP_N, "mean_crossings").sort_values("mean_crossings")

        plt.figure(figsize=(10, 6))
        plt.barh(top_df["infra_label"], top_df["mean_crossings"])
        plt.xlabel("Mean crossings across replications")
        label = infra_type if infra_type is not None else "all infrastructure"
        plt.title(f"Top {TOP_N} {label} in scenario {scenario}")
        plt.tight_layout()

        suffix = infra_type if infra_type is not None else "all"
        plt.savefig(
            OUTPUT_DIR / f"top_{TOP_N}_{suffix}_scenario_{scenario}.png",
            dpi=300
        )
        plt.close()


def plot_delta_vs_baseline_heatmap(
    delta_df: pd.DataFrame,
    infra_type: str,
    baseline_scenario: int = 0,
) -> None:
    """Plot a heatmap of baseline differences for the most relevant bridges or links."""
    type_df = delta_df[delta_df["infra_type"].str.lower() == infra_type.lower()].copy()
    if type_df.empty:
        return

    scenarios = sorted(type_df["scenario"].unique())
    scenarios = [s for s in scenarios if s != baseline_scenario]
    if not scenarios:
        return

    scenario_df = type_df[type_df["scenario"].isin(scenarios)].copy()
    scenario_df["abs_delta"] = scenario_df["delta_vs_baseline"].abs()

    relevance = (
        scenario_df.groupby(["infra_id", "infra_label"], as_index=False)
        .agg(
            max_abs_delta=("abs_delta", "max"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .sort_values(["max_abs_delta", "mean_abs_delta", "infra_label"], ascending=[False, False, True])
        .head(TOP_N)
    )
    if relevance.empty:
        return

    top_ids = relevance["infra_id"].tolist()
    top_labels = relevance["infra_label"].tolist()
    filtered = scenario_df[scenario_df["infra_id"].isin(top_ids)].copy()

    delta_matrix = (
        filtered.pivot_table(index="scenario", columns="infra_id", values="delta_vs_baseline", aggfunc="first")
        .reindex(index=scenarios, columns=top_ids)
    )
    scenario_matrix = (
        filtered.pivot_table(index="scenario", columns="infra_id", values="mean_crossings", aggfunc="first")
        .reindex(index=scenarios, columns=top_ids)
    )
    baseline_matrix = (
        filtered.pivot_table(index="scenario", columns="infra_id", values="baseline_mean_crossings", aggfunc="first")
        .reindex(index=scenarios, columns=top_ids)
    )

    vmax = float(delta_matrix.abs().max().max()) if not delta_matrix.empty else 0.0
    if vmax == 0:
        vmax = 1.0

    fig_width = max(12, 1.2 * len(top_ids) + 3)
    fig_height = max(5, 0.9 * len(scenarios) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(delta_matrix.fillna(0).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(top_ids)))
    ax.set_xticklabels(top_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"Scenario {s}" for s in scenarios])
    ax.set_xlabel(f"Top {TOP_N} {infra_type}s by strongest change vs baseline")
    ax.set_ylabel("Scenario")
    ax.set_title(f"{infra_type.capitalize()} mean crossings vs baseline scenario {baseline_scenario}")

    for row_idx, scenario_id in enumerate(scenarios):
        for col_idx, infra_id in enumerate(top_ids):
            delta_val = delta_matrix.loc[scenario_id, infra_id]
            if pd.isna(delta_val):
                label = "-"
                text_color = "black"
            else:
                mean_val = scenario_matrix.loc[scenario_id, infra_id]
                baseline_val = baseline_matrix.loc[scenario_id, infra_id]
                label = f"{delta_val:+.1f}\n{mean_val:.1f}|{baseline_val:.1f}"
                text_color = "white" if abs(float(delta_val)) > 0.55 * vmax else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Mean crossings minus baseline")

    note = "Cell annotation: delta on first line, scenario|baseline mean crossings on second line"
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"delta_vs_baseline_top_{TOP_N}_{infra_type}_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()



# ============================================================
# SAVE TABLES
# ============================================================

def save_tables(
    travel_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
    importance_df: pd.DataFrame,
    delta_df: pd.DataFrame
) -> None:
    """Write the main summary tables used by the rest of the analysis workflow."""
    travel_summary.to_csv(OUTPUT_DIR / "travel_time_summary_by_replication.csv", index=False)
    cross_summary.to_csv(OUTPUT_DIR / "crossing_summary_by_replication.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "infrastructure_importance_by_scenario.csv", index=False)
    delta_df.to_csv(OUTPUT_DIR / "infrastructure_delta_vs_baseline.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the full post-processing workflow from raw experiment CSVs to plots."""
    ensure_output_dir()

    print("Loading travel time data...")
    travel_df = load_travel_time_data()

    print("Loading infrastructure crossing data...")
    cross_df = load_crossing_data()

    print("Building summaries...")
    travel_summary = build_travel_time_summary(travel_df)
    cross_summary = build_crossing_summary(cross_df)
    importance_df = build_infrastructure_importance(cross_summary)
    delta_df = build_baseline_comparison(importance_df, baseline_scenario=0)

    print("Saving summary tables...")
    save_tables(travel_summary, cross_summary, importance_df, delta_df)

    print("Making travel time plots...")
    plot_mean_travel_time_boxplot(travel_summary)
    plot_total_travel_time_boxplot(travel_summary)
    plot_completed_trucks_boxplot(travel_summary)
    plot_mean_travel_time_bar(travel_summary)
    plot_travel_time_histograms(travel_df)

    print("Making infrastructure plots...")
    plot_top_infrastructure_by_scenario(importance_df, infra_type=None)
    plot_top_infrastructure_by_scenario(importance_df, infra_type="bridge")
    plot_top_infrastructure_by_scenario(importance_df, infra_type="link")

    plot_delta_vs_baseline_heatmap(delta_df, infra_type="bridge", baseline_scenario=0)
    plot_delta_vs_baseline_heatmap(delta_df, infra_type="link", baseline_scenario=0)

    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
