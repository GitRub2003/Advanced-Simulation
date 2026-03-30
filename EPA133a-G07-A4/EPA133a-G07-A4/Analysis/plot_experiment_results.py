from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR
OUTPUT_DIR = EXPERIMENTS_DIR / "plots"
TOP_N = 10


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir() -> None:
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


def plot_delta_vs_baseline(delta_df: pd.DataFrame, infra_type: str = None, baseline_scenario: int = 0) -> None:
    scenarios = sorted(delta_df["scenario"].unique())
    scenarios = [s for s in scenarios if s != baseline_scenario]

    for scenario in scenarios:
        df = delta_df[delta_df["scenario"] == scenario].copy()

        if infra_type is not None:
            df = df[df["infra_type"].str.lower() == infra_type.lower()].copy()

        if df.empty:
            continue

        # Biggest rerouting changes, whether increase or decrease
        df["abs_delta"] = df["delta_vs_baseline"].abs()
        top_df = df.nlargest(TOP_N, "abs_delta").sort_values("delta_vs_baseline")

        plt.figure(figsize=(10, 6))
        plt.barh(top_df["infra_label"], top_df["delta_vs_baseline"])
        plt.xlabel("Change in mean crossings vs baseline")
        label = infra_type if infra_type is not None else "all infrastructure"
        plt.title(f"Top {TOP_N} rerouting changes for {label}: scenario {scenario} vs baseline {baseline_scenario}")
        plt.tight_layout()

        suffix = infra_type if infra_type is not None else "all"
        plt.savefig(
            OUTPUT_DIR / f"delta_vs_baseline_top_{TOP_N}_{suffix}_scenario_{scenario}.png",
            dpi=300
        )
        plt.close()


# ============================================================
# OPTIONAL: TRAVEL TIME DISTRIBUTIONS
# ============================================================

def plot_travel_time_histograms(travel_df: pd.DataFrame) -> None:
    scenarios = sorted(travel_df["scenario"].unique())

    for scenario in scenarios:
        df = travel_df[travel_df["scenario"] == scenario]

        if df.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(df["total_driving_time"].dropna(), bins=30)
        plt.xlabel("Truck total driving time")
        plt.ylabel("Frequency")
        plt.title(f"Travel time distribution - scenario {scenario}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"hist_travel_time_scenario_{scenario}.png", dpi=300)
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
    travel_summary.to_csv(OUTPUT_DIR / "travel_time_summary_by_replication.csv", index=False)
    cross_summary.to_csv(OUTPUT_DIR / "crossing_summary_by_replication.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "infrastructure_importance_by_scenario.csv", index=False)
    delta_df.to_csv(OUTPUT_DIR / "infrastructure_delta_vs_baseline.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
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

    plot_delta_vs_baseline(delta_df, infra_type=None, baseline_scenario=0)
    plot_delta_vs_baseline(delta_df, infra_type="bridge", baseline_scenario=0)
    plot_delta_vs_baseline(delta_df, infra_type="link", baseline_scenario=0)

    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()