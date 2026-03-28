from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd


FILE_REGEX = re.compile(
    r"^truck_driving_times_scenario_(?P<scenario>\d+)_replicate_(?P<replicate>\d+)\.csv$"
)
INFRA_FILE_REGEX = re.compile(
    r"^infrastructure_crossings_scenario_(?P<scenario>\d+)_replicate_(?P<replicate>\d+)\.csv$"
)


def load_endpoint_label_map(base_dir: Path) -> dict[str, str]:
    """Build a legacy-ID to road-endpoint label map from the network CSV."""
    network_path = base_dir / "data" / "network_model.csv"
    if not network_path.exists():
        return {}

    df = pd.read_csv(network_path)
    required_columns = {"id", "road", "model_type"}
    if not required_columns.issubset(df.columns):
        return {}

    df = df.copy()
    df["road"] = df["road"].astype(str).str.strip()
    df["model_type"] = df["model_type"].astype(str).str.strip().str.lower()

    label_map: dict[str, str] = {}
    for road_name, road_df in df.groupby("road", sort=False):
        road_df = road_df.reset_index(drop=True)
        endpoint_rows = road_df[
            road_df["model_type"].isin({"source", "sink", "sourcesink"})
        ].index.tolist()
        first_endpoint_index = endpoint_rows[0] if endpoint_rows else None
        last_endpoint_index = endpoint_rows[-1] if endpoint_rows else None

        for row_index, row in road_df.iterrows():
            if row["model_type"] not in {"source", "sink", "sourcesink"}:
                continue

            if row_index == first_endpoint_index:
                label = f"{road_name} start"
            elif row_index == last_endpoint_index:
                label = f"{road_name} end"
            else:
                label = road_name

            label_map[str(row["id"])] = label

    return label_map


def apply_endpoint_labels(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    """Replace numeric endpoint IDs with readable road-endpoint labels when possible."""
    if df.empty:
        return df

    df = df.copy()
    for column in ["source_id", "sink_id"]:
        df[column] = df[column].astype(str).map(label_map).fillna(df[column].astype(str))
    return df


def load_route_average_data(experiments_dir: Path) -> pd.DataFrame:
    """Load all truck driving-time CSVs and aggregate them to route-level averages per replicate."""
    label_map = load_endpoint_label_map(experiments_dir.parent)
    rows = []
    for csv_file in experiments_dir.glob("truck_driving_times_scenario_*_replicate_*.csv"):
        match = FILE_REGEX.match(csv_file.name)
        if not match:
            continue

        scenario_id = int(match.group("scenario"))
        replicate_id = int(match.group("replicate"))

        df = pd.read_csv(csv_file)
        required_columns = {"truck_id", "source_id", "sink_id", "total_driving_time"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {csv_file.name}. Expected {sorted(required_columns)}."
            )
        df = apply_endpoint_labels(df, label_map)

        # Collapse raw truck records into one row per route for this replicate.
        route_df = (
            df.groupby(["source_id", "sink_id"], as_index=False)
            .agg(
                total_driving_time_sum=("total_driving_time", "sum"),
                truck_count=("truck_id", "count"),
                average_driving_time=("total_driving_time", "mean"),
            )
        )
        route_df["scenario_id"] = scenario_id
        route_df["replicate_id"] = replicate_id
        rows.append(route_df)

    if not rows:
        raise FileNotFoundError(
            f"No truck driving-time files found in {experiments_dir} with pattern "
            f"'truck_driving_times_scenario_*_replicate_*.csv'."
        )

    data = pd.concat(rows, ignore_index=True)
    data["route"] = data["source_id"].astype(str) + " -> " + data["sink_id"].astype(str)
    return data


def load_infrastructure_crossing_data(experiments_dir: Path) -> pd.DataFrame:
    """Load all link/bridge-crossing CSVs and combine them into one table."""
    rows = []
    for csv_file in experiments_dir.glob("infrastructure_crossings_scenario_*_replicate_*.csv"):
        match = INFRA_FILE_REGEX.match(csv_file.name)
        if not match:
            continue

        scenario_id = int(match.group("scenario"))
        replicate_id = int(match.group("replicate"))

        df = pd.read_csv(csv_file)
        required_columns = {"infra_id", "infra_label", "infra_type", "crossing_count", "unique_truck_count"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {csv_file.name}. Expected {sorted(required_columns)}."
            )

        df = df.copy()
        df["scenario_id"] = scenario_id
        df["replicate_id"] = replicate_id
        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No infrastructure crossing files found in {experiments_dir} with pattern "
            f"'infrastructure_crossings_scenario_*_replicate_*.csv'."
        )

    return pd.concat(rows, ignore_index=True)


def compute_pct_increase_vs_baseline(all_data: pd.DataFrame) -> pd.DataFrame:
    """Compare each non-baseline scenario against scenario 0 on a route-by-route basis."""
    # First pool all replicates per scenario so each route gets one weighted scenario average.
    scenario_route_mean = (
        all_data.groupby(["scenario_id", "source_id", "sink_id"], as_index=False)
        .agg(
            total_time_sum=("total_driving_time_sum", "sum"),
            total_truck_count=("truck_count", "sum"),
        )
    )
    scenario_route_mean = scenario_route_mean[scenario_route_mean["total_truck_count"] > 0].copy()
    scenario_route_mean["scenario_avg_time"] = (
        scenario_route_mean["total_time_sum"] / scenario_route_mean["total_truck_count"]
    )

    # Scenario 0 is treated as the baseline for all percentage comparisons.
    baseline = scenario_route_mean[scenario_route_mean["scenario_id"] == 0][
        ["source_id", "sink_id", "scenario_avg_time"]
    ].rename(columns={"scenario_avg_time": "baseline_avg_time"})

    compare = scenario_route_mean[scenario_route_mean["scenario_id"] != 0].merge(
        baseline, on=["source_id", "sink_id"], how="inner"
    )

    # Keep only valid baseline routes, then compute the percentage increase for each comparison route.
    compare = compare[compare["baseline_avg_time"] > 0].copy()
    compare["pct_increase"] = (
        (compare["scenario_avg_time"] - compare["baseline_avg_time"]) / compare["baseline_avg_time"] * 100.0
    )
    compare["route"] = compare["source_id"].astype(str) + " -> " + compare["sink_id"].astype(str)
    compare.replace([np.inf, -np.inf], np.nan, inplace=True)
    compare.dropna(subset=["pct_increase"], inplace=True)
    return compare


def compute_replicate_means(all_data: pd.DataFrame) -> pd.DataFrame:
    """Compute one mean travel-time value per scenario/replicate pair."""
    grouped = all_data.groupby(["scenario_id", "replicate_id"], as_index=False)

    has_weighted_inputs = {"total_driving_time_sum", "truck_count"}.issubset(all_data.columns)
    if has_weighted_inputs:
        # Use weighted totals so busy routes contribute proportionally to the replicate mean.
        replicate_means = (
            grouped[["total_driving_time_sum", "truck_count"]]
            .sum()
            .rename(columns={"total_driving_time_sum": "replicate_total_time", "truck_count": "replicate_total_trucks"})
        )
        replicate_means = replicate_means[replicate_means["replicate_total_trucks"] > 0].copy()
        replicate_means["replicate_mean_travel_time"] = (
            replicate_means["replicate_total_time"] / replicate_means["replicate_total_trucks"]
        )
        return replicate_means

    # Fallback if historical files do not include sum/count.
    replicate_means = grouped["average_driving_time"].mean().rename(columns={"average_driving_time": "replicate_mean_travel_time"})
    replicate_means["replicate_total_trucks"] = np.nan
    return replicate_means


def compute_infrastructure_crossing_means(infra_data: pd.DataFrame) -> pd.DataFrame:
    """Average link/bridge usage over replicates for each scenario/infrastructure pair."""
    summary = (
        infra_data.groupby(["scenario_id", "infra_id", "infra_label", "infra_type"], as_index=False)
        .agg(
            mean_crossing_count=("crossing_count", "mean"),
            std_crossing_count=("crossing_count", "std"),
            mean_unique_truck_count=("unique_truck_count", "mean"),
        )
    )
    summary["std_crossing_count"] = summary["std_crossing_count"].fillna(0.0)
    return summary


def plot_average_travel_time_per_scenario(all_data: pd.DataFrame, output_dir: Path) -> Path:
    """Create a bar chart of mean travel time per scenario with replicate-level standard deviation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    replicate_means = compute_replicate_means(all_data)
    scenario_summary = (
        replicate_means.groupby("scenario_id", as_index=False)["replicate_mean_travel_time"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "scenario_mean", "std": "scenario_std"})
    )
    scenario_summary["scenario_std"] = scenario_summary["scenario_std"].fillna(0.0)
    scenario_summary = scenario_summary.sort_values("scenario_id")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        scenario_summary["scenario_id"].astype(str),
        scenario_summary["scenario_mean"],
        yerr=scenario_summary["scenario_std"],
        capsize=6,
        color="#0072B2",
        alpha=0.9,
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average travel time (minutes)")
    ax.set_title("Average travel time per scenario with standard-deviation error bars")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "average_travel_time_per_scenario_with_std.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_top10_increases(compare_df: pd.DataFrame, output_dir: Path, scenarios: list[int] | None = None) -> list[Path]:
    """Plot the ten routes with the largest average travel-time increase for each scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    if scenarios is None:
        # By default, generate one figure for every comparison scenario found in the data.
        scenarios = sorted(compare_df["scenario_id"].unique().tolist())

    for scenario_id in scenarios:
        scenario_df = compare_df[compare_df["scenario_id"] == scenario_id].copy()
        if scenario_df.empty:
            continue

        top10 = scenario_df.sort_values("pct_increase", ascending=False).head(10).sort_values("pct_increase")
        if top10.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.barh(top10["route"], top10["pct_increase"], color="#D55E00")
        ax.set_xlabel("Increase vs scenario 0 (%)")
        ax.set_ylabel("Route (source -> sink)")
        ax.set_title(f"Top 10 route average driving-time increases: scenario {scenario_id} vs scenario 0")
        ax.grid(axis="x", alpha=0.25)

        for idx, value in enumerate(top10["pct_increase"]):
            ax.text(value, idx, f" {value:.1f}%", va="center", fontsize=8)

        fig.tight_layout()
        output_path = output_dir / f"top10_route_avg_driving_time_increase_pct_scenario_{scenario_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        generated_paths.append(output_path)

    return generated_paths


def plot_route_increase_heatmaps(
    compare_df: pd.DataFrame, output_dir: Path, scenarios: list[int]
) -> list[Path]:
    """Create source-to-sink heatmaps showing route-level percentage change versus scenario 0."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for scenario_id in scenarios:
        scenario_df = compare_df[compare_df["scenario_id"] == scenario_id].copy()
        if scenario_df.empty:
            continue

        # Re-shape the route table into a matrix so source/sink pairs can be shown as a heatmap.
        matrix_df = (
            scenario_df.pivot(index="source_id", columns="sink_id", values="pct_increase")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        if matrix_df.empty:
            continue

        matrix_values = matrix_df.to_numpy(dtype=float)
        valid_values = matrix_values[np.isfinite(matrix_values)]
        if valid_values.size == 0:
            continue

        max_increase = float(np.max(valid_values))
        if max_increase == 0.0:
            max_increase = 1.0

        norm = colors.Normalize(vmin=0.0, vmax=max_increase)
        masked_values = np.ma.masked_invalid(matrix_values)

        fig_width = max(8.0, 0.6 * len(matrix_df.columns) + 3.0)
        fig_height = max(6.0, 0.5 * len(matrix_df.index) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        heat = ax.imshow(masked_values, cmap="Reds", norm=norm, aspect="auto")
        colorbar = fig.colorbar(heat, ax=ax)
        colorbar.set_label("Driving time increase vs scenario 0 (%)")

        ax.set_xticks(np.arange(len(matrix_df.columns)))
        ax.set_xticklabels(matrix_df.columns.astype(str), rotation=90)
        ax.set_yticks(np.arange(len(matrix_df.index)))
        ax.set_yticklabels(matrix_df.index.astype(str))
        ax.set_xlabel("Sink ID")
        ax.set_ylabel("Source ID")
        ax.set_title(f"Route driving-time increase matrix (%): scenario {scenario_id} vs scenario 0")

        # Annotate values only for smaller matrices to keep readability.
        if matrix_df.shape[0] <= 20 and matrix_df.shape[1] <= 20:
            for row_idx in range(matrix_df.shape[0]):
                for col_idx in range(matrix_df.shape[1]):
                    value = matrix_values[row_idx, col_idx]
                    if np.isfinite(value):
                        ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=7)

        fig.tight_layout()
        output_path = output_dir / f"heatmap_route_driving_time_increase_pct_scenario_{scenario_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        generated_paths.append(output_path)

    return generated_paths


def plot_top_infrastructure_crossings_per_scenario(
    infra_data: pd.DataFrame, output_dir: Path, top_n: int = 15
) -> list[Path]:
    """Plot the busiest links/bridges for each scenario using mean crossings over replicates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    infra_summary = compute_infrastructure_crossing_means(infra_data)
    for scenario_id in sorted(infra_summary["scenario_id"].unique().tolist()):
        scenario_df = infra_summary[infra_summary["scenario_id"] == scenario_id].copy()
        if scenario_df.empty:
            continue

        top_df = (
            scenario_df.sort_values("mean_crossing_count", ascending=False)
            .head(top_n)
            .sort_values("mean_crossing_count")
        )
        if top_df.empty:
            continue

        fig_height = max(6.0, 0.42 * len(top_df) + 1.5)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        ax.barh(
            top_df["infra_label"],
            top_df["mean_crossing_count"],
            xerr=top_df["std_crossing_count"],
            capsize=4,
            color="#009E73",
            alpha=0.9,
        )
        ax.set_xlabel("Average link/bridge crossings per replicate")
        ax.set_ylabel("Infrastructure")
        ax.set_title(f"Most-used links and bridges in scenario {scenario_id}")
        ax.grid(axis="x", alpha=0.25)

        for idx, value in enumerate(top_df["mean_crossing_count"]):
            ax.text(value, idx, f" {value:.1f}", va="center", fontsize=8)

        fig.tight_layout()
        output_path = output_dir / f"top_infrastructure_by_crossings_scenario_{scenario_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        generated_paths.append(output_path)

    return generated_paths


def plot_infrastructure_crossing_heatmap(
    infra_data: pd.DataFrame, output_dir: Path, top_n: int = 20
) -> Path | None:
    """Plot a scenario-by-infrastructure heatmap for the busiest links/bridges overall."""
    output_dir.mkdir(parents=True, exist_ok=True)

    infra_summary = compute_infrastructure_crossing_means(infra_data)
    if infra_summary.empty:
        return None

    top_infra = (
        infra_summary.groupby(["infra_id", "infra_label", "infra_type"], as_index=False)["mean_crossing_count"]
        .sum()
        .sort_values("mean_crossing_count", ascending=False)
        .head(top_n)
    )
    if top_infra.empty:
        return None

    filtered = infra_summary.merge(
        top_infra[["infra_id", "infra_label", "infra_type"]],
        on=["infra_id", "infra_label", "infra_type"],
        how="inner",
    )

    # Multiple infrastructure IDs can share the same display label, so collapse those
    # rows before pivoting to keep one value per scenario/label pair.
    filtered = (
        filtered.groupby(["scenario_id", "infra_label"], as_index=False)
        .agg(mean_crossing_count=("mean_crossing_count", "sum"))
    )

    matrix_df = filtered.pivot(
        index="scenario_id",
        columns="infra_label",
        values="mean_crossing_count",
    ).sort_index(axis=0)

    ordered_labels = top_infra["infra_label"].drop_duplicates().tolist()
    matrix_df = matrix_df.reindex(columns=ordered_labels)
    if matrix_df.empty:
        return None

    matrix_values = matrix_df.to_numpy(dtype=float)
    max_value = float(np.nanmax(matrix_values)) if np.isfinite(matrix_values).any() else 1.0
    if max_value <= 0.0:
        max_value = 1.0

    fig_width = max(10.0, 0.45 * len(matrix_df.columns) + 3.0)
    fig_height = max(4.5, 0.7 * len(matrix_df.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heat = ax.imshow(matrix_values, cmap="Blues", aspect="auto", vmin=0.0, vmax=max_value)
    colorbar = fig.colorbar(heat, ax=ax)
    colorbar.set_label("Average crossings per replicate")

    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns.astype(str), rotation=90)
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index.astype(str))
    ax.set_xlabel("Infrastructure")
    ax.set_ylabel("Scenario")
    ax.set_title("Link and bridge usage heatmap for busiest infrastructure")

    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            value = matrix_values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    output_path = output_dir / "heatmap_infrastructure_crossings_by_scenario.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    """Load experiment outputs, build the comparison tables, and save the resulting figures."""
    base_dir = Path(__file__).resolve().parents[1]
    experiments_dir = base_dir / "Experiments"
    img_dir = base_dir / "img"

    all_data = load_route_average_data(experiments_dir)
    output_paths = []
    compare_df = compute_pct_increase_vs_baseline(all_data)

    # The average travel-time bar chart includes all scenarios in the loaded dataset.
    output_paths.append(plot_average_travel_time_per_scenario(all_data, img_dir))
    output_paths.extend(plot_route_increase_heatmaps(compare_df, img_dir, scenarios=sorted(compare_df["scenario_id"].unique().tolist())))
    output_paths.extend(plot_top10_increases(compare_df, img_dir, scenarios=sorted(compare_df["scenario_id"].unique().tolist())))

    try:
        infra_data = load_infrastructure_crossing_data(experiments_dir)
    except FileNotFoundError:
        infra_data = None

    if infra_data is not None:
        heatmap_path = plot_infrastructure_crossing_heatmap(infra_data, img_dir)
        if heatmap_path is not None:
            output_paths.append(heatmap_path)
        output_paths.extend(plot_top_infrastructure_crossings_per_scenario(infra_data, img_dir))

    if len(output_paths) == 0:
        print("No plots generated. Check whether scenario 0 and the comparison scenarios share route pairs.")
        return

    print("Generated plots:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
