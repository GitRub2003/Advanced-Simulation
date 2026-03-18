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


def load_route_average_data(experiments_dir: Path) -> pd.DataFrame:
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


def compute_pct_increase_vs_baseline(all_data: pd.DataFrame) -> pd.DataFrame:
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

    baseline = scenario_route_mean[scenario_route_mean["scenario_id"] == 0][
        ["source_id", "sink_id", "scenario_avg_time"]
    ].rename(columns={"scenario_avg_time": "baseline_avg_time"})

    compare = scenario_route_mean[scenario_route_mean["scenario_id"] != 0].merge(
        baseline, on=["source_id", "sink_id"], how="inner"
    )

    compare = compare[compare["baseline_avg_time"] > 0].copy()
    compare["pct_increase"] = (
        (compare["scenario_avg_time"] - compare["baseline_avg_time"]) / compare["baseline_avg_time"] * 100.0
    )
    compare["route"] = compare["source_id"].astype(str) + " -> " + compare["sink_id"].astype(str)
    compare.replace([np.inf, -np.inf], np.nan, inplace=True)
    compare.dropna(subset=["pct_increase"], inplace=True)
    return compare


def compute_replicate_means(all_data: pd.DataFrame) -> pd.DataFrame:
    grouped = all_data.groupby(["scenario_id", "replicate_id"], as_index=False)

    has_weighted_inputs = {"total_driving_time_sum", "truck_count"}.issubset(all_data.columns)
    if has_weighted_inputs:
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


def plot_average_travel_time_per_scenario(all_data: pd.DataFrame, output_dir: Path) -> Path:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    if scenarios is None:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for scenario_id in scenarios:
        scenario_df = compare_df[compare_df["scenario_id"] == scenario_id].copy()
        if scenario_df.empty:
            continue

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

        max_abs = float(np.max(np.abs(valid_values)))
        if max_abs == 0.0:
            max_abs = 1.0

        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        masked_values = np.ma.masked_invalid(matrix_values)

        fig_width = max(8.0, 0.6 * len(matrix_df.columns) + 3.0)
        fig_height = max(6.0, 0.5 * len(matrix_df.index) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        heat = ax.imshow(masked_values, cmap="RdBu_r", norm=norm, aspect="auto")
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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    experiments_dir = base_dir / "Experiments"
    img_dir = base_dir / "img"

    all_data = load_route_average_data(experiments_dir)
    compare_df = compute_pct_increase_vs_baseline(all_data)
    output_paths = []
    output_paths.append(plot_average_travel_time_per_scenario(all_data, img_dir))
    output_paths.extend(plot_route_increase_heatmaps(compare_df, img_dir, scenarios=[1, 2, 3]))
    output_paths.extend(plot_top10_increases(compare_df, img_dir, scenarios=[1, 2, 3]))

    if len(output_paths) == 0:
        print("No plots generated. Check whether scenario 0 and scenarios 1-3 share route pairs.")
        return

    print("Generated plots:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
