"""
Build experiment statistics and publication-ready PNG visualizations.

Outputs:
    - PNG figures saved in ./img
    - Tidy aggregated CSV saved as ./img/aggregated_metrics.csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------- Configuration -------------------------------
PROJECT_ROOT = Path(".").resolve()
OUTPUT_DIR = PROJECT_ROOT / "img"
EXPERIMENTS_DIR = PROJECT_ROOT / "EPA133a-Lab_Assignment_2" / "EPA133a-G07-A2" / "Experiments"
EXPECTED_SCENARIOS = 8
EXPECTED_REPLICATIONS = 10
TOP_N_WORST_BRIDGES = 5
TOP_K_PARETO = 20

TRUCK_FILE_RE = re.compile(
    r"^truck_driving_times_scenario_(\d+)_replicate_(\d+)\.csv$",
    flags=re.IGNORECASE,
)
BRIDGE_FILE_RE = re.compile(
    r"^bridge_total_wait_times_scenario_(\d+)_replicate_(\d+)\.csv$",
    flags=re.IGNORECASE,
)

BRIDGE_REQUIRED_COLS = ["bridge_id", "bridge_name", "road_name", "condition", "total_wait_time"]


def discover_files(project_root):
    stats = {
        "csv_found": 0,
        "truck_files": 0,
        "bridge_files": 0,
    }

    experiments_dir = EXPERIMENTS_DIR

    truck_files = {}
    bridge_files = {}
    for csv_path in experiments_dir.rglob("*.csv"):
        stats["csv_found"] += 1
        fname = csv_path.name
        truck_m = TRUCK_FILE_RE.match(fname)
        bridge_m = BRIDGE_FILE_RE.match(fname)

        if truck_m:
            scenario_id = int(truck_m.group(1))
            replication_id = int(truck_m.group(2))
            stats["truck_files"] += 1
            truck_files[(scenario_id, replication_id)] = csv_path
        elif bridge_m:
            scenario_id = int(bridge_m.group(1))
            replication_id = int(bridge_m.group(2))
            stats["bridge_files"] += 1
            bridge_files[(scenario_id, replication_id)] = csv_path
    return truck_files, bridge_files, stats


def load_truck_driving(files, stats) -> pd.DataFrame:
    records = []
    for (scenario_id, replication_id), file_path in files.items():

        df = pd.read_csv(file_path)

        d = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "replication_id": replication_id,
                "source_file": str(file_path),
                "total_driving_time": pd.to_numeric(df["total_driving_time"], errors="coerce"),
            }
        )
        records.append(d.dropna(subset=["total_driving_time"]))

    if not records:
        return pd.DataFrame(columns=["scenario_id", "replication_id", "source_file", "total_driving_time"])
    out = pd.concat(records, ignore_index=True)
    stats["truck_rows"] = int(len(out))
    return out


def _normalize_condition_series(s: pd.Series) -> pd.Series:
    def convert(v: object) -> Optional[str]:
        if pd.isna(v):
            return None
        x = str(v).strip().upper()
        m = re.search(r"\b([ABCD])\b", x)
        if m:
            return m.group(1)
        mapping = {
            "VERY GOOD": "A",
            "GOOD": "B",
            "FAIR": "C",
            "POOR": "D",
            "VERY POOR": "D",
            "BAD": "D",
        }
        for k, val in mapping.items():
            if k in x:
                return val
        return None

    return s.map(convert)


def _read_length_category(df: pd.DataFrame, stats: Dict, file_path: Path) -> pd.Series:
    if "length_category" in df.columns:
        return df["length_category"].astype(str).str.strip()
    stats["warnings"].append(
        f"Length plots skipped for {file_path}: 'length_category' column not present; columns={list(df.columns)}"
    )
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def load_bridge_wait_times(files: Dict[Tuple[int, int], Path], stats: Dict) -> pd.DataFrame:
    records = []
    for (scenario_id, replication_id), file_path in files.items():
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            stats["warnings"].append(f"Failed reading bridge file {file_path}: {e}")
            continue

        missing_cols = [c for c in BRIDGE_REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            stats["warnings"].append(
                f"Bridge file missing required columns {missing_cols}: {file_path} | columns={list(df.columns)}"
            )
            continue

        out = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "replication_id": replication_id,
                "source_file": str(file_path),
                "bridge_id": df["bridge_id"].astype(str).str.strip(),
                "bridge_name": df["bridge_name"].astype(str).str.strip(),
                "road_name": df["road_name"].astype(str).str.strip(),
                "total_wait_time": pd.to_numeric(df["total_wait_time"], errors="coerce"),
                "condition_category": _normalize_condition_series(df["condition"]),
                "length_category": _read_length_category(df, stats, file_path),
                "lat": pd.to_numeric(df["lat"], errors="coerce") if "lat" in df.columns else np.nan,
                "lon": pd.to_numeric(df["lon"], errors="coerce") if "lon" in df.columns else np.nan,
            }
        )

        bad = out["total_wait_time"].isna().sum()
        if bad > 0:
            stats["warnings"].append(f"{bad} non-numeric wait-time values in {file_path}")

        records.append(out.dropna(subset=["total_wait_time"]))

    if not records:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "replication_id",
                "source_file",
                "bridge_id",
                "bridge_name",
                "road_name",
                "total_wait_time",
                "condition_category",
                "length_category",
                "lat",
                "lon",
            ]
        )
    out = pd.concat(records, ignore_index=True)
    stats["bridge_rows"] = int(len(out))
    return out


def aggregate_metrics(truck_df: pd.DataFrame, bridge_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    if not truck_df.empty:
        td = truck_df.groupby(["scenario_id", "replication_id"], dropna=False)["total_driving_time"].agg(
            rows="size", sum_value="sum", mean_value="mean", median_value="median"
        )
        td = td.reset_index()
        td["metric_group"] = "driving_time"
        parts.append(td)

    if not bridge_df.empty:
        bw = bridge_df.groupby(["scenario_id", "replication_id"], dropna=False)["total_wait_time"].agg(
            rows="size", sum_value="sum", mean_value="mean", median_value="median"
        )
        bw = bw.reset_index()
        bw["metric_group"] = "bridge_wait_time"
        parts.append(bw)

    if not parts:
        return pd.DataFrame(
            columns=["scenario_id", "replication_id", "metric_group", "rows", "sum_value", "mean_value", "median_value"]
        )
    return pd.concat(parts, ignore_index=True)


def _save_heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, out_path: Path, fmt: str = ".1f") -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    arr = table.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index.astype(str))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_length_heatmaps(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "length_category"]).copy()
    if d.empty:
        stats["warnings"].append("Skipping length heatmaps: no length_category data.")
        return

    share_wait = (
        d.groupby(["scenario_id", "length_category"])["total_wait_time"]
        .sum()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum() if x.sum() != 0 else x * 0)
        .unstack(fill_value=0)
        .sort_index()
    )
    _save_heatmap(
        share_wait,
        "Share of Total Wait Time by Bridge Length Category and Scenario",
        "Bridge Length Category",
        "Scenario",
        out_dir / "all_scenarios_total_wait_time_length_category_heatmap_wait_share.png",
    )

    share_events = (
        d.groupby(["scenario_id", "length_category"])["total_wait_time"]
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum() if x.sum() != 0 else x * 0)
        .unstack(fill_value=0)
        .sort_index()
    )
    _save_heatmap(
        share_events,
        "Share of Bridge Wait Events by Length Category and Scenario",
        "Bridge Length Category",
        "Scenario",
        out_dir / "all_scenarios_total_wait_time_length_category_heatmap_event_share.png",
    )


def plot_length_line(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "length_category"]).copy()
    if d.empty:
        stats["warnings"].append("Skipping length line plot: no length_category data.")
        return
    g = (
        d.groupby(["scenario_id", "length_category"])["total_wait_time"]
        .mean()
        .reset_index()
    )
    categories = sorted(g["length_category"].dropna().unique(), key=lambda x: str(x))
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid in sorted(g["scenario_id"].dropna().unique()):
        s = g[g["scenario_id"] == sid].set_index("length_category").reindex(categories)
        ax.plot(categories, s["total_wait_time"], marker="o", label=f"Scenario {int(sid)}")
    ax.set_xlabel("Bridge Length Category")
    ax.set_ylabel("Average Total Wait Time")
    ax.set_title("Average Bridge Wait Time by Length Category")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_wait_time_length_category_line_mean.png", dpi=180)
    plt.close(fig)


def plot_condition_heatmap(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "condition_category"]).copy()
    if d.empty:
        stats["warnings"].append("Skipping condition heatmap: no condition_category data.")
        return
    d = d[d["condition_category"].isin(["A", "B", "C", "D"])]
    if d.empty:
        stats["warnings"].append("Skipping condition heatmap: condition values not mappable to A-D.")
        return
    table = (
        d.groupby(["scenario_id", "condition_category"])["total_wait_time"]
        .sum()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum() if x.sum() != 0 else x * 0)
        .unstack(fill_value=0)
        .reindex(columns=["A", "B", "C", "D"], fill_value=0)
        .sort_index()
    )
    _save_heatmap(
        table,
        "Share of Total Wait Time by Condition Category and Scenario",
        "Condition Category",
        "Scenario",
        out_dir / "all_scenarios_total_wait_time_condition_heatmap_wait_share.png",
    )


def plot_condition_line(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "condition_category"]).copy()
    d = d[d["condition_category"].isin(["A", "B", "C", "D"])]
    if d.empty:
        stats["warnings"].append("Skipping condition line plot: no A-D condition data.")
        return
    g = d.groupby(["scenario_id", "condition_category"])["total_wait_time"].mean().reset_index()
    cats = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid in sorted(g["scenario_id"].dropna().unique()):
        s = g[g["scenario_id"] == sid].set_index("condition_category").reindex(cats)
        ax.plot(cats, s["total_wait_time"], marker="o", label=f"Scenario {int(sid)}")
    ax.set_xlabel("Condition Category")
    ax.set_ylabel("Average Total Wait Time")
    ax.set_title("Average Bridge Wait Time by Condition Category")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_wait_time_condition_line_mean.png", dpi=180)
    plt.close(fig)


def plot_driving_boxplot(truck_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if truck_df.empty:
        return
    rep = (
        truck_df.groupby(["scenario_id", "replication_id"], dropna=False)["total_driving_time"]
        .sum()
        .reset_index()
    )
    rep = rep.dropna(subset=["scenario_id", "replication_id"])
    if rep.empty:
        stats["warnings"].append("Skipping driving boxplot: missing scenario/replication IDs.")
        return
    scenarios = sorted(rep["scenario_id"].astype(int).unique())
    data = [rep.loc[rep["scenario_id"].astype(int) == sid, "total_driving_time"].values for sid in scenarios]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, labels=[str(s) for s in scenarios], showmeans=True)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Driving Time per Replication (sum)")
    ax.set_title("Travel Time Distribution per Scenario (10 Replications Expected)")
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_driving_time_boxplot_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_worst_bridges_dot(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict, top_n: int = TOP_N_WORST_BRIDGES) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "bridge_id"]).copy()
    d = d[d["scenario_id"].between(1, 7, inclusive="both")]
    if d.empty:
        stats["warnings"].append("Skipping worst-bridges dot plot: no scenario 1-7 bridge-id data.")
        return
    agg = d.groupby("bridge_id")["total_wait_time"].agg(mean_wait="mean", std_wait="std", n="size").sort_values(
        "mean_wait", ascending=False
    )
    top = agg.head(top_n).sort_values("mean_wait", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(top))
    ax.errorbar(top["mean_wait"], y, xerr=top["std_wait"].fillna(0), fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index.astype(str))
    ax.set_xlabel("Mean Total Wait Time (Scenarios 1-7)")
    ax.set_ylabel("Bridge")
    ax.set_title(f"Top {min(top_n, len(top))} Worst Bridges by Mean Wait Time")
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_1_7_total_wait_time_worst_bridges_dotplot.png", dpi=180)
    plt.close(fig)


def plot_wait_map_like(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["bridge_id", "lat", "lon"]).copy()
    if d.empty:
        stats["warnings"].append("Skipping map-like plot: no usable lat/lon columns.")
        return
    g = d.groupby("bridge_id", as_index=False).agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        mean_wait=("total_wait_time", "mean"),
    )
    g = g.dropna(subset=["lat", "lon", "mean_wait"])
    if g.empty:
        stats["warnings"].append("Skipping map-like plot: no valid aggregated lat/lon/wait values.")
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(g["lon"], g["lat"], c=g["mean_wait"], s=np.clip(g["mean_wait"], 10, None) * 0.8, alpha=0.75)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean Total Wait Time")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Bridge Wait Time Spatial Pattern (Map-like Scatter)")
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_wait_time_bridge_map_scatter.png", dpi=180)
    plt.close(fig)


def plot_replication_heatmap(truck_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if truck_df.empty:
        return
    rep = (
        truck_df.groupby(["scenario_id", "replication_id"])["total_driving_time"]
        .sum()
        .unstack(fill_value=np.nan)
        .sort_index()
    )
    if rep.empty:
        return
    _save_heatmap(
        rep,
        "Total Driving Time per Scenario and Replication",
        "Replication",
        "Scenario",
        out_dir / "all_scenarios_total_driving_time_replication_heatmap.png",
    )


def plot_wait_ecdf(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "total_wait_time"]).copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid in sorted(d["scenario_id"].dropna().unique()):
        vals = np.sort(d.loc[d["scenario_id"] == sid, "total_wait_time"].to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=f"Scenario {int(sid)}", alpha=0.9)
    ax.set_xlabel("Total Wait Time")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of Bridge Wait Times by Scenario")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_wait_time_ecdf_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_pareto_topk(bridge_df: pd.DataFrame, out_dir: Path, stats: Dict, top_k: int = TOP_K_PARETO) -> None:
    if bridge_df.empty:
        return
    d = bridge_df.dropna(subset=["scenario_id", "bridge_id"]).copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid in sorted(d["scenario_id"].dropna().unique()):
        s = d[d["scenario_id"] == sid].groupby("bridge_id")["total_wait_time"].sum().sort_values(ascending=False)
        if s.empty:
            continue
        top = s.head(top_k)
        cum_share = top.cumsum() / s.sum() * 100 if s.sum() != 0 else top * 0
        ax.plot(np.arange(1, len(cum_share) + 1), cum_share.values, marker="o", label=f"Scenario {int(sid)}")
    ax.set_xlabel(f"Top-K Bridges (K <= {top_k})")
    ax.set_ylabel("Cumulative Share of Total Wait Time (%)")
    ax.set_title("Pareto Concentration of Bridge Wait Time by Scenario")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_wait_time_pareto_topk_by_scenario.png", dpi=180)
    plt.close(fig)


def plot_driving_vs_wait_scatter(agg_df: pd.DataFrame, out_dir: Path, stats: Dict) -> None:
    if agg_df.empty:
        return
    d_drive = agg_df[agg_df["metric_group"] == "driving_time"][["scenario_id", "replication_id", "sum_value"]].rename(
        columns={"sum_value": "total_driving_time_sum"}
    )
    d_wait = agg_df[agg_df["metric_group"] == "bridge_wait_time"][["scenario_id", "replication_id", "sum_value"]].rename(
        columns={"sum_value": "total_wait_time_sum"}
    )
    merged = d_drive.merge(d_wait, on=["scenario_id", "replication_id"], how="inner")
    merged = merged.dropna(subset=["total_driving_time_sum", "total_wait_time_sum"])
    if merged.empty:
        stats["warnings"].append("Skipping driving-vs-wait scatter: no matched scenario-replication aggregates.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = sorted(merged["scenario_id"].dropna().unique())
    for sid in scenarios:
        s = merged[merged["scenario_id"] == sid]
        ax.scatter(s["total_wait_time_sum"], s["total_driving_time_sum"], label=f"Scenario {int(sid)}", alpha=0.8)
    ax.set_xlabel("Total Wait Time per Replication (sum)")
    ax.set_ylabel("Total Driving Time per Replication (sum)")
    ax.set_title("Driving Time vs Wait Time by Scenario/Replication")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_scenarios_total_driving_vs_total_wait_scatter.png", dpi=180)
    plt.close(fig)


def _missing_pairs_report(df: pd.DataFrame, kind: str, stats: Dict) -> None:
    if df.empty:
        stats["warnings"].append(f"No rows loaded for {kind} data.")
        return
    if df["scenario_id"].isna().any() or df["replication_id"].isna().any():
        stats["warnings"].append(f"{kind}: rows with missing scenario_id/replication_id present.")
    known = (
        df.dropna(subset=["scenario_id", "replication_id"])[["scenario_id", "replication_id"]]
        .astype(int)
        .drop_duplicates()
    )
    expected = {(s, r) for s in range(EXPECTED_SCENARIOS) for r in range(EXPECTED_REPLICATIONS)}
    present = set(map(tuple, known.to_records(index=False)))
    missing = sorted(expected - present)
    if missing:
        stats["warnings"].append(f"{kind}: missing {len(missing)} expected scenario-replication pairs.")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    truck_files, bridge_files, stats = discover_files(PROJECT_ROOT)
    truck_df = load_truck_driving(truck_files, stats)
    bridge_df = load_bridge_wait_times(bridge_files, stats)

    _missing_pairs_report(truck_df, "truck", stats)
    _missing_pairs_report(bridge_df, "bridge", stats)

    agg_df = aggregate_metrics(truck_df, bridge_df)
    agg_csv = OUTPUT_DIR / "aggregated_metrics.csv"
    agg_df.to_csv(agg_csv, index=False)

    plot_length_heatmaps(bridge_df, OUTPUT_DIR, stats)
    plot_length_line(bridge_df, OUTPUT_DIR, stats)
    plot_condition_heatmap(bridge_df, OUTPUT_DIR, stats)
    plot_condition_line(bridge_df, OUTPUT_DIR, stats)
    plot_driving_boxplot(truck_df, OUTPUT_DIR, stats)
    plot_worst_bridges_dot(bridge_df, OUTPUT_DIR, stats)
    plot_wait_map_like(bridge_df, OUTPUT_DIR, stats)

    plot_replication_heatmap(truck_df, OUTPUT_DIR, stats)
    plot_wait_ecdf(bridge_df, OUTPUT_DIR, stats)
    plot_pareto_topk(bridge_df, OUTPUT_DIR, stats)
    plot_driving_vs_wait_scatter(agg_df, OUTPUT_DIR, stats)

    summary = {
        "csv_found": stats.get("csv_found", 0),
        "truck_files_classified": len(truck_files),
        "bridge_files_classified": len(bridge_files),
        "truck_rows_loaded": stats.get("truck_rows", 0),
        "bridge_rows_loaded": stats.get("bridge_rows", 0),
        "warnings_count": len(stats.get("warnings", [])),
        "aggregated_csv": str(agg_csv),
        "img_dir": str(OUTPUT_DIR),
    }
    print("\n=== Compact Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if stats.get("warnings"):
        print("\nWarnings:")
        for w in stats["warnings"][:60]:
            print(f"- {w}")
        if len(stats["warnings"]) > 60:
            print(f"- ... {len(stats['warnings']) - 60} more warnings omitted.")


if __name__ == "__main__":
    main()
