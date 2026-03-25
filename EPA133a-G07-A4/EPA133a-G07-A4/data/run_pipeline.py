from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

PIPELINE = [
    {
        "name": "rmms_traffic",
        "script": "rmms prep.py",
        "expected_outputs": ["rmms_traffic_raw.csv"],
    },
    {
        "name": "side_roads",
        "script": "Side roads choosing.py",
        "expected_outputs": ["side_road_candidates.csv"],
    },
    {
        "name": "data_preparation",
        "script": "Data Preparation multiple roads.py",
        "expected_outputs": ["road_objects.csv"],
    },
    {
        "name": "intersection_creation",
        "script": "Intersection creation.py",
        "expected_outputs": [
            "intersections_raw_matches.csv",
            "intersections_detected.csv",
        ],
    },
    {
        "name": "insert_intersections",
        "script": "Intersections In the CSV.py",
        "expected_outputs": ["network_model.csv"],
    },
    {
        "name": "plot_map",
        "script": "roadmap plotting with bridges.py",
        "expected_outputs": ["network_map.png"],
    },
]


def run_script(script_name: str) -> None:
    script_path = BASE / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n[RUN ] {script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=BASE, check=True)
    print(f"[ OK ] {script_name}")


def verify_outputs(outputs: list[str]) -> None:
    missing = [name for name in outputs if not (BASE / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Expected output file(s) not found after step: " + ", ".join(missing)
        )


def main() -> None:
    print("Starting pipeline...")

    for step in PIPELINE:
        run_script(step["script"])
        verify_outputs(step["expected_outputs"])

    print("\nPipeline finished successfully.")
    print(f"Main output: {(BASE / 'network_model.csv').resolve()}")
    print(f"Map output:  {(BASE / 'network_map.png').resolve()}")


if __name__ == "__main__":
    main()
