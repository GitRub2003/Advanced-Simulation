from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SIDE_ROADS_FILE = Path("side_road_candidates.csv")
BASE_ROADS = ("N1", "N2")
REQUIRED_ROADS = ("N106",)
SUPPORT_ROADS_BY_ROAD = {
    "N106": ("R160", "Z1619"),
}

ROAD_REF_PATTERN = re.compile(r"\b([NRZS]\d{1,4})\b", re.IGNORECASE)


def normalize_roads(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        road = str(value).strip().upper()
        if not road or road in seen:
            continue
        seen.add(road)
        ordered.append(road)

    return ordered


def get_required_support_roads(selected_roads: list[str]) -> list[str]:
    support_roads: list[str] = []

    for road in normalize_roads(selected_roads):
        support_roads.extend(SUPPORT_ROADS_BY_ROAD.get(road, ()))

    return normalize_roads(support_roads)


def extract_road_references(text: str) -> set[str]:
    return {match.upper() for match in ROAD_REF_PATTERN.findall(str(text))}


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def find_connector_roads(
    roads: pd.DataFrame,
    target_roads: list[str],
    allowed_prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Return roads that are explicitly referenced by, or explicitly reference,
    one of the target roads in the source metadata.
    """
    target_set = set(normalize_roads(target_roads))
    if not target_set:
        return []

    work = roads.copy()
    _require_columns(work, ["road"], "roads")

    if "name" not in work.columns:
        work["name"] = ""

    work["road"] = work["road"].astype(str).str.strip().str.upper()
    work["name"] = work["name"].fillna("").astype(str)

    known_roads = set(work["road"].dropna().unique())
    connectors: set[str] = set()

    for road_name, group in work.groupby("road", sort=False):
        refs: set[str] = set()
        for value in group["name"]:
            refs.update(extract_road_references(value))
        refs &= known_roads

        if road_name in target_set:
            connectors.update(refs - target_set)
        elif refs & target_set:
            connectors.add(road_name)

    if allowed_prefixes is not None:
        prefixes = tuple(prefix.upper() for prefix in allowed_prefixes)
        connectors = {road for road in connectors if road.startswith(prefixes)}

    return normalize_roads(connectors)


def load_selected_roads(
    path: Path = SIDE_ROADS_FILE,
    include_support_roads: bool = False,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'Side roads choosing.py' first."
        )

    df = pd.read_csv(path)
    _require_columns(df, ["road", "selected"], path.name)

    base = list(BASE_ROADS)
    selected = (
        df.loc[df["selected"] == True, "road"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    required = list(REQUIRED_ROADS)

    selected_roads = normalize_roads(base + required + selected)

    if include_support_roads:
        return normalize_roads(selected_roads + get_required_support_roads(selected_roads))

    return selected_roads
