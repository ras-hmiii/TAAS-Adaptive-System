"""JSON persistence for reproducible experiment records."""

import json
from pathlib import Path
from typing import Any

from .runner import ExperimentResult


def save_experiment(result: ExperimentResult, directory: str | Path = "results") -> Path:
    """Write one experiment result and return its artifact path."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{result.config.name}.json"
    payload: dict[str, Any] = {
        "experiment_id": result.config.name,
        "seed": result.config.seed,
        "parameters": result.config.parameters,
        "metrics": result.metrics,
        "duration_seconds": result.duration_seconds,
        "status": "completed",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    return path


def load_experiments(directory: str | Path = "results") -> list[dict[str, Any]]:
    target = Path(directory)
    if not target.is_dir():
        return []
    records = []
    for path in sorted(target.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if isinstance(record, dict):
            records.append(record)
    return records
