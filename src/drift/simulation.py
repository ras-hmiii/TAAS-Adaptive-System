"""Configurable, deterministic drift generation for experiments."""
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

@dataclass(frozen=True)
class DriftSimulationConfig:
    drift_type: str = "mean_shift"
    drift_point: int = 100
    magnitude: float = 1.0
    noise: float = 0.0
    seed: Optional[int] = 42

def simulate_drift(values: Iterable[float], config: DriftSimulationConfig = DriftSimulationConfig()) -> np.ndarray:
    data = np.asarray(list(values), dtype=float).copy()
    if not 0 <= config.drift_point <= len(data):
        raise ValueError("drift_point must be within the input sequence")
    rng = np.random.default_rng(config.seed)
    tail = np.arange(len(data)) >= config.drift_point
    if config.drift_type == "mean_shift": data[tail] += config.magnitude
    elif config.drift_type == "variance_shift":
        center = data[config.drift_point] if config.drift_point < len(data) else 0
        data[tail] = center + (data[tail] - center) * config.magnitude
    elif config.drift_type == "noise": data[tail] += rng.normal(0, config.magnitude, tail.sum())
    else: raise ValueError("drift_type must be mean_shift, variance_shift, or noise")
    if config.noise: data += rng.normal(0, config.noise, len(data))
    return data
