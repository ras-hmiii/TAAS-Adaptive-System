"""Reusable experiment skeleton operating on caller-provided, real data."""
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional
from time import perf_counter
from src.monitoring.metrics import classification_metrics

@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    parameters: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics: dict[str, float]
    duration_seconds: float

def run_experiment(config: ExperimentConfig, model: Any, X_train: Any, y_train: Iterable[Any],
                   X_test: Any, y_test: Iterable[Any], fit: Optional[Callable[..., Any]] = None) -> ExperimentResult:
    start = perf_counter()
    if fit is None: model.fit(X_train, y_train)
    else: fit(model, X_train, y_train, config)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return ExperimentResult(config, classification_metrics(y_test, predictions, scores),
                            perf_counter() - start)
