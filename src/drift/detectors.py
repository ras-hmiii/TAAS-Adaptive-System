"""Typed wrappers around River detectors with a dependency-safe fallback."""
from dataclasses import dataclass
from typing import Any, Optional
import math

try:
    from river import drift as _river_drift
except ImportError:
    _river_drift = None

@dataclass(frozen=True)
class DriftEvent:
    detector: str
    index: int
    value: float
    score: Optional[float] = None
    reason: str = "threshold"

@dataclass(frozen=True)
class DriftResult:
    detected: bool
    detector: str
    index: int
    value: float
    score: Optional[float] = None
    event: Optional[DriftEvent] = None

class _Wrapper:
    name = "detector"
    def __init__(self, detector: Any = None):
        self._detector, self.index, self.events = detector, -1, []
    @property
    def drift_detected(self) -> bool:
        return bool(getattr(self._detector, "drift_detected", False))
    def update(self, value: float) -> DriftResult:
        self.index += 1
        value = float(value)
        if self._detector is not None:
            self._detector.update(value)
            detected, score = self.drift_detected, getattr(self._detector, "estimation", None)
        else:
            detected, score = self._fallback(value)
        event = DriftEvent(self.name, self.index, value, score) if detected else None
        if event:
            self.events.append(event)
        return DriftResult(detected, self.name, self.index, value, score, event)
    def _fallback(self, value: float) -> tuple[bool, float]:
        return False, value
    def reset(self) -> None:
        if self._detector is not None and hasattr(self._detector, "reset"):
            self._detector.reset()
        self.index, self.events = -1, []

class ADWINDetector(_Wrapper):
    name = "adwin"
    def __init__(self, delta: float = 0.002, **kwargs: Any):
        super().__init__(_river_drift.ADWIN(delta=delta, **kwargs) if _river_drift else None)
        self.delta, self._window = delta, []
    def _fallback(self, value: float) -> tuple[bool, float]:
        self._window.append(value)
        if len(self._window) < 20: return False, value
        half = len(self._window) // 2
        left, right = self._window[:half], self._window[half:]
        difference = abs(sum(left) / len(left) - sum(right) / len(right))
        threshold = max(0.05, math.sqrt(math.log(2 / self.delta) / (2 * half)))
        if difference > threshold:
            self._window = self._window[half:]
            return True, difference
        if len(self._window) > 200: self._window = self._window[-100:]
        return False, difference

class PageHinkleyDetector(_Wrapper):
    name = "page_hinkley"
    def __init__(self, min_instances: int = 30, delta: float = 0.005,
                 threshold: float = 50.0, alpha: float = 0.9999, **kwargs: Any):
        detector = (_river_drift.PageHinkley(min_instances=min_instances, delta=delta,
                    threshold=threshold, alpha=alpha, **kwargs) if _river_drift else None)
        super().__init__(detector)
        self.min_instances, self.delta, self.threshold, self.alpha = min_instances, delta, threshold, alpha
        self._mean = self._sum = self._minimum = self._cumulative = 0.0
    def _fallback(self, value: float) -> tuple[bool, float]:
        n = self.index + 1
        self._sum += value; self._mean = self._sum / n
        # Cumulative deviations from the running mean approximate Page-Hinkley
        # without making the optional River dependency mandatory.
        self._cumulative += value - self._mean - self.delta
        self._minimum = min(self._minimum, self._cumulative)
        score = self._cumulative - self._minimum
        detected = n >= self.min_instances and score > self.threshold
        if detected: self._sum = self._mean = self._minimum = self._cumulative = 0.0
        return detected, score
