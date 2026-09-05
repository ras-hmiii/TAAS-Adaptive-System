"""Small, framework-independent Elastic Weight Consolidation utilities."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EWCState:
    """Reference parameters and importance weights from a prior task."""

    reference: dict[str, np.ndarray]
    importance: dict[str, np.ndarray]
    strength: float = 1.0

    def penalty(self, parameters: Mapping[str, np.ndarray]) -> float:
        """Return the quadratic penalty for moving away from prior parameters."""
        total = 0.0
        for name, reference in self.reference.items():
            if name not in parameters or name not in self.importance:
                raise KeyError(f"Missing EWC parameter: {name}")
            current = np.asarray(parameters[name])
            importance = np.asarray(self.importance[name])
            if current.shape != reference.shape or importance.shape != reference.shape:
                raise ValueError(f"EWC shapes do not match for parameter: {name}")
            total += float(np.sum(importance * np.square(current - reference)))
        return 0.5 * self.strength * total
