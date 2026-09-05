"""Conservative adaptation decisioning: recommendations never auto-promote."""
from dataclasses import dataclass
from typing import Mapping, Optional

@dataclass(frozen=True)
class AdaptationDecision:
    action: str
    reason: str
    drift_detected: bool
    candidate_score: Optional[float] = None
    baseline_score: Optional[float] = None
    requires_approval: bool = True

def evaluate_adaptation(drift_detected: bool, baseline_score: float,
                         candidate_score: Optional[float] = None,
                         min_improvement: float = 0.0) -> AdaptationDecision:
    if not drift_detected:
        return AdaptationDecision("monitor", "no drift detected", False, candidate_score, baseline_score)
    if candidate_score is None:
        return AdaptationDecision("retrain", "drift detected; candidate evaluation required", True, None, baseline_score)
    if candidate_score >= baseline_score + min_improvement:
        return AdaptationDecision("recommend_promotion", "candidate meets improvement threshold", True,
                                  candidate_score, baseline_score)
    return AdaptationDecision("reject_candidate", "candidate did not improve baseline", True,
                              candidate_score, baseline_score)
