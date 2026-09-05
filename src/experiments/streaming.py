"""Deterministic, model-agnostic streaming adaptation experiments.

The runner deliberately owns no persistence or deployment concerns.  A caller
supplies batches, a model factory, and (optionally) a candidate trainer.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from src.adaptation.workflow import AdaptationDecision, evaluate_adaptation
from src.audit import AuditEvent, AuditLog
from src.drift import ADWINDetector, DriftResult
from src.monitoring import classification_metrics


@dataclass(frozen=True)
class StreamingBatchResult:
    """The observable result for one batch."""

    index: int
    drift: DriftResult
    pre_metrics: dict[str, float]
    post_metrics: dict[str, float]
    decision: AdaptationDecision


@dataclass(frozen=True)
class StreamingExperimentResult:
    """Complete run output, including the audit trail."""

    batches: tuple[StreamingBatchResult, ...]
    audit_events: tuple[AuditEvent, ...]
    model: Any

    @property
    def decisions(self) -> tuple[AdaptationDecision, ...]:
        return tuple(item.decision for item in self.batches)


def _metric(metrics: Mapping[str, float], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise ValueError("score_metric must be present in classification metrics")
    return float(value)


class StreamingAdaptationExperiment:
    """Run a reproducible adaptation loop over feature/label batches.

    ``candidate_trainer`` is called as ``trainer(model, features, labels)``.
    It may also return a model (useful for functional trainers); a trainer that
    returns ``None`` means that the supplied model was fitted in place.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        detector: Any | None = None,
        candidate_trainer: Optional[Callable[..., Any]] = None,
        *,
        min_improvement: float = 0.0,
        score_metric: str = "f1",
        drift_value: Optional[Callable[[dict[str, float], Any, Any], float]] = None,
        audit_log: Optional[AuditLog] = None,
    ):
        self.model_factory = model_factory
        self.detector = ADWINDetector() if detector is None else detector
        self.candidate_trainer = candidate_trainer
        self.min_improvement = min_improvement
        self.score_metric = score_metric
        self.drift_value = drift_value or (
            lambda metrics, _features, _labels: 1.0 - metrics["accuracy"]
        )
        self.audit_log = AuditLog() if audit_log is None else audit_log

    def _train_candidate(self, features: Any, labels: Any) -> Any:
        candidate = self.model_factory()
        if self.candidate_trainer is None:
            candidate.fit(features, labels)
            return candidate
        trained = self.candidate_trainer(candidate, features, labels)
        return candidate if trained is None else trained

    def run(
        self,
        batches: Iterable[tuple[Any, Iterable[Any]]],
        *,
        model: Any | None = None,
    ) -> StreamingExperimentResult:
        """Consume batches once and return metrics, decisions, and audit events."""
        current = self.model_factory() if model is None else model
        results: list[StreamingBatchResult] = []
        for index, batch in enumerate(batches):
            try:
                features, labels = batch
            except (TypeError, ValueError) as exc:
                raise ValueError("each batch must be a (features, labels) pair") from exc
            labels = list(labels)
            self.audit_log.record("batch_received", batch_index=index, size=len(labels))

            predictions = current.predict(features)
            pre = classification_metrics(labels, predictions)
            drift = self.detector.update(float(self.drift_value(pre, features, labels)))
            if drift.detected:
                self.audit_log.record(
                    "drift_detected", batch_index=index, detector=drift.detector,
                    value=drift.value, score=drift.score,
                )
                candidate = self._train_candidate(features, labels)
                post = classification_metrics(labels, candidate.predict(features))
                decision = evaluate_adaptation(
                    True, _metric(pre, self.score_metric),
                    _metric(post, self.score_metric), self.min_improvement,
                )
                if decision.action == "recommend_promotion":
                    current = candidate
            else:
                post, decision = pre, evaluate_adaptation(
                    False, _metric(pre, self.score_metric), None, self.min_improvement
                )
            self.audit_log.record(
                "adaptation_decision", batch_index=index, action=decision.action,
                reason=decision.reason, pre_metrics=dict(pre), post_metrics=dict(post),
            )
            results.append(StreamingBatchResult(index, drift, pre, post, decision))
        return StreamingExperimentResult(tuple(results), tuple(self.audit_log.events), current)


def run_streaming_adaptation(
    batches: Iterable[tuple[Any, Iterable[Any]]],
    model_factory: Callable[[], Any],
    candidate_trainer: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> StreamingExperimentResult:
    """Functional convenience wrapper around :class:`StreamingAdaptationExperiment`."""
    return StreamingAdaptationExperiment(
        model_factory, candidate_trainer=candidate_trainer, **kwargs
    ).run(batches)
