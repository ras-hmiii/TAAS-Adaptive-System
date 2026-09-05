import numpy as np

from src.experiments import StreamingAdaptationExperiment


class AlwaysDrift:
    def __init__(self):
        self.index = -1

    def update(self, value):
        self.index += 1
        from src.drift import DriftEvent, DriftResult
        event = DriftEvent("test", self.index, value)
        return DriftResult(True, "test", self.index, value, value, event)


class ThresholdModel:
    def __init__(self):
        self.threshold = 0

    def fit(self, features, labels):
        self.threshold = float(np.mean(features))

    def predict(self, features):
        return [int(float(x[0]) >= self.threshold) for x in features]


def test_streaming_run_records_metrics_decision_and_audit_events():
    batches = [
        (np.array([[0], [1]]), [0, 1]),
        (np.array([[10], [11]]), [0, 1]),
    ]
    result = StreamingAdaptationExperiment(
        ThresholdModel, detector=AlwaysDrift(), min_improvement=0
    ).run(batches)

    assert len(result.batches) == 2
    assert result.batches[0].pre_metrics["accuracy"] == 0.5
    assert result.batches[0].decision.action == "recommend_promotion"
    assert any(event.event_type == "drift_detected" for event in result.audit_events)
    assert sum(event.event_type == "adaptation_decision" for event in result.audit_events) == 2


def test_no_drift_does_not_train_or_replace_model():
    class NoDrift:
        def update(self, value):
            from src.drift import DriftResult
            return DriftResult(False, "test", 0, value)

    calls = []

    def factory():
        model = ThresholdModel()
        original = model.fit
        def fit(features, labels):
            calls.append(1)
            original(features, labels)
        model.fit = fit
        return model

    result = StreamingAdaptationExperiment(factory, detector=NoDrift()).run(
        [(np.array([[0], [1]]), [0, 1])]
    )
    assert calls == []
    assert result.batches[0].decision.action == "monitor"
