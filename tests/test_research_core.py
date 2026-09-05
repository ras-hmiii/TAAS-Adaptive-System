import numpy as np
from sklearn.linear_model import LogisticRegression

from src.drift import ADWINDetector, DriftSimulationConfig, simulate_drift
from src.monitoring import classification_metrics
from src.adaptation import EWCState, ModelMetadata, ModelRegistry, evaluate_adaptation
from src.experiments import ExperimentConfig, load_experiments, run_experiment, save_experiment

def test_simulation_is_deterministic_and_shifts_tail():
    out = simulate_drift(np.zeros(10), DriftSimulationConfig(drift_point=5, magnitude=2))
    assert np.all(out[:5] == 0)
    assert np.all(out[5:] == 2)

def test_detector_returns_structured_result():
    result = ADWINDetector().update(1)
    assert result.detector == "adwin"
    assert result.index == 0
    assert result.event is None

def test_metrics_and_decision_do_not_auto_promote():
    metrics = classification_metrics([0, 1, 1], [0, 1, 0])
    assert metrics["accuracy"] == 2 / 3
    decision = evaluate_adaptation(True, .7, .8)
    assert decision.action == "recommend_promotion"
    assert decision.requires_approval

def test_registry_and_experiment_runner():
    registry = ModelRegistry()
    registry.register(ModelMetadata("v1", "logistic"))
    assert registry.get("v1").status == "candidate"
    x = np.array([[0], [1], [2], [3]])
    result = run_experiment(ExperimentConfig(), LogisticRegression(), x, [0, 0, 1, 1], x, [0, 0, 1, 1])
    assert result.metrics["accuracy"] == 1.0


def test_ewc_penalty_is_zero_at_reference_and_positive_after_update():
    state = EWCState(
        reference={"weight": np.array([1.0, 2.0])},
        importance={"weight": np.array([2.0, 1.0])},
        strength=2.0,
    )
    assert state.penalty({"weight": np.array([1.0, 2.0])}) == 0.0
    assert state.penalty({"weight": np.array([2.0, 2.0])}) == 2.0


def test_experiment_result_can_be_persisted(tmp_path):
    result = run_experiment(
        ExperimentConfig(name="persisted"),
        LogisticRegression(),
        np.array([[0], [1], [2], [3]]),
        [0, 0, 1, 1],
        np.array([[0], [1], [2], [3]]),
        [0, 0, 1, 1],
    )
    path = save_experiment(result, tmp_path)
    records = load_experiments(tmp_path)
    assert path.name == "persisted.json"
    assert records[0]["metrics"]["accuracy"] == 1.0
