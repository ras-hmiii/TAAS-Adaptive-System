"""
Tests for the tabular pipeline: ingestion -> model training -> SHAP explainability.

Uses synthetic data shaped like the UCI Adult Income schema so tests run
fast and don't depend on network access or the real dataset being present.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.ingestion.load_tabular import preprocess
from src.models.train_tabular import train_model, evaluate_model
from src.explainability.shap_tabular import TabularExplainer


@pytest.fixture
def synthetic_adult_df():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "workclass": np.random.choice(["Private", "Self-emp", "Govt"], n),
        "fnlwgt": np.random.randint(10000, 500000, n),
        "education": np.random.choice(["Bachelors", "HS-grad", "Masters"], n),
        "education_num": np.random.randint(1, 16, n),
        "marital_status": np.random.choice(["Married", "Single"], n),
        "occupation": np.random.choice(["Tech", "Sales", "Exec"], n),
        "relationship": np.random.choice(["Husband", "Wife", "Not-in-family"], n),
        "race": np.random.choice(["White", "Black", "Asian"], n),
        "sex": np.random.choice(["Male", "Female"], n),
        "capital_gain": np.random.randint(0, 5000, n),
        "capital_loss": np.random.randint(0, 1000, n),
        "hours_per_week": np.random.randint(20, 60, n),
        "native_country": np.random.choice(["United-States", "Mexico"], n),
        "income": np.random.choice(["<=50K", ">50K"], n, p=[0.6, 0.4]),
    })


@pytest.fixture
def trained_model_and_data(synthetic_adult_df):
    X, y, encoders = preprocess(synthetic_adult_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


class TestIngestion:
    def test_preprocess_returns_numeric_features(self, synthetic_adult_df):
        X, y, encoders = preprocess(synthetic_adult_df)
        assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)

    def test_preprocess_no_missing_values(self, synthetic_adult_df):
        X, y, encoders = preprocess(synthetic_adult_df)
        assert X.isnull().sum().sum() == 0

    def test_preprocess_binary_target(self, synthetic_adult_df):
        X, y, encoders = preprocess(synthetic_adult_df)
        assert set(y.unique()).issubset({0, 1})

    def test_preprocess_encoders_created_for_categoricals(self, synthetic_adult_df):
        X, y, encoders = preprocess(synthetic_adult_df)
        expected = {"workclass", "education", "marital_status", "occupation",
                    "relationship", "race", "sex", "native_country"}
        assert expected.issubset(set(encoders.keys()))


class TestModelTraining:
    def test_train_model_returns_fitted_model(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_predictions_are_binary(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        preds = model.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_sum_to_one(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        probs = model.predict_proba(X_test)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_model_returns_expected_metrics(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test, split_name="test")
        assert set(metrics.keys()) == {"accuracy", "f1", "roc_auc"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestSHAPExplainability:
    def test_explain_instance_returns_correct_length(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        explainer = TabularExplainer(model, feature_names=X_train.columns.tolist())
        result = explainer.explain_instance(X_test.iloc[0])
        assert len(result["shap_values"]) == X_train.shape[1]

    def test_shap_values_reconstruct_prediction(self, trained_model_and_data):
        """SHAP's core guarantee: sum(shap_values) + base_value == prediction."""
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        explainer = TabularExplainer(model, feature_names=X_train.columns.tolist())
        result = explainer.explain_instance(X_test.iloc[0])
        reconstructed = sum(result["shap_values"]) + result["base_value"]
        assert abs(reconstructed - result["prediction"]) < 1e-4

    def test_explain_batch_shape(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        explainer = TabularExplainer(model, feature_names=X_train.columns.tolist())
        batch_values = explainer.explain_batch(X_test)
        assert batch_values.shape == (len(X_test), X_train.shape[1])

    def test_fidelity_score_in_valid_range(self, trained_model_and_data):
        model, X_train, X_test, y_train, y_test = trained_model_and_data
        explainer = TabularExplainer(model, feature_names=X_train.columns.tolist())
        score = explainer.fidelity_score(X_test)
        assert 0.0 <= score <= 1.0
