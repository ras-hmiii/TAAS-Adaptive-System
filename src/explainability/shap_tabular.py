"""
SHAP-based explainability for the tabular Random Forest model.

Implements FR01: generate a SHAP explanation vector for every tabular
model prediction, using TreeExplainer (fast, exact for tree ensembles).
"""

import shap
import numpy as np
import pandas as pd


class TabularExplainer:
    """Wraps a trained tree-based model with SHAP TreeExplainer."""

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def _normalize_output(self, shap_output, expected_value):
        """
        SHAP's TreeExplainer return shape varies by library version and
        model type. Observed formats for a binary-classification
        RandomForest:
          - list of 2 arrays: [class0_values (n,f), class1_values (n,f)]
          - 2D array (n, f): values for the positive class directly
          - 3D array (n, f, 2): values for both classes, last axis = class

        This normalizes all three to a single (n, f) array for class 1,
        plus a scalar base_value for class 1.
        """
        if isinstance(shap_output, list):
            values = np.asarray(shap_output[1])
        else:
            arr = np.asarray(shap_output)
            if arr.ndim == 3:
                values = arr[:, :, 1]
            else:
                values = arr

        base_value = np.atleast_1d(expected_value).astype(float)
        base_value = base_value[-1]

        return values, float(base_value)

    def explain_instance(self, x_row: pd.Series | pd.DataFrame) -> dict:
        """
        Generate a SHAP explanation for a single prediction.

        Returns a dict with:
            - shap_values: per-feature attribution (list[float])
            - base_value: model's expected output before attributions
            - prediction: model's predicted probability for class 1
            - feature_values: the actual input values, for display
        """
        if isinstance(x_row, pd.Series):
            x_row = x_row.to_frame().T

        shap_output = self.explainer.shap_values(x_row)
        values, base_value = self._normalize_output(shap_output, self.explainer.expected_value)

        prediction = self.model.predict_proba(x_row)[0, 1]

        return {
            "shap_values": values[0].tolist(),
            "base_value": base_value,
            "prediction": float(prediction),
            "feature_names": self.feature_names,
            "feature_values": x_row.iloc[0].to_dict(),
        }

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Generate SHAP values for a batch — used for fidelity evaluation."""
        shap_output = self.explainer.shap_values(X)
        values, _ = self._normalize_output(shap_output, self.explainer.expected_value)
        return values

    def fidelity_score(self, X: pd.DataFrame) -> float:
        """
        Simple fidelity check (FR-relevant, used in Expected Results §10):
        ratio of (sum of SHAP values + base value) vs actual model output,
        averaged across samples. SHAP values should sum to (output - base).
        Returns mean absolute reconstruction error as a fidelity proxy —
        lower is better; subtract from 1 for a 0-1 "fidelity score".
        """
        shap_output = self.explainer.shap_values(X)
        values, base_value = self._normalize_output(shap_output, self.explainer.expected_value)

        reconstructed = values.sum(axis=1) + base_value
        actual = self.model.predict_proba(X)[:, 1]

        mean_abs_error = np.mean(np.abs(reconstructed - actual))
        return max(0.0, 1.0 - mean_abs_error)


if __name__ == "__main__":
    import joblib
    from src.models.train_tabular import MODEL_PATH, ENCODERS_PATH

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    print("Loaded model and encoders. TabularExplainer ready to use.")
    print("Example usage:")
    print("  explainer = TabularExplainer(model, feature_names=X.columns.tolist())")
    print("  result = explainer.explain_instance(X.iloc[0])")
