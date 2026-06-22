"""
Baseline tabular model: Random Forest classifier on UCI Adult Income.

Trains the model, evaluates on the held-out test set, and saves the
trained model + label encoders so the explainability module (SHAP) can
load and use them later.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.ingestion.load_tabular import get_splits

MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "tabular_encoders.joblib")


def train_model(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name: str = "test") -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
    }

    print(f"\n--- {split_name.upper()} metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y, preds))

    return metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, encoders = get_splits()

    print("Training Random Forest baseline...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_val, y_val, split_name="validation")
    evaluate_model(model, X_test, y_test, split_name="test")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Encoders saved to {ENCODERS_PATH}")


if __name__ == "__main__":
    main()
