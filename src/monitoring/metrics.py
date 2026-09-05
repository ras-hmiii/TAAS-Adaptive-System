"""Stable classification metrics with graceful handling of one-class samples."""
from typing import Any, Iterable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_metrics(y_true: Iterable[Any], y_pred: Iterable[Any],
                           y_score: Iterable[float] | None = None) -> dict[str, float]:
    true, pred = list(y_true), list(y_pred)
    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
    }
    if y_score is not None:
        try: result["roc_auc"] = float(roc_auc_score(true, list(y_score)))
        except ValueError: result["roc_auc"] = float("nan")
    return result
