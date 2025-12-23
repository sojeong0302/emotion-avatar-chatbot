# src/train/metrics.py
from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "f1_macro": 0.0}
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"acc": float(acc), "f1_macro": float(f1m)}
