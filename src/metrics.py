"""Evaluation metrics for multi-label classification.

All metrics are computed from real model outputs. Nothing here invents numbers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict:
    """ROC-AUC per label. Skips labels with only one class present in y_true."""
    out = {}
    for i, name in enumerate(labels):
        yt = y_true[:, i]
        if yt.min() == yt.max():  # only positives or only negatives -> AUC undefined
            out[name] = None
            continue
        out[name] = float(roc_auc_score(yt, y_prob[:, i]))
    return out


def mean_auc(auc_dict: dict) -> float:
    vals = [v for v in auc_dict.values() if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def sensitivity_specificity(y_true: np.ndarray, y_prob: np.ndarray,
                            labels: list[str], threshold: float = 0.5) -> dict:
    """Sensitivity (recall) and specificity per label at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    for i, name in enumerate(labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out[name] = {"sensitivity": float(sens), "specificity": float(spec),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    return out
