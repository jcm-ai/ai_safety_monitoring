from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure arrays are 2D: (n_samples, n_labels).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _clip_probs(y_prob: np.ndarray) -> np.ndarray:
    """
    Clip probabilities to [0, 1] to avoid numerical issues.
    """
    return np.clip(np.asarray(y_prob), 0.0, 1.0)

def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute multilabel metrics at a fixed decision threshold.

    Args:
        y_true: Binary ground truth matrix of shape (n_samples, n_labels).
        y_prob: Predicted probabilities matrix of shape (n_samples, n_labels).
        threshold: Decision threshold for binarizing probabilities.
        labels: Optional label names for per-label breakdown.

    Returns:
        Dict with macro/micro precision, recall, f1, ROC-AUC, and per-label metrics.
    """
    Y_true = _ensure_2d(y_true).astype(int)
    Y_prob = _clip_probs(_ensure_2d(y_prob))
    Y_pred = (Y_prob >= float(threshold)).astype(int)

    # Macro and micro metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )

    # Per-label metrics
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0
    )

    # ROC-AUC (may fail if a column has only one class present)
    roc_auc_macro = None
    roc_auc_micro = None
    roc_auc_per: List[Optional[float]] = [None] * Y_true.shape[1]
    try:
        roc_auc_macro = float(roc_auc_score(Y_true, Y_prob, average="macro"))
    except Exception:
        pass
    try:
        roc_auc_micro = float(roc_auc_score(Y_true, Y_prob, average="micro"))
    except Exception:
        pass
    # Per label
    for i in range(Y_true.shape[1]):
        yt = Y_true[:, i]
        yp = Y_prob[:, i]
        # Need both classes present
        if np.unique(yt).size == 2:
            try:
                roc_auc_per[i] = float(roc_auc_score(yt, yp))
            except Exception:
                roc_auc_per[i] = None

    # Assemble per-label dict
    if labels is None:
        labels = [f"label_{i}" for i in range(Y_true.shape[1])]
    per_label: Dict[str, Dict[str, Optional[float]]] = {}
    for i, name in enumerate(labels):
        per_label[name] = {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
            "roc_auc": roc_auc_per[i] if roc_auc_per[i] is not None else None,
        }

    # Top-level summary (include "f1" for compatibility with tuning scripts)
    summary = {
        "threshold": float(threshold),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_micro": roc_auc_micro,
        "per_label": per_label,
        # Convenience key expected by tuning code (uses f1_macro)
        "f1": float(f1_macro),
    }
    return summary

def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute binary classification metrics at a fixed threshold.

    Args:
        y_true: Binary ground truth array of shape (n_samples,).
        y_prob: Predicted probabilities array of shape (n_samples,).

    Returns:
        Dict with precision, recall, f1, accuracy, ROC-AUC, confusion matrix, and threshold.
    """
    yt = np.asarray(y_true).astype(int).reshape(-1)
    yp_prob = _clip_probs(np.asarray(y_prob).reshape(-1))
    yp = (yp_prob >= float(threshold)).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    acc = accuracy_score(yt, yp)

    roc_auc = None
    # ROC-AUC requires both classes present
    if np.unique(yt).size == 2:
        try:
            roc_auc = float(roc_auc_score(yt, yp_prob))
        except Exception:
            roc_auc = None

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def sweep_thresholds_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Sweep thresholds for a binary task and return the best threshold by F1.

    Args:
        y_true: Binary ground truth (n_samples,).
        y_prob: Predicted probabilities (n_samples,).
        thresholds: Optional list of thresholds to evaluate.

    Returns:
        (best_threshold, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = [round(t, 3) for t in np.linspace(0.1, 0.9, 33)]
    best_t = thresholds[0]
    best = {"f1": -1.0}
    for t in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=t)
        if m["f1"] > best["f1"]:
            best, best_t = m, t
    return best_t, best
