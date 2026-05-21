"""Slide-level AUC and threshold-optimal F1 evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve


class SlideEvaluator:
    """Compute slide-level classification metrics.

    Args:
        y_true:  Ground-truth binary labels (N,).
        y_score: Predicted probabilities or logits (N,).
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self._threshold: float | None = None

    def auc(self) -> float:
        """Area under the ROC curve."""
        return float(roc_auc_score(self.y_true, self.y_score))

    def optimal_threshold(self) -> float:
        """Youden J statistic threshold (max sensitivity + specificity − 1)."""
        if self._threshold is None:
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score)
            j = tpr - fpr
            self._threshold = float(thresholds[np.argmax(j)])
        return self._threshold

    def f1(self, threshold: float | None = None) -> float:
        """F1 at the given (or Youden-optimal) threshold."""
        thr = threshold if threshold is not None else self.optimal_threshold()
        preds = (self.y_score >= thr).astype(int)
        return float(f1_score(self.y_true, preds))

    def summary(self) -> dict[str, float]:
        return {
            "auc": self.auc(),
            "f1": self.f1(),
            "threshold": self.optimal_threshold(),
            "n": len(self.y_true),
            "prevalence": float(self.y_true.mean()),
        }
