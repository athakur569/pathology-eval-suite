"""Delta-gap (ΔG) computation between in-domain and cross-domain evaluation sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class DeltaGapResult:
    auc_in: float
    auc_cross: float
    delta_gap: float

    def __str__(self) -> str:
        return (
            f"ΔG = {self.delta_gap:+.4f}  "
            f"(in={self.auc_in:.4f}, cross={self.auc_cross:.4f})"
        )


class DeltaGap:
    """Compute ΔG = AUC_in_domain − AUC_cross_domain.

    A positive ΔG means the model degrades when applied outside its training
    distribution — the core signal this framework monitors.

    Args:
        y_true_in:    Labels for in-domain test set.
        y_score_in:   Scores for in-domain test set.
        y_true_cross: Labels for held-out cross-domain test set.
        y_score_cross: Scores for held-out cross-domain test set.
    """

    def __init__(
        self,
        y_true_in: np.ndarray,
        y_score_in: np.ndarray,
        y_true_cross: np.ndarray,
        y_score_cross: np.ndarray,
    ) -> None:
        self.y_true_in = np.asarray(y_true_in)
        self.y_score_in = np.asarray(y_score_in)
        self.y_true_cross = np.asarray(y_true_cross)
        self.y_score_cross = np.asarray(y_score_cross)

    def compute(self) -> DeltaGapResult:
        auc_in = float(roc_auc_score(self.y_true_in, self.y_score_in))
        auc_cross = float(roc_auc_score(self.y_true_cross, self.y_score_cross))
        return DeltaGapResult(
            auc_in=auc_in,
            auc_cross=auc_cross,
            delta_gap=auc_in - auc_cross,
        )
