"""Expected Calibration Error and reliability curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    ece: float
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray


class CalibrationEval:
    """Compute ECE and binned reliability statistics.

    Args:
        y_true:  Ground-truth binary labels (N,).
        y_prob:  Predicted probabilities in [0, 1] (N,).
        n_bins:  Number of equal-width confidence bins.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.n_bins = n_bins

    def compute(self) -> CalibrationResult:
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_indices = np.digitize(self.y_prob, bins[1:-1])
        bin_conf = np.zeros(self.n_bins)
        bin_acc = np.zeros(self.n_bins)
        bin_cnt = np.zeros(self.n_bins, dtype=int)

        for b in range(self.n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_conf[b] = self.y_prob[mask].mean()
                bin_acc[b] = self.y_true[mask].mean()
                bin_cnt[b] = mask.sum()

        n = len(self.y_true)
        ece = float(np.sum(bin_cnt * np.abs(bin_acc - bin_conf)) / n)
        return CalibrationResult(
            ece=ece,
            bin_confidences=bin_conf,
            bin_accuracies=bin_acc,
            bin_counts=bin_cnt,
        )
