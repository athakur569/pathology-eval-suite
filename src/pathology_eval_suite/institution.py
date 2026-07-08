"""Per-institution AUC breakdown and fairness gap reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class InstitutionBreakdownResult:
    per_institution: dict[str, float] = field(default_factory=dict)
    fairness_gap: float = 0.0
    macro_mean: float = 0.0
    macro_std: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"institution": k, "auc": v} for k, v in self.per_institution.items()]
        return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


class InstitutionBreakdown:
    """Compute per-institution AUC and the fairness gap (max − min AUC).

    Args:
        y_true:       Binary labels (N,).
        y_score:      Predicted scores (N,).
        institution:  Institution identifier for each sample (N,).
        min_samples:  Skip institutions with fewer samples than this.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        institution: np.ndarray,
        min_samples: int = 20,
    ) -> None:
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self.institution = np.asarray(institution)
        self.min_samples = min_samples

    def compute(self) -> InstitutionBreakdownResult:
        per_inst: dict[str, float] = {}
        for inst in np.unique(self.institution):
            mask = self.institution == inst
            if mask.sum() < self.min_samples:
                continue
            yt, ys = self.y_true[mask], self.y_score[mask]
            if len(np.unique(yt)) < 2:
                continue
            per_inst[str(inst)] = float(roc_auc_score(yt, ys))

        aucs = list(per_inst.values())
        gap = (max(aucs) - min(aucs)) if len(aucs) >= 2 else 0.0
        return InstitutionBreakdownResult(
            per_institution=per_inst,
            fairness_gap=gap,
            macro_mean=float(np.mean(aucs)) if aucs else 0.0,
            macro_std=float(np.std(aucs)) if aucs else 0.0,
        )
