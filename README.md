# pathology-eval-suite

Evaluation toolkit for computational pathology: slide-level AUC, calibration, per-institution breakdowns, and delta-gap (ΔG) computation across experimental conditions.

## Features

- `SlideEvaluator` — AUC, AUROC, F1 at optimal threshold for slide-level predictions
- `DeltaGap` — in-domain vs cross-domain performance gap (ΔG = AUC_in − AUC_cross)
- `CalibrationEval` — ECE and reliability curves (15-bin histogram)
- `InstitutionBreakdown` — per-TSS subgroup metrics to surface institution-level disparity

## Install

```bash
pip install pathology-eval-suite
```

## Usage

```python
from pathology_eval_suite import SlideEvaluator, DeltaGap

ev = SlideEvaluator(y_true, y_score)
print(ev.auc(), ev.f1())

dg = DeltaGap(in_domain_scores, cross_domain_scores)
print(dg.compute())
```
