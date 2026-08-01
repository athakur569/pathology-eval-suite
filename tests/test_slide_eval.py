"""Tests for SlideEvaluator and DeltaGap."""

import numpy as np
import pytest

from pathology_eval_suite import SlideEvaluator, DeltaGap


def _perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    return y, s


def test_auc_perfect():
    y, s = _perfect()
    ev = SlideEvaluator(y, s)
    assert ev.auc() == pytest.approx(1.0)


def test_f1_above_zero():
    y, s = _perfect()
    ev = SlideEvaluator(y, s)
    assert ev.f1() > 0.0


def test_summary_keys():
    y, s = _perfect()
    result = SlideEvaluator(y, s).summary()
    assert set(result.keys()) == {"auc", "f1", "threshold", "n", "prevalence"}


def test_delta_gap_positive():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1])
    s_in = np.array([0.1, 0.2, 0.8, 0.9])
    s_cross = np.array([0.3, 0.4, 0.6, 0.55])
    dg = DeltaGap(y, s_in, y, s_cross)
    result = dg.compute()
    assert result.delta_gap > 0
    assert result.auc_in > result.auc_cross
