"""pathology-eval-suite: evaluation metrics for computational pathology."""
from .slide_eval import SlideEvaluator
from .delta_gap import DeltaGap
from .calibration import CalibrationEval
from .institution import InstitutionBreakdown

__version__ = "0.1.0"
__all__ = ["SlideEvaluator", "DeltaGap", "CalibrationEval", "InstitutionBreakdown"]
