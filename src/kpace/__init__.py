"""K-PACE: Kinematically-Constrained Pitch-control Attribution for Counterfactual Error."""

from .config import (
    A_MAX,
    DELTA_T,
    GUILTY_THRESHOLD,
    LAMBDA,
    PITCH_LENGTH,
    PITCH_WIDTH,
    T_REACT,
    V_MAX,
)
from .kinematic_pitch_optimizer import KinematicPitchOptimizer
from .load_metrica import MetricaDataLoader

__version__ = "1.0.0"

__all__ = [
    "KinematicPitchOptimizer",
    "MetricaDataLoader",
    "T_REACT",
    "A_MAX",
    "V_MAX",
    "LAMBDA",
    "GUILTY_THRESHOLD",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "DELTA_T",
]
