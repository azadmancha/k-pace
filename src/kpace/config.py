"""Model constants and kinematic parameters for K-PACE."""

# Biomechanical sprint kinematics
T_REACT: float = 0.7
LAMBDA: float = 0.14
V_MAX: float = 8.0
A_MAX: float = 4.5

# Pitch geometry and discretization
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0
GRID_CELLS_X: int = 105
GRID_CELLS_Y: int = 68

# Counterfactual decision horizon
DELTA_T: float = 1.2

# Diagnostic triage threshold (% threat reduction)
GUILTY_THRESHOLD: float = 10.0

# Goal danger zone geometry
DANGER_X_MIN: float = 80.0
DANGER_Y_MIN: float = 24.0
DANGER_Y_MAX: float = 44.0
DANGER_FLOOR_M: float = 3.0
HEURISTIC_SCALE: float = 200.0
HEURISTIC_DECAY: float = 1.8

# Parametric threat kernel weights
PARAMETRIC_SCALE: float = 50.0
PARAMETRIC_CENTRAL_DIST: float = 15.0
PARAMETRIC_CENTRAL_ANGLE: float = 0.5
PARAMETRIC_WIDE_X: float = 100.0
PARAMETRIC_WIDE_SPREAD: float = 10.0
PARAMETRIC_WIDE_OFFSET: float = 20.0
PARAMETRIC_WIDE_WEIGHT: float = 0.3
PARAMETRIC_HALF_X: float = 85.0
PARAMETRIC_HALF_SPREAD: float = 15.0
PARAMETRIC_HALF_OFFSET: float = 15.0
PARAMETRIC_HALF_WIDTH: float = 8.0
PARAMETRIC_HALF_WEIGHT: float = 0.4

# Dynamic threat kernel scaling
MA2024_SCALE: float = 150.0
DYNAMIC_ANGLE_WIDTH: float = 0.75
DYNAMIC_BASE_MIX: float = 0.5
