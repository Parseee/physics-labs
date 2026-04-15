from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PARAMS_PATH = Path(__file__).with_name("params.json")


def _load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_P = _load_params()

# Constants
EPS0 = float(_P["constants"]["eps0"])
E_CHARGE = float(_P["constants"]["e_charge"])
E_MASS = float(_P["constants"]["e_mass"])
KB = float(_P["constants"]["kb"])

# Emission
RICHARDSON_A = float(_P["emission"]["richardson_a"])
WORK_FUNCTION_EV = float(_P["emission"]["work_function_ev"])
CATHODE_TEMPERATURE_K = float(_P["emission"]["cathode_temperature_k"])
CATHODE_AREA = float(_P["emission"]["cathode_area"])

# Geometry
GAP_LENGTH = float(_P["geometry"]["gap_length"])
GRID_ENABLED = bool(_P["geometry"]["grid_enabled"])
GRID_POSITION = float(_P["geometry"]["grid_position"])
GRID_TRANSPARENCY = float(_P["geometry"]["grid_transparency"])

# Electrodes
V_CATHODE = float(_P["electrodes"]["v_cathode"])
V_ANODE = float(_P["electrodes"]["v_anode"])
V_GRID = float(_P["electrodes"]["v_grid"])

# Numerics
N_GRID = int(_P["numerics"]["n_grid"])
POISSON_MAX_ITER = int(_P["numerics"]["poisson_max_iter"])
POISSON_TOL = float(_P["numerics"]["poisson_tol"])
SOR_OMEGA = float(_P["numerics"]["sor_omega"])

# Particle simulation
DT = float(_P["particles"]["dt"])
T_END = float(_P["particles"]["t_end"])
N_TEST_PARTICLES = int(_P["particles"]["n_test_particles"])
N_TRAJECTORIES_TO_STORE = int(_P["particles"]["n_trajectories_to_store"])
INJECT_MACRO_PER_STEP = int(_P["particles"]["inject_macro_per_step"])
THERMAL_SIGMA_FACTOR = float(_P["particles"]["thermal_sigma_factor"])

# Sweep
VA_SWEEP = np.asarray(_P["sweep"]["va_sweep"], dtype=float)
RUN_SPACE_CHARGE = bool(_P["sweep"]["run_space_charge"])
RUN_TRIODE_SWEEP = bool(_P["sweep"]["run_triode_sweep"])
VG_SWEEP = np.asarray(_P["sweep"]["vg_sweep"], dtype=float)

# Animation
ANIMATION_ENABLED = bool(_P["animation"]["enabled"])
ANIMATION_MAX_PARTICLES = int(_P["animation"]["max_particles"])
ANIMATION_TRAIL_POINTS = int(_P["animation"]["trail_points"])
ANIMATION_INTERVAL_MS = int(_P["animation"]["interval_ms"])


def reload_params() -> None:
    global _P
    _P = _load_params()


def emission_current(temp_k: float = CATHODE_TEMPERATURE_K) -> float:
    work_joule = WORK_FUNCTION_EV * E_CHARGE
    j = RICHARDSON_A * temp_k**2 * math.exp(-work_joule / (KB * temp_k))
    return j * CATHODE_AREA
