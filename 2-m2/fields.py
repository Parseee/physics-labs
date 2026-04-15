from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from config import (
    E_CHARGE,
    E_MASS,
    EPS0,
    GAP_LENGTH,
    GRID_ENABLED,
    GRID_POSITION,
    GRID_TRANSPARENCY,
    N_GRID,
    POISSON_MAX_ITER,
    POISSON_TOL,
    SOR_OMEGA,
    V_ANODE,
    V_CATHODE,
    V_GRID,
)


def make_grid(length: float = GAP_LENGTH, n_grid: int = N_GRID) -> tuple[np.ndarray, float]:
    x = np.linspace(0.0, length, n_grid)
    return x, float(x[1] - x[0])


def build_triode_fixed_nodes(
    x: np.ndarray,
    v_grid: float,
    grid_enabled: bool,
    v_cathode: float,
    v_anode: float,
) -> dict[int, float]:
    nodes: dict[int, float] = {}
    if not grid_enabled:
        return nodes

    base_influence = max(0.0, 1.0 - GRID_TRANSPARENCY)
    influence = float(np.clip(3.0 * base_influence, 0.0, 1.0))
    if influence <= 0.0:
        return nodes

    idx = int(np.argmin(np.abs(x - GRID_POSITION)))
    x_rel = float(x[idx] / x[-1]) if x[-1] != 0 else 0.5
    v_vac = v_cathode + (v_anode - v_cathode) * x_rel
    # Transparent grid should weakly perturb the vacuum profile, not fully clamp it.
    v_eff = (1.0 - influence) * v_vac + influence * v_grid
    nodes[idx] = float(v_eff)
    return nodes


def solve_potential_1d(
    x: np.ndarray,
    v_cathode: float,
    v_anode: float,
    rho: np.ndarray | None = None,
    fixed_nodes: dict[int, float] | None = None,
    max_iter: int = POISSON_MAX_ITER,
    tol: float = POISSON_TOL,
    omega: float = SOR_OMEGA,
) -> np.ndarray:
    # Keep these parameters in signature for compatibility with existing calls.
    _ = (max_iter, omega)

    x = np.asarray(x, dtype=float)
    n = x.size
    rho_arr = np.zeros(n, dtype=float) if rho is None else np.asarray(rho, dtype=float)

    fixed = {} if fixed_nodes is None else dict(fixed_nodes)
    fixed[0] = v_cathode
    fixed[n - 1] = v_anode
    anchors = sorted(fixed.items(), key=lambda kv: kv[0])

    def rhs(xi: float, y: np.ndarray) -> np.ndarray:
        rho_i = float(np.interp(xi, x, rho_arr))
        # y = [phi, dphi/dx], with Poisson equation d2phi/dx2 = -rho/eps0
        return np.array([y[1], -rho_i / EPS0], dtype=float)

    def solve_segment(i0: int, v0: float, i1: int, v1: float) -> np.ndarray:
        x0 = float(x[i0])
        x1 = float(x[i1])
        x_seg = x[i0:i1 + 1]
        if x1 == x0:
            return np.array([v0], dtype=float)

        guess = (v1 - v0) / (x1 - x0)

        def mismatch(slope: float) -> float:
            sol = solve_ivp(
                rhs,
                (x0, x1),
                np.array([v0, slope], dtype=float),
                t_eval=np.array([x1], dtype=float),
                rtol=tol,
                atol=max(tol * 0.1, 1e-12),
            )
            return float(sol.y[0, -1] - v1)

        width = max(abs(guess), 1.0)
        f_lo = mismatch(guess - width)
        f_hi = mismatch(guess + width)
        for _ in range(50):
            if f_lo * f_hi <= 0.0:
                break
            width *= 2.0
            f_lo = mismatch(guess - width)
            f_hi = mismatch(guess + width)
        else:
            raise RuntimeError("Failed to bracket shooting slope for solve_potential_1d.")

        root = root_scalar(
            mismatch,
            bracket=(guess - width, guess + width),
            method="brentq",
            xtol=tol,
        )
        if not root.converged:
            raise RuntimeError("Shooting method did not converge for solve_potential_1d.")

        sol = solve_ivp(
            rhs,
            (x0, x1),
            np.array([v0, float(root.root)], dtype=float),
            t_eval=x_seg,
            rtol=tol,
            atol=max(tol * 0.1, 1e-12),
        )
        return sol.y[0]

    phi_parts: list[np.ndarray] = []
    for seg_idx in range(len(anchors) - 1):
        i0, v0 = anchors[seg_idx]
        i1, v1 = anchors[seg_idx + 1]
        segment = solve_segment(i0, float(v0), i1, float(v1))
        if seg_idx > 0:
            segment = segment[1:]
        phi_parts.append(segment)

    return np.concatenate(phi_parts)


def electric_field_from_potential(phi: np.ndarray, dx: float) -> np.ndarray:
    return -np.gradient(phi, dx)


def vacuum_field_solution(
    v_anode: float = V_ANODE,
    v_cathode: float = V_CATHODE,
    v_grid: float = V_GRID,
    grid_enabled: bool = GRID_ENABLED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, dx = make_grid()
    fixed = build_triode_fixed_nodes(
        x,
        v_grid=v_grid,
        grid_enabled=grid_enabled,
        v_cathode=v_cathode,
        v_anode=v_anode,
    )
    phi = solve_potential_1d(
        x=x,
        v_cathode=v_cathode,
        v_anode=v_anode,
        rho=None,
        fixed_nodes=fixed,
    )
    e = electric_field_from_potential(phi, dx)
    return x, phi, e


def child_langmuir_current(v_anode: float, area: float, gap: float) -> float:
    if v_anode <= 0:
        return 0.0
    pref = (4.0 / 9.0) * EPS0 * np.sqrt(2.0 * E_CHARGE / E_MASS)
    j = pref * (v_anode ** 1.5) / (gap**2)
    return float(j * area)
