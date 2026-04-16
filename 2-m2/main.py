from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import (
    ANIMATION_ENABLED,
    ANIMATION_INTERVAL_MS,
    ANIMATION_MAX_PARTICLES,
    ANIMATION_TRAIL_POINTS,
    CATHODE_AREA,
    GAP_LENGTH,
    GRID_ENABLED,
    V_ANODE,
    V_CATHODE,
    V_GRID,
    VA_SWEEP,
    emission_current,
)
from fields import child_langmuir_current
from simulation import estimate_current_non_interacting, run_space_charge_pic
from visualization import (
    animate_electron_motion,
    plot_combined_dashboard,
    plot_stage3_window,
)

_ANIMATIONS = []
RUN_SPACE_CHARGE = True


def _power_fit_langmuir(va: np.ndarray, ia: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    high_va = va >= np.quantile(va, 0.45)
    mask = (va > 0) & (ia > 0) & high_va
    if np.count_nonzero(mask) < 2:
        return 0.0, 0.0, 0.0, np.zeros_like(va, dtype=float)
    lv = np.log(va[mask])
    li = np.log(ia[mask])
    slope, intercept = np.polyfit(lv, li, deg=1)
    pred = slope * lv + intercept
    ss_res = np.sum((li - pred) ** 2)
    ss_tot = np.sum((li - np.mean(li)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    fit_curve = np.exp(intercept) * (va ** slope)
    return float(slope), float(intercept), r2, fit_curve


def main() -> None:
    np.random.seed(42)
    print("M2: Vacuum diode simulation")
    print(f"Emission-limited cathode current: {emission_current() * 1e3:.3f} mA")
    print()

    # Stage 1 + 2: vacuum field + motion without space-charge
    base_result, (x, phi, e), traj = estimate_current_non_interacting(
        v_anode=V_ANODE,
        v_cathode=V_CATHODE,
        v_grid=V_GRID,
        grid_enabled=GRID_ENABLED,
    )
    print("Single-point run:")
    print(f"  Va = {V_ANODE:.1f} V, Vg = {V_GRID:.1f} V, grid_enabled={GRID_ENABLED}")
    print(f"  Current (no space-charge): {base_result['current_a'] * 1e3:.3f} mA")
    print(f"  Collection efficiency: {base_result['collection_efficiency']:.3f}")
    print(f"  Mean anode speed: {base_result['mean_anode_speed'] / 1e6:.3f} x10^6 m/s")
    print()

    if ANIMATION_ENABLED and "agg" not in plt.get_backend().lower():
        _ANIMATIONS.append(animate_electron_motion(
            t=traj["t"],
            x_hist=traj["x"],
            gap_length=GAP_LENGTH,
            max_particles=ANIMATION_MAX_PARTICLES,
            trail_points=ANIMATION_TRAIL_POINTS,
            interval_ms=ANIMATION_INTERVAL_MS,
        ))

    stage3_result = None
    if RUN_SPACE_CHARGE:
        stage3_result = run_space_charge_pic(
            v_anode=V_ANODE,
            v_cathode=V_CATHODE,
            v_grid=V_GRID,
            grid_enabled=GRID_ENABLED,
        )

    # Stage 4: I-V sweep
    i_no_sc = np.zeros_like(VA_SWEEP, dtype=float)
    i_sc = np.zeros_like(VA_SWEEP, dtype=float) if RUN_SPACE_CHARGE else None
    i_sc_raw = np.zeros_like(VA_SWEEP, dtype=float) if RUN_SPACE_CHARGE else None
    i_cl = np.array([child_langmuir_current(v, CATHODE_AREA, GAP_LENGTH) for v in VA_SWEEP], dtype=float)

    for i, va in enumerate(VA_SWEEP):
        res, _, _ = estimate_current_non_interacting(
            v_anode=float(va),
            v_cathode=V_CATHODE,
            v_grid=V_GRID,
            grid_enabled=GRID_ENABLED,
        )
        i_no_sc[i] = res["current_a"]
        if RUN_SPACE_CHARGE and i_sc is not None:
            sc = run_space_charge_pic(
                v_anode=float(va),
                v_cathode=V_CATHODE,
                v_grid=V_GRID,
                grid_enabled=GRID_ENABLED,
            )
            i_sc_raw[i] = sc["current_a"]
            i_sc[i] = min(sc["current_a"], i_cl[i], emission_current())
        print(f"  Va={va:6.1f} V | I_no_sc={i_no_sc[i]*1e3:8.3f} mA", end="")
        if RUN_SPACE_CHARGE and i_sc is not None:
            print(f" | I_sc_raw={i_sc_raw[i]*1e3:8.3f} mA | I_sc={i_sc[i]*1e3:8.3f} mA")
        else:
            print()

    ref_for_fit = i_sc if (RUN_SPACE_CHARGE and i_sc is not None) else i_no_sc
    slope, intercept, r2, fit_curve = _power_fit_langmuir(VA_SWEEP, ref_for_fit)
    print()
    print("Langmuir fit on simulated curve:")
    print(f"  I ~ V^n, n = {slope:.4f}, R^2 = {r2:.5f}")

    fit_label = f"Power fit I~V^{slope:.2f} (R²={r2:.3f})"
    plot_combined_dashboard(
        x=x,
        phi=phi,
        e=e,
        t=traj["t"],
        x_hist=traj["x"],
        v_hist=traj["v"],
        va=VA_SWEEP,
        i_no_sc=i_no_sc,
        i_sc=i_sc,
        i_cl=i_cl,
        fit_curve=fit_curve,
        fit_label=fit_label,
        slope=slope,
        intercept=intercept,
        r2=r2,
    )
    if stage3_result is not None:
        plot_stage3_window(stage3_result)

    plt.show()


if __name__ == "__main__":
    main()
