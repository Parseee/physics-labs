from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_potential_and_field(x: np.ndarray, phi: np.ndarray, e: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(x * 1e3, phi, color="tab:blue", lw=2)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("Potential φ [V]")
    ax1.set_title("Electrostatic potential")
    ax1.grid(alpha=0.3)

    ax2.plot(x * 1e3, e / 1e3, color="tab:red", lw=2)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("Electric field E [kV/m]")
    ax2.set_title("Electric field")
    ax2.grid(alpha=0.3)
    fig.tight_layout()


def plot_trajectories(t: np.ndarray, x_hist: np.ndarray, v_hist: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t_ns = t * 1e9
    for i in range(x_hist.shape[0]):
        if not np.isfinite(x_hist[i]).any():
            continue
        ax1.plot(t_ns, x_hist[i] * 1e3, lw=0.8, alpha=0.8)
        ax2.plot(t_ns, v_hist[i] / 1e6, lw=0.8, alpha=0.8)
    ax1.set_ylabel("x [mm]")
    ax1.set_title("Electron trajectories (subset)")
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("v [10^6 m/s]")
    ax2.set_xlabel("t [ns]")
    ax2.grid(alpha=0.3)
    fig.tight_layout()


def plot_iv_curves(
    va: np.ndarray,
    i_no_sc: np.ndarray,
    i_sc: np.ndarray | None,
    i_cl: np.ndarray,
    fit_curve: np.ndarray,
    fit_label: str,
) -> None:
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(va, i_no_sc * 1e3, "o-", label="No space-charge")
    if i_sc is not None:
        plt.plot(va, i_sc * 1e3, "s-", label="With space-charge")
    plt.plot(va, i_cl * 1e3, "--", label="Child-Langmuir theory")
    plt.plot(va, fit_curve * 1e3, ":", lw=2.0, label=fit_label)
    plt.xlabel("Anode voltage Va [V]")
    plt.ylabel("Anode current Ia [mA]")
    plt.title("Vacuum diode I-V characteristics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_langmuir_check(va: np.ndarray, i_ref: np.ndarray, slope: float, intercept: float, r2: float) -> None:
    mask = (va > 0) & (i_ref > 0) & (va >= np.quantile(va, 0.45))
    lv = np.log(va[mask])
    li = np.log(i_ref[mask])
    fit = slope * lv + intercept
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(lv, li, "o", label="Simulation data")
    plt.plot(lv, fit, "-", label=f"fit slope={slope:.3f}, R²={r2:.4f}")
    plt.xlabel("ln(Va)")
    plt.ylabel("ln(Ia)")
    plt.title("Langmuir 3/2 law check")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_combined_dashboard(
    x: np.ndarray,
    phi: np.ndarray,
    e: np.ndarray,
    t: np.ndarray,
    x_hist: np.ndarray,
    v_hist: np.ndarray,
    va: np.ndarray,
    i_no_sc: np.ndarray,
    i_sc: np.ndarray | None,
    i_cl: np.ndarray,
    fit_curve: np.ndarray,
    fit_label: str,
    slope: float,
    intercept: float,
    r2: float,
) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("M2 dashboard: stages 1, 2 and 4 (excluding stage 3)", fontsize=14)

    ax = axs[0, 0]
    ax.plot(x * 1e3, phi, color="tab:blue", lw=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Potential φ [V]")
    ax.set_title("Stage 1: electrostatic potential")
    ax.grid(alpha=0.3)

    ax = axs[0, 1]
    ax.plot(x * 1e3, e / 1e3, color="tab:red", lw=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Electric field E [kV/m]")
    ax.set_title("Stage 1: electric field")
    ax.grid(alpha=0.3)

    t_ns = t * 1e9
    ax = axs[1, 0]
    for i in range(x_hist.shape[0]):
        if np.isfinite(x_hist[i]).any():
            ax.plot(t_ns, x_hist[i] * 1e3, lw=0.8, alpha=0.8)
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("x [mm]")
    ax.set_title("Stage 2: electron trajectories")
    ax.grid(alpha=0.3)

    ax = axs[1, 1]
    for i in range(v_hist.shape[0]):
        if np.isfinite(v_hist[i]).any():
            ax.plot(t_ns, v_hist[i] / 1e6, lw=0.8, alpha=0.8)
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("v [10^6 m/s]")
    ax.set_title("Stage 2: electron velocities")
    ax.grid(alpha=0.3)

    ax = axs[2, 0]
    ax.plot(va, i_no_sc * 1e3, "o-", label="No space-charge")
    if i_sc is not None:
        ax.plot(va, i_sc * 1e3, "s-", label="With space-charge")
    ax.plot(va, i_cl * 1e3, "--", label="Child-Langmuir theory")
    ax.plot(va, fit_curve * 1e3, ":", lw=2.0, label=fit_label)
    ax.set_xlabel("Anode voltage Va [V]")
    ax.set_ylabel("Anode current Ia [mA]")
    ax.set_title("Stage 4: I-V characteristics")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[2, 1]
    mask = (va > 0) & (i_no_sc > 0 if i_sc is None else i_sc > 0) & (va >= np.quantile(va, 0.45))
    i_ref = i_no_sc if i_sc is None else i_sc
    lv = np.log(va[mask])
    li = np.log(i_ref[mask])
    ax.plot(lv, li, "o", label="Simulation data")
    ax.plot(lv, slope * lv + intercept, "-", label=f"fit slope={slope:.3f}, R²={r2:.4f}")
    ax.set_xlabel("ln(Va)")
    ax.set_ylabel("ln(Ia)")
    ax.set_title("Stage 4: Langmuir 3/2 check")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])


def animate_electron_motion(
    t: np.ndarray,
    x_hist: np.ndarray,
    gap_length: float,
    max_particles: int = 24,
    trail_points: int = 24,
    interval_ms: int = 20,
) -> FuncAnimation:
    n_particles = min(max_particles, x_hist.shape[0])
    x = x_hist[:n_particles]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_title("Electron motion animation (accelerated time)")
    ax.set_xlabel("x [mm]")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0.0, gap_length * 1e3)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(0.0, color="tab:blue", lw=2, alpha=0.9, label="Cathode")
    ax.axvline(gap_length * 1e3, color="tab:red", lw=2, alpha=0.9, label="Anode")
    ax.legend(loc="upper center", ncol=2)

    y_levels = np.linspace(-0.75, 0.75, n_particles)
    dots = [ax.plot([], [], "o", ms=5)[0] for _ in range(n_particles)]
    trails = [ax.plot([], [], "-", lw=1.0, alpha=0.35)[0] for _ in range(n_particles)]
    time_text = ax.text(0.98, 0.92, "", transform=ax.transAxes, ha="right")

    def _update(frame: int):
        for i in range(n_particles):
            xi = x[i, frame]
            if not np.isfinite(xi) or xi < 0.0 or xi > gap_length:
                dots[i].set_data([], [])
                trails[i].set_data([], [])
                continue
            i0 = max(0, frame - trail_points)
            tail = x[i, i0:frame + 1]
            valid = np.isfinite(tail) & (tail >= 0.0) & (tail <= gap_length)
            if valid.any():
                xt = (tail[valid] * 1e3)
                yt = np.full(xt.size, y_levels[i])
                trails[i].set_data(xt, yt)
            else:
                trails[i].set_data([], [])
            dots[i].set_data([xi * 1e3], [y_levels[i]])
        time_text.set_text(f"t = {t[frame] * 1e9:.2f} ns")
        return [*dots, *trails, time_text]

    ani = FuncAnimation(
        fig,
        _update,
        frames=t.size,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return ani
