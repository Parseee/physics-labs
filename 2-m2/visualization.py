import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_combined_dashboard(
    x,
    phi,
    e,
    t,
    x_hist,
    v_hist,
    va,
    i_no_sc,
    i_sc,
    i_cl,
    fit_curve,
    fit_label,
    slope,
    intercept,
    r2,
):
    fig, axs = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("M2 dashboard: stages 1, 2 and 4 (excluding stage 3)", fontsize=14)

    axs[0, 0].plot(x * 1e3, phi, lw=2)
    axs[0, 0].set_title("Stage 1: potential")
    axs[0, 0].set_xlabel("x [mm]")
    axs[0, 0].set_ylabel("phi [V]")
    axs[0, 0].grid(alpha=0.3)

    axs[0, 1].plot(x * 1e3, e / 1e3, lw=2, color="tab:red")
    axs[0, 1].set_title("Stage 1: electric field")
    axs[0, 1].set_xlabel("x [mm]")
    axs[0, 1].set_ylabel("E [kV/m]")
    axs[0, 1].grid(alpha=0.3)

    t_ns = t * 1e9
    for row in x_hist:
        if np.isfinite(row).any():
            axs[1, 0].plot(t_ns, row * 1e3, lw=0.8, alpha=0.8)
    axs[1, 0].set_title("Stage 2: trajectories")
    axs[1, 0].set_xlabel("t [ns]")
    axs[1, 0].set_ylabel("x [mm]")
    axs[1, 0].grid(alpha=0.3)

    for row in v_hist:
        if np.isfinite(row).any():
            axs[1, 1].plot(t_ns, row / 1e6, lw=0.8, alpha=0.8)
    axs[1, 1].set_title("Stage 2: velocities")
    axs[1, 1].set_xlabel("t [ns]")
    axs[1, 1].set_ylabel("v [10^6 m/s]")
    axs[1, 1].grid(alpha=0.3)

    axs[2, 0].plot(va, i_no_sc * 1e3, "o-", label="No space-charge")
    if i_sc is not None:
        axs[2, 0].plot(va, i_sc * 1e3, "s-", label="With space-charge")
    axs[2, 0].plot(va, i_cl * 1e3, "--", label="Child-Langmuir")
    axs[2, 0].plot(va, fit_curve * 1e3, ":", lw=2.0, label=fit_label)
    axs[2, 0].set_title("Stage 4: I-V")
    axs[2, 0].set_xlabel("Va [V]")
    axs[2, 0].set_ylabel("Ia [mA]")
    axs[2, 0].grid(alpha=0.3)
    axs[2, 0].legend()

    i_ref = i_no_sc if i_sc is None else i_sc
    mask = (va > 0) & (i_ref > 0) & (va >= np.quantile(va, 0.45))
    lv = np.log(va[mask])
    li = np.log(i_ref[mask])
    axs[2, 1].plot(lv, li, "o", label="Simulation")
    axs[2, 1].plot(lv, slope * lv + intercept, "-", label=f"n={slope:.3f}, R²={r2:.4f}")
    axs[2, 1].set_title("Stage 4: Langmuir check")
    axs[2, 1].set_xlabel("ln(Va)")
    axs[2, 1].set_ylabel("ln(Ia)")
    axs[2, 1].grid(alpha=0.3)
    axs[2, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])


def animate_electron_motion(t, x_hist, gap_length, max_particles=24, trail_points=24, interval_ms=20):
    n_particles = min(max_particles, x_hist.shape[0])
    x = x_hist[:n_particles]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_title("Electron motion animation")
    ax.set_xlabel("x [mm]")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0.0, gap_length * 1e3)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(0.0, lw=2, alpha=0.9, label="Cathode")
    ax.axvline(gap_length * 1e3, lw=2, alpha=0.9, color="tab:red", label="Anode")
    ax.legend(loc="upper center", ncol=2)

    y_levels = np.linspace(-0.75, 0.75, n_particles)
    dots = [ax.plot([], [], "o", ms=5)[0] for _ in range(n_particles)]
    trails = [ax.plot([], [], "-", lw=1.0, alpha=0.35)[0] for _ in range(n_particles)]
    time_text = ax.text(0.98, 0.92, "", transform=ax.transAxes, ha="right")

    def update(frame):
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
                xt = tail[valid] * 1e3
                yt = np.full(xt.size, y_levels[i])
                trails[i].set_data(xt, yt)
            else:
                trails[i].set_data([], [])
            dots[i].set_data([xi * 1e3], [y_levels[i]])

        time_text.set_text(f"t = {t[frame] * 1e9:.2f} ns")
        return [*dots, *trails, time_text]

    return FuncAnimation(fig, update, frames=t.size, interval=interval_ms, blit=False, repeat=True)
