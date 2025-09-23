import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# -----------------------
# User input
# -----------------------
with open("config.json", "r") as f:
    params = json.load(f)
    m = params.get("m", 0.1)
    p = params.get("p", 1)
    k_hooke = params.get("wall_hooke", 1e5)
    ball_radius = params.get("r", 0.05)
    x0 = params.get("innitial_point_x", 0.25)
    y0 = params.get("innitial_point_y", 0.15)
    angle_deg = params.get("angle_deg", 30)

# -----------------------
# Simulation parameters
# -----------------------
dt = 1e-4
t_max = 5.0

table_x1, table_x2 = 0.0, 0.5
table_y1, table_y2 = 0.0, 0.3

# -----------------------
# Energy-based functions
# -----------------------


def potential_energy(x, y):
    left_overlap = max(0, ball_radius - abs(x - table_x1))
    right_overlap = max(0, ball_radius - abs(table_x2 - x))
    bottom_overlap = max(0, ball_radius - abs(y - table_y1))
    top_overlap = max(0, ball_radius - abs(table_y2 - y))
    return 0.5 * k_hooke * (left_overlap**2 + right_overlap**2 +
                            bottom_overlap**2 + top_overlap**2)


def contact_forces(x, y):
    Fx = 0
    Fy = 0
    left_overlap = max(0, ball_radius - abs(x - table_x1))
    right_overlap = max(0, ball_radius - abs(table_x2 - x))
    bottom_overlap = max(0, ball_radius - abs(y - table_y1))
    top_overlap = max(0, ball_radius - abs(table_y2 - y))

    Fx += k_hooke * left_overlap
    Fx -= k_hooke * right_overlap
    Fy += k_hooke * bottom_overlap
    Fy -= k_hooke * top_overlap
    return Fx, Fy


# -----------------------
# Time integration
# -----------------------
times, xs, ys, vxs, vys, Fxs, Fys, E_s = [0], [x0], [y0], [], [], [], [], []


def simulate():
    angle = np.deg2rad(angle_deg)
    vx = (p / m) * np.cos(angle)
    vy = (p / m) * np.sin(angle)
    x, y = x0, y0
    t = 0.0
    E_total = 0.5 * m * (vx**2 + vy**2) + potential_energy(x, y)

    while t < t_max:
        Fx, Fy = contact_forces(x, y)
        Fxs.append(Fx)
        Fys.append(Fy)
        E_s.append(E_total)

        # энергия -> скорость (по компонентам)
        # (но тут проще обновлять vx, vy через силы)
        ax = Fx / m
        ay = Fy / m
        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        t += dt
        times.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)


simulate()

# -----------------------
# Prepare animation data
# -----------------------
n_frames = 1000
real_times = np.array(times)
frame_times = np.linspace(real_times[0], real_times[-1], n_frames)
frame_xs = np.interp(frame_times, real_times, xs)
frame_ys = np.interp(frame_times, real_times, ys)
frame_Fxs = np.interp(frame_times, real_times[:-1], Fxs)
frame_Fys = np.interp(frame_times, real_times[:-1], Fys)
frame_Es = np.interp(frame_times, real_times[:-1], E_s)

interval_ms = 1000 * (frame_times[-1] - frame_times[0]) / n_frames

# -----------------------
# Create figure with subplots
# -----------------------
fig, (ax_table, ax_force, ax_energy) = plt.subplots(
    3, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1, 1]})
fig.tight_layout(pad=3.0)

# --- Top subplot: Table with ball ---
ax_table.set_xlim(table_x1 - 0.05, table_x2 + 0.05)
ax_table.set_ylim(table_y1 - 0.05, table_y2 + 0.05)
ax_table.set_aspect('equal')
ax_table.set_xticks([])
ax_table.set_yticks([])
ax_table.set_title("2D billiard ball motion (energy-based)")

# Draw table as rectangle
ax_table.plot([table_x1, table_x2, table_x2, table_x1, table_x1],
              [table_y1, table_y1, table_y2, table_y2, table_y1],
              'k', lw=2)

ball = Circle((x0, y0), radius=ball_radius, color='tab:blue')
ax_table.add_patch(ball)

# --- subplot: Force graph (magnitude) ---
F_magnitude = np.sqrt(np.array(Fxs)**2 + np.array(Fys)**2)
ax_force.set_xlim(frame_times[0], frame_times[-1])
ax_force.set_ylim(0, max(F_magnitude) * 1.1 if len(F_magnitude) else 1)
ax_force.set_xlabel("time [s]")
ax_force.set_ylabel("|F| [N]")
(line_force,) = ax_force.plot([], [], lw=2, color='red')

# --- subplot: Energy graph ---
ax_energy.set_xlim(frame_times[0], frame_times[-1])
ax_energy.set_ylim(min(frame_Es) - 0.05*abs(min(frame_Es)),
                   max(frame_Es) + 0.05*abs(max(frame_Es)))
ax_energy.set_xlabel("time [s]")
ax_energy.set_ylabel("total energy [J]")
(line_energy,) = ax_energy.plot([], [], lw=2, color='green')

# -----------------------
# Animation update functions
# -----------------------


def init():
    ball.center = (frame_xs[0], frame_ys[0])
    line_force.set_data([], [])
    line_energy.set_data([], [])
    return ball, line_force, line_energy


def update(frame):
    ball.center = (frame_xs[frame], frame_ys[frame])
    line_force.set_data(frame_times[:frame], np.sqrt(
        frame_Fxs[:frame]**2 + frame_Fys[:frame]**2))
    line_energy.set_data(frame_times[:frame], frame_Es[:frame])
    return ball, line_force, line_energy


ani = animation.FuncAnimation(fig, update, frames=n_frames,
                              init_func=init, interval=interval_ms,
                              blit=True, repeat=True)

plt.show()
