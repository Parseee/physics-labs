import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


with open("config.json", "r") as f:
    params = json.load(f)
    m1 = params.get("m1", 0.1)
    m2 = params.get("m2", 0.1)
    p = params.get("p", 0.45)
    k_hooke = params.get("ball_hooke", 1e5)
    r = params.get("r", 0.05)

    # позиции и угол первого шара
    x1_0 = params.get("x1_0", 0.25)
    y1_0 = params.get("y1_0", 0.15)
    angle_deg = params.get("angle_deg", 10)

    # позиции второго шара
    x2_0 = params.get("x2_0", 0.35)
    y2_0 = params.get("y2_0", 0.15)


# Simulation parameters
dt = 1e-4
t_max = 5.0
table_x1, table_x2 = 0.0, 0.5
table_y1, table_y2 = 0.0, 0.3


angle = np.deg2rad(angle_deg)
vx1, vy1 = (p / m1) * np.cos(angle), (p / m1) * np.sin(angle)
vx2, vy2 = 0.0, 0.0

x1, y1 = x1_0, y1_0
x2, y2 = x2_0, y2_0

times, x1s, y1s, x2s, y2s, E_s = [0], [x1], [y1], [x2], [y2], []


def wall_reflection(x, y, vx, vy):
    """Отражения от стен без потерь энергии"""
    if x - r < table_x1:
        x = table_x1 + r
        vx = abs(vx)
    elif x + r > table_x2:
        x = table_x2 - r
        vx = -abs(vx)

    if y - r < table_y1:
        y = table_y1 + r
        vy = abs(vy)
    elif y + r > table_y2:
        y = table_y2 - r
        vy = -abs(vy)

    return x, y, vx, vy


def potential_energy(x, y):
    e = 0.0
    if x - r < table_x1:
        e += 0.5 * k_hooke * (table_x1 - (x - r))**2
    if x + r > table_x2:
        e += 0.5 * k_hooke * ((x + r) - table_x2)**2
    if y - r < table_y1:
        e += 0.5 * k_hooke * (table_y1 - (y - r))**2
    if y + r > table_y2:
        e += 0.5 * k_hooke * ((y + r) - table_y2)**2
    return e


def ball_collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.hypot(dx, dy)

    if dist < 2 * r:
        nx, ny = dx / dist, dy / dist

        # fix balls positions
        overlap = 2 * r - dist
        x1 -= nx * overlap / 2
        y1 -= ny * overlap / 2
        x2 += nx * overlap / 2
        y2 += ny * overlap / 2

        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)
        nx, ny = dx / dist, dy / dist
        tx, ty = -ny, nx  # normal equation

        # (n, t)
        v1n = vx1 * nx + vy1 * ny
        v1t = vx1 * tx + vy1 * ty
        v2n = vx2 * nx + vy2 * ny
        v2t = vx2 * tx + vy2 * ty

        # new impulse
        v1n_new = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
        v2n_new = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)

        vx1_new = v1n_new * nx + v1t * tx
        vy1_new = v1n_new * ny + v1t * ty
        vx2_new = v2n_new * nx + v2t * tx
        vy2_new = v2n_new * ny + v2t * ty

        return vx1_new, vy1_new, vx2_new, vy2_new, x1, y1, x2, y2

    return vx1, vy1, vx2, vy2, x1, y1, x2, y2


t = 0.0
while t < t_max:
    x1 += vx1 * dt
    y1 += vy1 * dt
    x2 += vx2 * dt
    y2 += vy2 * dt

    x1, y1, vx1, vy1 = wall_reflection(x1, y1, vx1, vy1)
    x2, y2, vx2, vy2 = wall_reflection(x2, y2, vx2, vy2)

    result = ball_collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    vx1, vy1, vx2, vy2, x1, y1, x2, y2 = result

    E_kin = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
    E_pot = potential_energy(x1, y1) + potential_energy(x2, y2)
    E_s.append(E_kin + E_pot)

    t += dt
    times.append(t)
    x1s.append(x1)
    y1s.append(y1)
    x2s.append(x2)
    y2s.append(y2)


# -----------------------
# Precompute the animation
# -----------------------
n_frames = 2000
frame_times = np.linspace(times[0], times[-1], n_frames)
frame_x1s = np.interp(frame_times, times, x1s)
frame_y1s = np.interp(frame_times, times, y1s)
frame_x2s = np.interp(frame_times, times, x2s)
frame_y2s = np.interp(frame_times, times, y2s)
frame_Es = np.interp(frame_times, times[:-1], E_s)

interval_ms = 1000 * (frame_times[-1] - frame_times[0]) / n_frames


fig, (ax_table, ax_energy) = plt.subplots(
    2, 1, figsize=(6, 4),  gridspec_kw={'height_ratios': [2, 1]})
ax_table.set_xlim(table_x1 - 0.05, table_x2 + 0.05)
ax_table.set_ylim(table_y1 - 0.05, table_y2 + 0.05)
ax_table.set_aspect('equal')
ax_table.set_xticks([])
ax_table.set_yticks([])
ax_table.set_title("Two balls billiard simulation")

# table
ax_table.plot([table_x1, table_x2, table_x2, table_x1, table_x1],
              [table_y1, table_y1, table_y2, table_y2, table_y1],
              'k', lw=2)

ball1 = Circle((x1_0, y1_0), radius=r, color='tab:blue')
ball2 = Circle((x2_0, y2_0), radius=r, color='tab:orange')
ax_table.add_patch(ball1)
ax_table.add_patch(ball2)

ax_energy.set_xlim(frame_times[0], frame_times[-1])
ax_energy.set_ylim(min(frame_Es) * 0.98, max(frame_Es) * 1.02)
ax_energy.set_xlabel("time [s]")
ax_energy.set_ylabel("total energy [J]")
(line_energy,) = ax_energy.plot([], [], lw=2, color='green')


def init():
    ball1.center = (frame_x1s[0], frame_y1s[0])
    ball2.center = (frame_x2s[0], frame_y2s[0])
    line_energy.set_data([], [])
    return ball1, ball2, line_energy


def update(frame):
    ball1.center = (frame_x1s[frame], frame_y1s[frame])
    ball2.center = (frame_x2s[frame], frame_y2s[frame])
    line_energy.set_data(frame_times[:frame], frame_Es[:frame])
    return ball1, ball2, line_energy


ani = animation.FuncAnimation(fig, update, frames=n_frames,
                              init_func=init, interval=interval_ms,
                              blit=True, repeat=True)

plt.show()
