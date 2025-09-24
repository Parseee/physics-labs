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


dt = 1e-4
t_max = 5.0
wall_x_1 = 0.0
wall_x_2 = 0.5


def contact_force(x):
    l_dx = max(0, ball_radius - abs(x - wall_x_1))
    r_dx = max(0, ball_radius - abs(wall_x_2 - x))
    return k_hooke * (l_dx - r_dx)


times, xs, vs, F_conts, E_s = [0], [x0], [p / m], [], []


def simulate():
    v_in = p / m
    x = x0
    v = -v_in
    t = 0
    while t < t_max:
        F_cont = contact_force(x)
        F_conts.append(F_cont)
        E_s.append((m * v**2) / 2 + ((F_cont**2 / k_hooke)) / 2)
        a = F_cont / m
        v += a * dt
        x += v * dt
        t += dt
        times.append(t)
        xs.append(x)
        vs.append(v)


simulate()

# Prepare animation data
n_frames = 1000
real_times = np.array(times)
frame_times = np.linspace(real_times[0], real_times[-1], n_frames)
frame_xs = np.interp(frame_times, real_times, xs)
frame_Fs = np.interp(frame_times, real_times[:-1], F_conts)
frame_Es = np.interp(frame_times, real_times[:-1], E_s)

interval_ms = 1000 * (frame_times[-1] - frame_times[0]) / n_frames

fig, (ax_ball, ax_force, ax_momentum) = plt.subplots(
    3, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.tight_layout(pad=3.0)

# Ball motion subplot
ax_ball.set_xlim(-0.05, 0.55)
ax_ball.set_ylim(-0.1, 0.1)
ax_ball.set_aspect('equal')
ax_ball.set_yticks([])
ax_ball.set_xlabel("x [m]")
ax_ball.set_title("Ball bouncing between two walls")

# Draw walls
ax_ball.plot([wall_x_1, wall_x_1], [-0.1, 0.1], 'k', lw=3)
ax_ball.plot([wall_x_2, wall_x_2], [-0.1, 0.1], 'k', lw=3)

ball = Circle((x0, 0), radius=ball_radius, color='tab:blue')
ax_ball.add_patch(ball)

# Force graph
ax_force.set_xlim(frame_times[0], frame_times[-1])
ax_force.set_ylim(min(frame_Fs) - 0.1*abs(min(frame_Fs)),
                  max(frame_Fs) + 0.1*abs(max(frame_Fs)))
ax_force.set_xlabel("time [s]")
ax_force.set_ylabel("contact force [N]")
(line_force,) = ax_force.plot([], [], lw=2, color='red')

# Energy graph
ax_momentum.set_xlim(frame_times[0], frame_times[-1])
ax_momentum.set_ylim(min(frame_Es) - 0.1*abs(min(frame_Es)),
                     max(frame_Es) + 0.1*abs(max(frame_Es)))
ax_momentum.set_xlabel("Time [s]")
ax_momentum.set_ylabel("Total energy [J]")
(line_momentum,) = ax_momentum.plot([], [], lw=2, color='orange')


def init():
    ball.center = (frame_xs[0], 0)
    line_force.set_data([], [])
    line_momentum.set_data([], [])
    return ball, line_force


def update(frame):
    ball.center = (frame_xs[frame], 0)
    line_force.set_data(frame_times[:frame], frame_Fs[:frame])
    line_momentum.set_data(frame_times[:frame], frame_Es[:frame])
    return ball, line_force, line_momentum


ani = animation.FuncAnimation(fig, update, frames=n_frames,
                              init_func=init, interval=interval_ms,
                              blit=True, repeat=True)

plt.show()
