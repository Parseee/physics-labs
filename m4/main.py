import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

with open("config.json", "r") as f:
    params = json.load(f)

g = params.get("g", 9.81)
m = params.get("m", 0.5)
R = params.get("R", 0.1)
mu = params.get("mu", 0.8)
x0 = params.get("x0", 0.0)
y0 = params.get("y0", 0.0)
vx0 = params.get("vx0", 10)
vy0 = params.get("vy0", 0)
w0 = params.get("w0", -260.0)
theta = np.radians(params.get("alpha", 0))  # in x-axis

sph_inertia = (2/5) * m * R**2
total_time = 10.0
tol = 1e-4


def rhs(t, S):
    x, y, vx, vy, w = S
    v = np.array([vx, vy])
    speed = np.linalg.norm(v)

    e_s = np.array([np.cos(theta), np.sin(theta)])
    N = m * g * np.cos(theta)

    v_parallel = v.dot(e_s)
    slip = v_parallel - w * R

    if abs(slip) < tol:
        a_scalar = m * g * np.sin(theta) / (m + sph_inertia / R**2)

        F_s = - (sph_inertia / R**2) * a_scalar
        if abs(F_s) <= mu * N:
            a_vec = a_scalar * e_s
            alpha = a_scalar / R
            ax, ay = a_vec[0], a_vec[1]
            return [vx, vy, ax, ay, alpha]

    if abs(slip) < tol and speed < tol:
        slip_sign = np.sign(slip)
    else:
        slip_sign = np.sign(slip) if abs(slip) > 0 else (
            np.sign(v_parallel) if abs(v_parallel) > 0 else 1.0)

    F_f = mu * N
    f_vec = - slip_sign * F_f * e_s

    a_gravity = g * np.sin(theta) * e_s
    a_vec = f_vec / m + a_gravity

    ax, ay = a_vec[0], a_vec[1]
    alpha = slip_sign * (F_f * R) / sph_inertia

    return [vx, vy, ax, ay, alpha]


t_eval = np.linspace(0, total_time, 20000)
solution = solve_ivp(rhs, (0, total_time), [
                     x0, y0, vx0, vy0, w0], t_eval=t_eval)

t = solution.t
x = solution.y[0]
y = solution.y[1]

total_speed = np.sqrt(solution.y[2]**2 + solution.y[3]**2)
K_trans_sol = 0.5 * m * total_speed**2
K_rotat_sol = 0.5 * sph_inertia * solution.y[4]**2
total_energy = K_trans_sol + K_rotat_sol


frames = 3000
frame_t = np.linspace(t[0], t[-1], frames)
frame_x = np.interp(frame_t, t, x)
frame_y = np.interp(frame_t, t, y)

interval_ms = 1000 * (frame_t[-1] - frame_t[0]) / frames


fig, (ax_sim, ax_speed, ax_energy) = plt.subplots(
    3, 1, figsize=(7, 9), gridspec_kw={'height_ratios': [4, 1.5, 1.5]})

ax_sim.set_xlim(min(x) - 0.5, max(x) + 0.5)
ax_sim.set_ylim(min(y) - 0.5, max(y) + 0.5)
ax_sim.set_aspect('equal')  # sides ratio
ax_sim.set_title("Ball Motion on Table")
ax_sim.set_xlabel("x [m]")
ax_sim.set_ylabel("y [m]")


ball = Circle((frame_x[0], frame_y[0]), R, color="royalblue")
ax_sim.add_patch(ball)
(line_path,) = ax_sim.plot([], [], lw=2, color="gray")

ax_speed.set_xlim(frame_t[0], frame_t[-1])
ax_speed.set_ylim(0, max(total_speed) * 1.1)
ax_speed.set_title("Speed |v(t)|")
ax_speed.set_xlabel("time [s]")
min_length = min(len(frame_t), len(total_speed))
(line_speed,) = ax_speed.plot(
    frame_t[:min_length], total_speed[:min_length], lw=2, color='blue')

ax_energy.set_xlim(frame_t[0], frame_t[-1])
ax_energy.set_ylim(0, max(total_energy) * 1.1)
ax_energy.set_title("Total Energy |E(t)|")
ax_energy.set_xlabel("time [s]")
min_length = min(len(frame_t), len(total_energy))
(line_energy,) = ax_energy.plot(
    frame_t[:min_length], total_energy[:min_length], lw=2, color='red')


def init():
    ball.center = (frame_x[0], frame_y[0])
    line_path.set_data([], [])
    return ball, line_path


def update(frame):
    ball.center = (frame_x[frame], frame_y[frame])
    line_path.set_data(frame_x[:frame], frame_y[:frame])
    return ball, line_path


ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, interval=interval_ms,
    blit=True, repeat=True
)

plt.show()
