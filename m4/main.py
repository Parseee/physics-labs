import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

# ---------------------------------------------------
# LOAD PARAMETERS
# ---------------------------------------------------
with open("config.json", "r") as f:
    params = json.load(f)

g = params.get("g", 9.81)
m = params.get("m", 0.5)
R = params.get("R", 0.1)
mu = params.get("mu", 0.2)
x0 = params.get("x0", 0.0)
y0 = params.get("y0", 0.0)
vx0 = params.get("vx0", 0)
vy0 = params.get("vy0", 10)
w0 = params.get("w0", 10.0)
theta = np.radians(params.get("alpha", 25))  # in x-axis

sph_inertia = (2/5) * m * R**2
total_time = 3.0
tol = 1e-4


def rhs(t, S):
    x, y, vx, vy, w = S
    v = np.array([vx, vy])
    speed = np.linalg.norm(v)

    # if speed < 1e-8 or abs(w) < 1e-8:
    #     return [0, 0, 0, 0, 0]

    slip_dir = np.sign(speed - w * R)
    no_slip = abs(speed - w * R) < tol

    if no_slip:
        print("no slip")
        k = 0.2  # rolling resistance coefficient
        f_roll = -k * v
        ax, ay = f_roll[0] / m + g * np.sin(theta), f_roll[1] / m
        alpha = 0
    else:
        print("no slip")
        f = -mu * m * g * v / speed
        ax = f[0] / m + g * np.sin(theta)
        ay = f[1] / m
        alpha = (np.linalg.norm(f) * R) / sph_inertia * np.sign(w)

        F_f = mu * m * g
        f_vec = -F_f * v / speed if speed > 0 else np.array([0, 0])

        ax = f_vec[0] / m + g * np.sin(theta)
        ay = f_vec[1] / m

        alpha = slip_dir * (F_f * R) / sph_inertia

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

# ---------------------------------------------------
# SINGLE-PLOT TABLE + ANIMATED BALL
# ---------------------------------------------------
fig, (ax_sim, ax_speed, ax_energy) = plt.subplots(
    3, 1, figsize=(7, 9), gridspec_kw={'height_ratios': [4, 1.5, 1.5]})

ax_sim.set_xlim(min(x) * 0.9, max(x) * 1.1)
ax_sim.set_ylim(min(y) * 0.9, max(y) * 1.1)
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
