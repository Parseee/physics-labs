import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

with open("config.json", "r") as f:
    params = json.load(f)

g = params.get("g", 9.81)
m = params.get("m", 10.5)
R = params.get("R", 0.1)
mu = params.get("mu", 0.5)
x0 = params.get("x0", 0.0)
y0 = params.get("y0", 0.0)
vx0 = params.get("vx0", 1.0)
vy0 = params.get("vy0", 1.0)
w0 = params.get("w0", -10.0)
# OX - slope
theta = np.radians(params.get("alpha", -10.0))

sph_inertia = (2.0 / 5.0) * m * R**2
total_time = 10.0
tol = 1e-5
eps_stop = 1e-5


def rhs(t, S):
    x, y, vx, vy, w = S

    v = np.array([vx, vy])
    speed = np.linalg.norm(v)

    g_down = g * np.sin(theta)

    if abs(g_down) < 1e-12:
        if speed < eps_stop and abs(w) < eps_stop:
            return [0, 0, 0, 0, 0]

    # if speed > 1e-8:
    #     e_s = v / speed
    # else:
    e_s = np.array([1.0, 0.0])

    N = m * g * np.cos(theta)

    v_parallel = v.dot(e_s)
    slip = v_parallel - w * R

    a_gravity = g * np.sin(theta) * e_s

    a_no_slip_scalar = (m * g * np.sin(theta)) / (m + sph_inertia / R**2)
    F_f_required = m * a_no_slip_scalar - m * g * \
        np.sin(theta)

    if abs(slip) <= tol:
        if abs(F_f_required) <= mu * N:
            a_vec = a_no_slip_scalar * e_s
            ax, ay = a_vec[0], a_vec[1]
            alpha = a_no_slip_scalar / R
            if abs(ax) < eps_stop:
                ax = 0.0
            if abs(ay) < eps_stop:
                ay = 0.0
            if abs(alpha) < eps_stop:
                alpha = 0.0
            return [vx, vy, ax, ay, alpha]

    # slipping - sliding
    if abs(slip) > 0:
        slip_dir = np.sign(slip)
    else:
        slip_dir = np.sign(v_parallel) if abs(v_parallel) > 0 else 1.0

    F_f_mag = mu * N
    f_vec = - slip_dir * F_f_mag * e_s
    a_vec = a_gravity + f_vec / m
    ax, ay = a_vec[0], a_vec[1]

    # beta = - (f_parallel * R) / I
    f_parallel = f_vec.dot(e_s)
    alpha = - (f_parallel * R) / sph_inertia

    if abs(ax) < eps_stop:
        ax = 0.0
    if abs(ay) < eps_stop:
        ay = 0.0
    if abs(alpha) < eps_stop:
        alpha = 0.0

    return [vx, vy, ax, ay, alpha]


# интегрирование
t_eval = np.linspace(0.0, total_time, 2000)
y0_vec = [x0, y0, vx0, vy0, w0]
solution = solve_ivp(rhs, (0.0, total_time), y0_vec,
                     t_eval=t_eval, rtol=1e-8, atol=1e-10, method='RK45')

t = solution.t
x = solution.y[0]
y = solution.y[1]

linear_speed = np.sqrt(solution.y[2]**2 + solution.y[3]**2)
angular_speed = solution.y[4]
K_trans_sol = 0.5 * m * linear_speed**2
K_rotat_sol = 0.5 * sph_inertia * solution.y[4]**2
total_energy = K_trans_sol + K_rotat_sol


# --- Анимация (упрощённо, как у вас) ---
frames = 400
frame_t = np.linspace(t[0], t[-1], frames)
frame_x = np.interp(frame_t, t, x)
frame_y = np.interp(frame_t, t, y)

interval_ms = 1000 * (frame_t[-1] - frame_t[0]) / frames

fig, (ax_sim, ax_speed, ax_omega, ax_energy) = plt.subplots(
    4, 1, figsize=(7, 11),
    gridspec_kw={'height_ratios': [4, 1.5, 1.5, 1.5]}
)


ax_sim.set_xlim(min(x) - 0.5, max(x) + 0.5)
ax_sim.set_ylim(min(y) - 0.5, max(y) + 0.5)
ax_sim.set_aspect('equal')
ax_sim.set_title("Ball Motion on Surface")
ax_sim.set_xlabel("x [m]")
ax_sim.set_ylabel("y [m]")

ball = Circle((frame_x[0], frame_y[0]), R, color="red")
ax_sim.add_patch(ball)
(line_path,) = ax_sim.plot([], [], lw=2, color="gray")

ax_speed.set_xlim(frame_t[0], frame_t[-1])
ax_speed.set_ylim(0, max(linear_speed) * 1.1 if linear_speed.max() > 0
                  else 1.0)
ax_speed.set_title("Speed |v(t)|")
ax_speed.set_xlabel("time [s]")
(line_speed,) = ax_speed.plot([], [], lw=2, color="blue")

ax_omega.set_xlim(frame_t[0], frame_t[-1])
# ax_omega.set_ylim(min(angular_speed)-1, max(angular_speed)+1)
w_min, w_max = min(angular_speed), max(angular_speed)
if w_min == w_max:
    # Если скорость константна или 0, делаем искусственный отступ
    ax_omega.set_ylim(w_min - 1.0, w_max + 1.0)
else:
    ax_omega.set_ylim(w_min * 1.1 if w_min < 0 else w_min * 0.9,
                      w_max * 1.1 if w_max > 0 else w_max * 0.9)
ax_omega.set_title("Angular Speed ω(t)")
ax_omega.set_xlabel("time [s]")
(line_omega,) = ax_omega.plot([], [], lw=2, color="green")

ax_energy.set_xlim(frame_t[0], frame_t[-1])
ax_energy.set_ylim(0, max(total_energy) *
                   1.1 if total_energy.max() > 0 else 1.0)
ax_energy.set_title("Total Kinetic Energy |T(t)|")
ax_energy.set_xlabel("time [s]")
(line_energy,) = ax_energy.plot([], [], lw=2, color="orange")


def init():
    ball.center = (frame_x[0], frame_y[0])
    line_path.set_data([], [])
    line_speed.set_data([], [])
    line_omega.set_data([], [])
    line_energy.set_data([], [])

    return ball, line_path, line_speed, line_omega, line_energy


def update(frame):
    ball.center = (frame_x[frame], frame_y[frame])
    line_path.set_data(frame_x[:frame], frame_y[:frame])
    line_speed.set_data(frame_t[:frame], np.interp(
        frame_t[:frame], t, linear_speed))
    line_omega.set_data(frame_t[:frame], np.interp(
        frame_t[:frame], t, angular_speed))
    line_energy.set_data(frame_t[:frame], np.interp(
        frame_t[:frame], t, total_energy))
    return ball, line_path, line_speed, line_omega, line_energy


ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, interval=interval_ms,
    blit=True, repeat=True
)

plt.show()
