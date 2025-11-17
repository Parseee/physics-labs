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
mu = params.get("mu", 0.1)
x0 = params.get("x0", 0.0)
y0 = params.get("y0", 0.0)
vx0 = params.get("vx0", 1.5)
vy0 = params.get("vy0", 0.5)
w0 = params.get("w0", 1.0)

I = (2/5) * m * R**2
t_max = 10.0
tol = 1e-4


# ---------------------------------------------------
# DIFF. EQUATION
# ---------------------------------------------------
def rhs(t, S):
    x, y, vx, vy, w = S
    v = np.array([vx, vy])
    speed = np.linalg.norm(v)

    if speed < 1e-8 or abs(w) < 1e-8:
        return [0, 0, 0, 0, 0]

    slip = speed - w * R

    if speed < 1e-6:
        return [0, 0, 0, 0, 0]

    no_slip = abs(speed - abs(w)*R) < 1e-4

    if no_slip:
        k = 0.2  # rolling resistance coefficient (tunable)
        f_roll = -k * v
        ax, ay = f_roll[0] / m, f_roll[1] / m
        alpha = 0
    else:
        # sliding friction
        f = -mu * m * g * v / speed
        ax = f[0] / m
        ay = f[1] / m
        alpha = (np.linalg.norm(f) * R) / I * np.sign(w)

        F_f = mu * m * g
        f_vec = -F_f * v / speed if speed > 0 else np.array([0, 0])

        ax = f_vec[0] / m
        ay = f_vec[1] / m

        alpha = np.sign(slip) * (F_f * R) / I

    return [vx, vy, ax, ay, alpha]


# ---------------------------------------------------
# SOLVE ODE
# ---------------------------------------------------
t_eval = np.linspace(0, t_max, 20000)
sol = solve_ivp(rhs, (0, t_max), [x0, y0, vx0, vy0, w0], t_eval=t_eval)

t = sol.t
x = sol.y[0]
y = sol.y[1]


# ---------------------------------------------------
# INTERPOLATION FOR SMOOTH ANIMATION
# ---------------------------------------------------
frames = 3000
frame_t = np.linspace(t[0], t[-1], frames)
frame_x = np.interp(frame_t, t, x)
frame_y = np.interp(frame_t, t, y)

interval_ms = 1000 * (frame_t[-1] - frame_t[0]) / frames


# ---------------------------------------------------
# SINGLE-PLOT TABLE + ANIMATED BALL
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
ax.set_aspect('equal')
ax.set_title("Ball Motion on Table")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# Ball
ball = Circle((frame_x[0], frame_y[0]), R, color="royalblue")
ax.add_patch(ball)

# Trajectory line
(path_line,) = ax.plot([], [], lw=2, color="gray")


def init():
    ball.center = (frame_x[0], frame_y[0])
    path_line.set_data([], [])
    return ball, path_line


def update(i):
    ball.center = (frame_x[i], frame_y[i])
    path_line.set_data(frame_x[:i], frame_y[:i])
    return ball, path_line


ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, interval=interval_ms,
    blit=True, repeat=True
)

plt.show()
