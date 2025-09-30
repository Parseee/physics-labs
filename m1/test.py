import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json

# -------- config ---------
with open("config.json", "r") as f:
    params = json.load(f)

g = params.get("g", 9.81)                       # free fall acceleration
k = params.get("k", 0.47)                       # resistance coefficient
m = params.get("m", 0.100)                      # mass of stone
initial_velocity = params.get("initial_velocity", 20.0)
angle_of_projection = params.get("angle_of_projection", 45)
# -------------------------

time_step = 0.0005          # Time step in seconds

gamma = k / m

if gamma == 0:
    gamma = 1/g


def symbolic_linear(v0, theta, dt=0.0005):
    theta_rad = np.radians(theta)
    print(theta_rad)

    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)

    x, y = 0.0, 0.0

    traj_x = []
    traj_y = []

    t = 0.0

    while y >= 0:
        traj_x.append(x)
        traj_y.append(y)

        x = ((vx / gamma) * (1 - np.exp(-gamma * t)))
        y = ((1 / gamma) * (vy + g / gamma) *
             (1 - np.exp(-gamma * t)) - (g * t) / gamma)

        t += dt

    traj_x.append(x)  # add last bit of calculation
    traj_y.append(y)  # add last bit of calculation

    return traj_x, traj_y


def equations(t, state):
    x, vx, y, vy = state
    dxdt = vx
    dvxdt = -gamma * vx
    dydt = vy
    dvydt = -g - gamma * vy
    return [dxdt, dvxdt, dydt, dvydt]


def numeric_linear(v0, theta, dt=0.05):
    theta_rad = np.radians(theta)
    t_land = 2 * v0 * np.sin(theta_rad) * (1 - k) / g
    t_span = (0, t_land)
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    x = 0.0
    y = 0.0
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)

    # --- event function to detect ground hit ---
    def hit_ground(t, state):
        return state[2]  # y position
    hit_ground.terminal = True
    hit_ground.direction = -1
    # -------------------------------------------

    solution = solve_ivp(equations, t_span,
                         [x, v0x, y, v0y],
                         t_eval=t_eval,
                         events=hit_ground)

    if len(solution.y) > 0:
        print(solution)
        return solution.y[0], solution.y[2]
    return [0], [0]


traj_symbolic_x, traj_symbolic_y = symbolic_linear(
    initial_velocity, angle_of_projection, time_step)

traj_numeric_x, traj_numeric_y = numeric_linear(
    initial_velocity, angle_of_projection, time_step)

fig, ax = plt.subplots()
ax.set_xlim(0,
            max(max(traj_symbolic_x), max(traj_numeric_x)) +
            max(max(traj_symbolic_x), max(traj_numeric_x)) * 0.1)
ax.set_ylim(0,
            max(max(traj_symbolic_y), max(traj_numeric_y)) +
            max(max(traj_symbolic_y), max(traj_numeric_y)) * 0.1)
ax.set_title('Trajectory of a Thrown Stone')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
ax.plot(traj_symbolic_x, traj_symbolic_y, lw=2, color='red', label='Symbolic')
ax.plot(traj_numeric_x, traj_numeric_y, lw=2, color='blue', label='Numerical')
ax.legend()

constants_text = f"""
Constants:
g = {g} m/s^2
k = {k}
m = {m} kg
Θ = {angle_of_projection} deg
v = {initial_velocity} m/s
"""
plt.text(0.05, 0.95, constants_text, transform=ax.transAxes, fontsize=8,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.text(
    0.1, 0.1, f"Landing point ({max(
        x for (x, y) in zip(traj_numeric_x, traj_numeric_y)):.2f})")

plt.show()
