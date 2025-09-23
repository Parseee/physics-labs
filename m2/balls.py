import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json

# -------- config ---------
with open("config.json", "r") as f:
    params = json.load(f)

g = params.get("g", 9.81)                       # free fall acceleration
m = params.get("m", 0.100)                      # mass of stone
C_d = params.get("C_d", 0.47)                   # dimensionless k
rho = params.get("rho", 1.225)                  # air density kg/m^3
r = params.get("r", 0.02)                       # stone radius (m)
initial_velocity = params.get("initial_velocity", 20.0)
angle_of_projection = params.get("angle_of_projection", 45)
# -------------------------

A = np.pi * r**2
C_eff = 0.5 * C_d * rho * A


def equations(t, state):
    x, vx, y, vy = state
    v = np.hypot(vx, vy)
    if v == 0.0:
        ax = 0.0
        ay = 0.0
    else:
        ax = - (C_eff / m) * v * vx
        ay = - (C_eff / m) * v * vy
    dxdt = vx
    dvxdt = ax
    dydt = vy
    dvydt = -g + ay
    return [dxdt, dvxdt, dydt, dvydt]


def numeric_quadratic(v0, theta, dt=0.05):
    t_land = 2 * v0 * np.sin(np.radians(theta)) / g
    t_span = (0, t_land)
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    x = 0.0
    y = 0.0
    v0x = v0 * np.cos(np.radians(theta))
    v0y = v0 * np.sin(np.radians(theta))

    solution = solve_ivp(equations, t_span,
                         [x, v0x, y, v0y],
                         t_eval=t_eval)

    return solution.y[0], solution.y[2]


traj_numeric_x, traj_numeric_y = numeric_quadratic(
    initial_velocity, angle_of_projection)

fig, ax = plt.subplots()
ax.plot(traj_numeric_x, traj_numeric_y, lw=2, color='red', label='numeric')
ax.set_xlim(0, max(traj_numeric_x) + 1)
ax.set_ylim(0, max(traj_numeric_y) + 1)
ax.set_title('Trajectory of a Thrown Stone')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
ax.legend()

constants_text = f"""
constants:
g = {g} m/s^2
k = {C_d}
m = {m} kg
Θ = {angle_of_projection} deg
v = {initial_velocity} m/s
"""
plt.text(0.05, 0.95, constants_text, transform=ax.transAxes, fontsize=8,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

landing_x = max(traj_numeric_x)
plt.text(0.1, 0.1, f"Landing point ({landing_x:.2f})",
         transform=ax.transAxes, fontsize=9)

plt.show()
