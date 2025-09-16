import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

g = 9.81  # free fall acceleration
k = 0.1  # resistance coefficient
m = 0.100   # mass of stone

initial_velocity = 20.0  # Initial velocity in m/s
angle_of_projection = 45  # Angle in degrees
time_step = 0.05  # Time step in seconds


def symbolic_linear(v0, theta, dt=0.05):
    theta_rad = np.radians(theta)

    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)

    gamma = k / m

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
    dvxdt = -k/m * vx
    dydt = vy
    dvydt = -g - k/m * vy
    return [dxdt, dvxdt, dydt, dvydt]


def numeric_linear(v0, theta, dt=0.05):
    t_land = 2 * v0 * np.sin(theta) * (1 - k) / g
    t_span = (0, t_land)
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    x = 0.0
    y = 0.0
    v0x = v0 * np.cos(np.radians(theta))
    v0y = v0 * np.sin(np.radians(theta))

    solution = solve_ivp(equations, t_span,
                         [x, v0x, y, v0y],
                         t_eval=t_eval)

    print(solution.y[2])
    return solution.y[0], solution.y[2]
    # x = solution.y[0]
    # y = solution.y[2]
    # t = solution.t


traj_symbolic_x, traj_symbolic_y = symbolic_linear(
    initial_velocity, angle_of_projection, time_step)

traj_numeric_x, traj_numeric_y = numeric_linear(
    initial_velocity, angle_of_projection, time_step)

fig, ax = plt.subplots()
ax.set_xlim(0, max(max(traj_symbolic_x), max(traj_numeric_x)) + 1)
ax.set_ylim(0, max(max(traj_symbolic_y), max(traj_numeric_y)) + 1)
ax.set_title('Real-Time Trajectory of a Thrown Stone')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
line1, = ax.plot([], [], lw=2, color='red', label='symbolic')
line2, = ax.plot([], [], lw=2, color='blue', label='numerical')
ax.legend()

constants_text = f"""
constants:
g = {g} м/s^2
k = {k}
m = {m} kg
Θ = {angle_of_projection} deg
v = {initial_velocity} m/s
"""
plt.text(0.05, 0.95, constants_text, transform=ax.transAxes, fontsize=8,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    line1.set_data(traj_symbolic_x[:i], traj_symbolic_y[:i])
    line2.set_data(traj_numeric_x[:i], traj_numeric_y[:i])
    return line1, line2


ani = animation.FuncAnimation(fig, animate,
                              frames=max(len(traj_symbolic_x),
                                         len(traj_numeric_x)) + 1,
                              init_func=init,
                              blit=True,
                              interval=50,
                              repeat=False)

plt.text(
    0.1, 0.1, f"Landing point ({
        max(x for (x, y) in zip(traj_numeric_x, traj_numeric_y)):.2f})")

plt.show()
