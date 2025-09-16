import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.81  # free fall acceleration
k = 0.1  # resistance coefficient
m = 0.100   # mass of stone


def simulate_thrown_stone(v0, theta, dt=0.01):
    # Convert angle to radians
    theta_rad = np.radians(theta)

    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)

    gamma = k / m

    x, y = 0, 0

    trajectory_x = []
    trajectory_y = []

    t = 0

    while y >= 0:
        # Update positions
        trajectory_x.append(x)
        trajectory_y.append(y)

        x += (vx / gamma) * (1 - np.exp(-gamma * t))
        y += (1 / gamma) * (vy + g / gamma) * \
            (1 - np.exp(-gamma * t)) - (g * t) / gamma

        t += dt

    trajectory_x.append(x)  # add last bit of calculation
    trajectory_y.append(y)  # add last bit of calculation

    return trajectory_x, trajectory_y


def derivatives(state):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    ax = - (k / m) * v * vx
    ay = -g - (k / m) * v * vy
    return np.array([vx, vy, ax, ay])


def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + dt/2 * k1)
    k3 = derivatives(state + dt/2 * k2)
    k4 = derivatives(state + dt * k3)
    return state + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Parameters
initial_velocity = 20.0  # Initial velocity in m/s
angle_of_projection = 45  # Angle in degrees
time_step = 0.05  # Time step in seconds

# Run the simulation to get trajectory points
trajectory_x, trajectory_y = simulate_thrown_stone(
    initial_velocity, angle_of_projection, time_step)

# Set up the figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(0, max(trajectory_x) + 1)
ax.set_ylim(0, max(trajectory_y) + 1)
ax.set_title('Real-Time Trajectory of a Thrown Stone')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
line, = ax.plot([], [], lw=2)

# Initialization function for the animation


def init():
    line.set_data([], [])
    return line,

# Animation function


def animate(i):
    line.set_data(trajectory_x[:i], trajectory_y[:i])
    return line,


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(
    trajectory_x) + 1, init_func=init, blit=True, interval=50, repeat=False)

# Show the animation
plt.show()
