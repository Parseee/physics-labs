import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

# Читаем параметры
with open("config.json", "r") as f:
    params = json.load(f)
    m1 = params.get("m1", 0.1)
    m2 = params.get("m2", 0.1)
    p = params.get("p", 0.45)
    k_hooke = params.get("ball_hooke", 1e5)
    r = params.get("r", 0.05)
    x1_0 = params.get("x1_0", 0.25)
    y1_0 = params.get("y1_0", 0.15)
    angle_deg = params.get("angle_deg", 10)
    x2_0 = params.get("x2_0", 0.35)
    y2_0 = params.get("y2_0", 0.15)

# Параметры стола
table_x1, table_x2 = 0.0, 0.5
table_y1, table_y2 = 0.0, 0.3

# Начальные скорости
angle = np.deg2rad(angle_deg)
vx1_0, vy1_0 = (p / m1) * np.cos(angle), (p / m1) * np.sin(angle)
vx2_0, vy2_0 = 0.0, 0.0

# ODE система: state = [x1,y1,vx1,vy1,x2,y2,vx2,vy2]


def ode_system(t, state):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

    # Силы от стен
    Fx1, Fy1, Fx2, Fy2 = 0.0, 0.0, 0.0, 0.0
    if x1 - r < table_x1:
        Fx1 += k_hooke * (table_x1 - (x1 - r))
    if x1 + r > table_x2:
        Fx1 -= k_hooke * ((x1 + r) - table_x2)
    if y1 - r < table_y1:
        Fy1 += k_hooke * (table_y1 - (y1 - r))
    if y1 + r > table_y2:
        Fy1 -= k_hooke * ((y1 + r) - table_y2)

    if x2 - r < table_x1:
        Fx2 += k_hooke * (table_x1 - (x2 - r))
    if x2 + r > table_x2:
        Fx2 -= k_hooke * ((x2 + r) - table_x2)
    if y2 - r < table_y1:
        Fy2 += k_hooke * (table_y1 - (y2 - r))
    if y2 + r > table_y2:
        Fy2 -= k_hooke * ((y2 + r) - table_y2)

    # Силы контакта между шарами
    dx = x2 - x1
    dy = y2 - y1
    dist = np.hypot(dx, dy)
    if dist < 2*r:
        overlap = 2*r - dist
        nx, ny = dx / dist, dy / dist
        F = k_hooke * overlap
        Fx1 -= F * nx
        Fy1 -= F * ny
        Fx2 += F * nx
        Fy2 += F * ny

    ax1, ay1 = Fx1 / m1, Fy1 / m1
    ax2, ay2 = Fx2 / m2, Fy2 / m2

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]


# Начальное состояние
state0 = [x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0]

# Интегрирование через solve_ivp
solution = solve_ivp(ode_system, [0, 5.0],
                     state0, max_step=1e-4, dense_output=True)

# Получаем траектории
n_frames = 10000
frame_times = np.linspace(0, solution.t[-1], n_frames)
sol_values = solution.sol(frame_times)

x1s, y1s, vx1s, vy1s, x2s, y2s, vx2s, vy2s = sol_values

E_kin = 0.5*m1*(vx1s**2+vy1s**2) + 0.5*m2*(vx2s**2+vy2s**2)
# потенциальная энергия (упругость стен и контактов)
E_pot = np.zeros_like(frame_times)
for i in range(len(frame_times)):
    e = 0.0
    if x1s[i]-r < table_x1:
        e += 0.5*k_hooke*(table_x1-(x1s[i]-r))**2
    if x1s[i]+r > table_x2:
        e += 0.5*k_hooke*((x1s[i]+r)-table_x2)**2
    if y1s[i]-r < table_y1:
        e += 0.5*k_hooke*(table_y1-(y1s[i]-r))**2
    if y1s[i]+r > table_y2:
        e += 0.5*k_hooke*((y1s[i]+r)-table_y2)**2
    if x2s[i]-r < table_x1:
        e += 0.5*k_hooke*(table_x1-(x2s[i]-r))**2
    if x2s[i]+r > table_x2:
        e += 0.5*k_hooke*((x2s[i]+r)-table_x2)**2
    if y2s[i]-r < table_y1:
        e += 0.5*k_hooke*(table_y1-(y2s[i]-r))**2
    if y2s[i]+r > table_y2:
        e += 0.5*k_hooke*((y2s[i]+r)-table_y2)**2
    dx = x2s[i]-x1s[i]
    dy = y2s[i]-y1s[i]
    dist = np.hypot(dx, dy)
    if dist < 2*r:
        e += 0.5*k_hooke*(2*r-dist)**2
    E_pot[i] = e

E_total = E_kin + E_pot

# Анимация
fig, (ax_table, ax_energy) = plt.subplots(
    2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [2, 1]})
ax_table.set_xlim(table_x1-0.05, table_x2+0.05)
ax_table.set_ylim(table_y1-0.05, table_y2+0.05)
ax_table.set_aspect('equal')
ax_table.set_xticks([])
ax_table.set_yticks([])
ax_table.set_title("Two balls: with ODE")
ax_table.plot([table_x1, table_x2, table_x2, table_x1, table_x1],
              [table_y1, table_y1, table_y2, table_y2, table_y1], 'k', lw=2)

ball1 = Circle((x1s[0], y1s[0]), radius=r, color='tab:blue')
ball2 = Circle((x2s[0], y2s[0]), radius=r, color='tab:orange')
ax_table.add_patch(ball1)
ax_table.add_patch(ball2)

ax_energy.set_xlim(frame_times[0], frame_times[-1])
ax_energy.set_ylim(min(E_total)*0.98, max(E_total)*1.02)
ax_energy.set_xlabel("time [s]")
ax_energy.set_ylabel("total energy [J]")
(line_energy,) = ax_energy.plot([], [], lw=2, color='green')


def init():
    ball1.center = (x1s[0], y1s[0])
    ball2.center = (x2s[0], y2s[0])
    line_energy.set_data([], [])
    return ball1, ball2, line_energy


def update(frame):
    ball1.center = (x1s[frame], y1s[frame])
    ball2.center = (x2s[frame], y2s[frame])
    line_energy.set_data(frame_times[:frame], E_total[:frame])
    return ball1, ball2, line_energy


ani = animation.FuncAnimation(fig, update, frames=n_frames,
                              init_func=init, interval=20,
                              blit=True, repeat=True)
plt.show()
