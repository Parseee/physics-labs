import numpy as np

from config import (
    CATHODE_TEMPERATURE_K,
    DT,
    E_CHARGE,
    E_MASS,
    GAP_LENGTH,
    GRID_POSITION,
    GRID_TRANSPARENCY,
    INJECT_MACRO_PER_STEP,
    N_TEST_PARTICLES,
    N_TRAJECTORIES_TO_STORE,
    T_END,
    THERMAL_SIGMA_FACTOR,
    emission_current,
)
from fields import build_triode_fixed_nodes, electric_field_from_potential, make_grid, solve_potential_1d, vacuum_field_solution

Q_ELECTRON = -E_CHARGE


def _sample_emission_speeds(n, temp_k=CATHODE_TEMPERATURE_K):
    sigma = THERMAL_SIGMA_FACTOR * np.sqrt(1.5 * 1.380649e-23 * temp_k / E_MASS)
    return np.abs(np.random.normal(0.0, sigma, size=n))


def _interp_field(x_grid, e_grid, pos):
    return np.interp(pos, x_grid, e_grid, left=e_grid[0], right=e_grid[-1])


def _grid_pass_mask(velocities, v_grid, v_cathode):
    barrier_ev = max(v_cathode - v_grid, 0.0)
    barrier_j = E_CHARGE * barrier_ev
    kinetic_j = 0.5 * E_MASS * velocities**2
    energetic_pass = kinetic_j >= barrier_j
    mesh_pass = np.random.random(velocities.size) <= GRID_TRANSPARENCY
    return energetic_pass & mesh_pass


def run_non_interacting(
    x_grid,
    e_grid,
    n_particles=N_TEST_PARTICLES,
    dt=DT,
    t_end=T_END,
    grid_enabled=False,
    v_grid=0.0,
    v_cathode=0.0,
):
    n_steps = int(np.ceil(t_end / dt))
    t = np.arange(n_steps) * dt

    x_hist = np.full((N_TRAJECTORIES_TO_STORE, n_steps), np.nan)
    v_hist = np.full((N_TRAJECTORIES_TO_STORE, n_steps), np.nan)

    pos = np.full(n_particles, 1e-9)
    vel = _sample_emission_speeds(n_particles)
    alive = np.ones(n_particles, dtype=bool)

    reached_anode = np.zeros(n_particles, dtype=bool)
    reached_cathode = np.zeros(n_particles, dtype=bool)
    reached_grid = np.zeros(n_particles, dtype=bool)
    anode_speeds = np.zeros(n_particles, dtype=float)

    n_store = min(N_TRAJECTORIES_TO_STORE, n_particles)
    x_hist[:n_store, 0] = pos[:n_store]
    v_hist[:n_store, 0] = vel[:n_store]

    for k in range(1, n_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        prev_pos = pos[idx].copy()
        e_loc = _interp_field(x_grid, e_grid, pos[idx])
        vel[idx] += (Q_ELECTRON / E_MASS) * e_loc * dt
        pos[idx] += vel[idx] * dt

        if grid_enabled:
            crossed = (prev_pos < GRID_POSITION) & (pos[idx] >= GRID_POSITION)
            if crossed.any():
                crossed_idx = idx[crossed]
                passed = _grid_pass_mask(np.abs(vel[crossed_idx]), v_grid=v_grid, v_cathode=v_cathode)
                blocked_idx = crossed_idx[~passed]
                if blocked_idx.size > 0:
                    reached_grid[blocked_idx] = True
                    alive[blocked_idx] = False
                    pos[blocked_idx] = GRID_POSITION
                    vel[blocked_idx] = 0.0

        hit_anode = pos[idx] >= GAP_LENGTH
        hit_cathode = pos[idx] <= 0.0
        reached_anode[idx[hit_anode]] = True
        reached_cathode[idx[hit_cathode]] = True
        anode_speeds[idx[hit_anode]] = np.abs(vel[idx[hit_anode]])
        alive[idx[hit_anode | hit_cathode]] = False

        x_hist[:n_store, k] = pos[:n_store]
        v_hist[:n_store, k] = vel[:n_store]

    return {
        "t": t,
        "x": x_hist,
        "v": v_hist,
        "reached_anode": reached_anode,
        "reached_cathode": reached_cathode,
        "reached_grid": reached_grid,
        "anode_speeds": anode_speeds,
    }


def estimate_current_non_interacting(v_anode, v_cathode, v_grid, grid_enabled):
    xg, phi, e = vacuum_field_solution(
        v_anode=v_anode,
        v_cathode=v_cathode,
        v_grid=v_grid,
        grid_enabled=grid_enabled,
    )
    traj = run_non_interacting(
        xg,
        e,
        grid_enabled=grid_enabled,
        v_grid=v_grid,
        v_cathode=v_cathode,
    )

    emit_i = emission_current()
    eff = float(np.mean(traj["reached_anode"]))
    current = emit_i * eff
    anode_speeds = traj["anode_speeds"][traj["reached_anode"]]
    mean_speed = float(np.mean(anode_speeds)) if anode_speeds.size else 0.0

    result = {
        "current_a": current,
        "collection_efficiency": eff,
        "mean_anode_speed": mean_speed,
    }
    return result, (xg, phi, e), traj


def _deposit_charge_1d(x_grid, pos, q_macro):
    rho = np.zeros_like(x_grid)
    dx = x_grid[1] - x_grid[0]
    valid = pos[(pos >= 0.0) & (pos <= x_grid[-1])]
    if valid.size == 0:
        return rho

    left_idx = np.floor(valid / dx).astype(int)
    left_idx = np.clip(left_idx, 0, x_grid.size - 2)
    frac = (valid - x_grid[left_idx]) / dx
    np.add.at(rho, left_idx, q_macro * (1.0 - frac) / dx)
    np.add.at(rho, left_idx + 1, q_macro * frac / dx)
    return rho


def run_space_charge_pic(
    v_anode,
    v_cathode,
    v_grid,
    grid_enabled,
    dt=DT,
    t_end=T_END,
    inject_macro_per_step=INJECT_MACRO_PER_STEP,
):
    x_grid, dx = make_grid()
    fixed_nodes = build_triode_fixed_nodes(
        x_grid,
        v_grid=v_grid,
        grid_enabled=grid_enabled,
        v_cathode=v_cathode,
        v_anode=v_anode,
    )

    n_steps = int(np.ceil(t_end / dt))
    emit_i = emission_current()
    q_macro = -emit_i * dt / max(1, inject_macro_per_step)

    pos = np.empty(0, dtype=float)
    vel = np.empty(0, dtype=float)
    current_trace = np.zeros(n_steps, dtype=float)

    phi = np.linspace(v_cathode, v_anode, x_grid.size)
    e_grid = electric_field_from_potential(phi, dx)

    for k in range(n_steps):
        pos = np.concatenate([pos, np.full(inject_macro_per_step, 1e-9)])
        vel = np.concatenate([vel, _sample_emission_speeds(inject_macro_per_step)])

        rho = _deposit_charge_1d(x_grid, pos, q_macro=q_macro)
        phi = solve_potential_1d(
            x=x_grid,
            v_cathode=v_cathode,
            v_anode=v_anode,
            rho=rho,
            fixed_nodes=fixed_nodes,
        )
        e_grid = electric_field_from_potential(phi, dx)

        if pos.size > 0:
            prev_pos = pos.copy()
            e_loc = _interp_field(x_grid, e_grid, pos)
            vel += (Q_ELECTRON / E_MASS) * e_loc * dt
            pos += vel * dt

            if grid_enabled:
                crossed = (prev_pos < GRID_POSITION) & (pos >= GRID_POSITION)
                if crossed.any():
                    crossed_idx = np.where(crossed)[0]
                    passed = _grid_pass_mask(np.abs(vel[crossed_idx]), v_grid=v_grid, v_cathode=v_cathode)
                    blocked_idx = crossed_idx[~passed]
                    if blocked_idx.size > 0:
                        pos[blocked_idx] = GRID_POSITION
                        vel[blocked_idx] = 0.0

        hit_anode = pos >= GAP_LENGTH
        current_trace[k] = (-q_macro * np.count_nonzero(hit_anode)) / dt

        alive = (pos > 0.0) & (pos < GAP_LENGTH)
        if grid_enabled:
            alive &= (pos != GRID_POSITION)
        pos = pos[alive]
        vel = vel[alive]

    steady = current_trace[n_steps // 2 :]
    return {
        "current_a": float(np.mean(steady)),
        "x": x_grid,
        "phi": phi,
        "e": e_grid,
        "current_trace": current_trace,
    }
