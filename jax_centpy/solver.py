
import jax
import jax.numpy as jnp
import time
from typing import Dict, List, Tuple, Callable

from core import Pars1d, Equation1d
import schemes
import limiters
import time_integration


def compute_dt(pars: Pars1d, eqn: Equation1d, u: jnp.ndarray) -> float:
    max_speed = jnp.max(eqn.spectral_radius(u))
    safe_speed = jnp.maximum(max_speed, 1e-6)

    return pars.cfl * pars.dx / safe_speed


class Solver1d:

    def __init__(self, pars: Pars1d, eqn: Equation1d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        self.pars = pars
        self.eqn = eqn

        limiter_map = {
            "minmod": limiters.minmod,
            "superbee": limiters.superbee,
            "mc": limiters.monotonized_central,
            "van_leer": limiters.van_leer
        }
        self.limiter = limiter_map.get(limiter_name, limiters.minmod)

        if scheme_name == "sd2":
            def _rhs(t, u):
                return schemes.compute_rhs_sd2(t, u, self.pars, self.eqn, self.limiter)

            self.rhs_fn = _rhs
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented yet.")

        self.step_fn = time_integration.step_ssp_rk3

        @jax.jit
        def update_step(t, u, dt):
            return self.step_fn(t, u, dt, self.rhs_fn)

        self.update_step_jit = update_step

        self.compute_dt_jit = jax.jit(lambda u: compute_dt(self.pars, self.eqn, u))

    def solve(self) -> Dict[str, jnp.ndarray]:
        dx = self.pars.dx
        x = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)

        u = self.eqn.initial_data(x)
        t = 0.0

        saved_t = [t]
        saved_u = [u]

        next_output_time = self.pars.dt_out

        print(f"Starting simulation: {self.eqn.name}")
        print(f"Grid: {self.pars.J} points, Scheme: SD2/{self.limiter.__name__}")
        start_wall_time = time.time()

        step_count = 0

        while t < self.pars.t_final:
            dt = float(self.compute_dt_jit(u))

            if t + dt > next_output_time:
                dt = next_output_time - t

            if t + dt > self.pars.t_final:
                dt = self.pars.t_final - t

            u = self.update_step_jit(t, u, dt)

            t += dt
            step_count += 1

            if t >= next_output_time - 1e-9:
                saved_u.append(u)
                saved_t.append(t)
                next_output_time += self.pars.dt_out

        end_wall_time = time.time()
        print(f"Simulation finished in {end_wall_time - start_wall_time:.4f}s")
        print(f"Total steps: {step_count}")

        return {
            "x": x,
            "t": jnp.array(saved_t),
            "u_n": jnp.stack(saved_u)
        }


class FastSolver1d(Solver1d):
    """
    Оптимизированная версия солвера для бенчмарков.
    Использует jax.lax.while_loop для выполнения всего цикла на устройстве.
    Не сохраняет промежуточные шаги (только начальное и конечное состояние).
    """

    def __init__(self, pars, eqn, scheme_name="sd2"):
        super().__init__(pars, eqn, scheme_name)

        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0):

        def cond_fun(state):
            t, _, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, step_idx = state

            max_speed = jnp.max(self.eqn.spectral_radius(u))
            safe_speed = jnp.maximum(max_speed, 1e-6)
            dt = self.pars.cfl * self.pars.dx / safe_speed

            dt = jnp.minimum(dt, self.pars.t_final - t)

            u_new = self.step_fn(t, u, dt, self.rhs_fn)

            return t + dt, u_new, step_idx + 1

        init_state = (0.0, u0, 0)
        final_t, final_u, final_steps = jax.lax.while_loop(cond_fun, body_fun, init_state)

        return final_t, final_u, final_steps

    def solve(self):
        dx = self.pars.dx
        x = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)
        u0 = self.eqn.initial_data(x)

        final_t, final_u, steps = self.solve_jit(u0)

        return {
            "x": x,
            "t": jnp.array([0.0, final_t]),
            "u_n": jnp.stack([u0, final_u]),
            "steps": steps
        }
