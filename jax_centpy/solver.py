
import jax
import jax.numpy as jnp
import time
from typing import Dict, List, Tuple, Callable, Any

from core import Pars1d, Equation1d, Pars2d, Equation2d
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
        elif scheme_name == "sd3":
            def _rhs(t, u):
                return schemes.compute_rhs_sd3(t, u, self.pars, self.eqn)

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


def compute_dt_2d(pars: Pars2d, eqn: Equation2d, u: jnp.ndarray) -> float:
    max_speed_x = jnp.max(eqn.spectral_radius_x(u))
    max_speed_y = jnp.max(eqn.spectral_radius_y(u))

    safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
    safe_speed_y = jnp.maximum(max_speed_y, 1e-6)

    return pars.cfl / (safe_speed_x / pars.dx + safe_speed_y / pars.dy)


class Solver2d:
    def __init__(self, pars: Pars2d, eqn: Equation2d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        self.pars = pars
        self.eqn = eqn
        self.scheme_name = scheme_name

        limiter_map = {
            "minmod": limiters.minmod,
            "superbee": limiters.superbee,
            "mc": limiters.monotonized_central,
            "van_leer": limiters.van_leer,
            "average": limiters.average
        }
        self.limiter = limiter_map.get(limiter_name, limiters.minmod)

        if scheme_name == "sd2":
            def _rhs(t, u):
                return schemes.compute_rhs_sd2_2d(t, u, self.pars, self.eqn, self.limiter)

            self.rhs_fn = _rhs
        elif scheme_name == "sd3":
            def _rhs(t, u):
                return schemes.compute_rhs_sd3_2d(t, u, self.pars, self.eqn)

            self.rhs_fn = _rhs
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented for 2D yet.")

        self.step_fn = time_integration.step_ssp_rk2

        @jax.jit
        def update_step(t, u, dt):
            return self.step_fn(t, u, dt, self.rhs_fn)

        self.update_step_jit = update_step
        self.compute_dt_jit = jax.jit(lambda u: compute_dt_2d(self.pars, self.eqn, u))

    def solve(self) -> Dict[str, Any]:
        # Генерация 2D сетки
        x_1d = jnp.linspace(self.pars.x_init + self.pars.dx / 2, self.pars.x_final - self.pars.dx / 2, self.pars.Jx)
        y_1d = jnp.linspace(self.pars.y_init + self.pars.dy / 2, self.pars.y_final - self.pars.dy / 2, self.pars.Jy)
        X, Y = jnp.meshgrid(x_1d, y_1d, indexing='ij')

        u = self.eqn.initial_data(X, Y)
        t = 0.0

        saved_t = [t]
        saved_u = [u]
        next_output_time = self.pars.dt_out

        print(f"Starting 2D simulation: {self.eqn.name}")
        print(f"Grid: {self.pars.Jx}x{self.pars.Jy}, Scheme: {self.scheme_name}/{self.limiter.__name__}")
        #start_wall_time = time.time()

        while t < self.pars.t_final:
            dt = float(self.compute_dt_jit(u))

            if t + dt > next_output_time:
                dt = next_output_time - t

            if t + dt > self.pars.t_final:
                dt = self.pars.t_final - t

            u = self.update_step_jit(t, u, dt)
            t += dt

            if t >= next_output_time - 1e-9:
                saved_t.append(t)
                saved_u.append(u)
                next_output_time += self.pars.dt_out

                #print(f"t = {t:.4f} / {self.pars.t_final:.4f}")

        #end_wall_time = time.time()
        #print(f"Net lead time: {end_wall_time - start_wall_time:.2f} s")

        return {"t": jnp.array(saved_t), "u": jnp.stack(saved_u), "X": X, "Y": Y}


class FastSolver2d(Solver2d):
    """
    Оптимизированная JAX-версия (через lax.while_loop).
    Идеально для бенчмарков.
    """

    def __init__(self, pars: Pars2d, eqn: Equation2d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        super().__init__(pars, eqn, scheme_name, limiter_name)
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0):
        def cond_fun(state):
            t, _, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, step_idx = state

            max_speed_x = jnp.max(self.eqn.spectral_radius_x(u))
            max_speed_y = jnp.max(self.eqn.spectral_radius_y(u))
            safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
            safe_speed_y = jnp.maximum(max_speed_y, 1e-6)

            dt = self.pars.cfl / (safe_speed_x / self.pars.dx + safe_speed_y / self.pars.dy)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            u_new = self.step_fn(t, u, dt, self.rhs_fn)
            return t + dt, u_new, step_idx + 1

        init_state = (0.0, u0, 0)
        final_t, final_u, total_steps = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_u, total_steps
