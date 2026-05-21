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
            self.step_fn = time_integration.step_ssp_rk2
        elif scheme_name == "sd3":
            def _rhs(t, u):
                return schemes.compute_rhs_sd3(t, u, self.pars, self.eqn)

            self.rhs_fn = _rhs
            self.step_fn = time_integration.step_ssp_rk3
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented yet.")

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


class FastSolverWithAllLayers1d(Solver1d):
    """
    Оптимизированная версия солвера.
    Использует jax.lax.while_loop, весь цикл выполняется на устройстве.
    Сохраняет временные слои через интервал dt_out (как Solver1d),
    записывая их в предаллоцированный буфер.
    """

    def __init__(self, pars, eqn, scheme_name="sd2", limiter_name="minmod"):
        super().__init__(pars, eqn, scheme_name, limiter_name)
        # Число выходных слоёв (не считая t=0)
        self.n_out = int(round(pars.t_final / pars.dt_out))
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0):
        n_out = self.n_out
        J = self.pars.J

        # Предаллоцируем буферы: +1 для начального состояния
        saved_u = jnp.zeros((n_out + 1, J))
        saved_t = jnp.zeros(n_out + 1)

        saved_u = saved_u.at[0].set(u0)
        saved_t = saved_t.at[0].set(0.0)

        # state: (t, u, step_idx, out_idx, saved_t, saved_u)
        init_state = (0.0, u0, 0, 1, saved_t, saved_u)

        def cond_fun(state):
            t, u, step_idx, out_idx, saved_t, saved_u = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, step_idx, out_idx, saved_t, saved_u = state

            # CFL шаг
            max_speed = jnp.max(self.eqn.spectral_radius(u))
            safe_speed = jnp.maximum(max_speed, 1e-6)
            dt = self.pars.cfl * self.pars.dx / safe_speed

            # Следующее время вывода
            next_out_time = out_idx * self.pars.dt_out

            # Подрезаем dt, чтобы попасть точно на out и на t_final
            dt = jnp.minimum(dt, next_out_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            u_new = self.step_fn(t, u, dt, self.rhs_fn)
            t_new = t + dt

            # Сохраняем, если достигли времени вывода
            on_output = t_new >= next_out_time - 1e-9
            saved_u = jax.lax.cond(
                on_output,
                lambda s: s.at[out_idx].set(u_new),
                lambda s: s,
                saved_u
            )
            saved_t = jax.lax.cond(
                on_output,
                lambda s: s.at[out_idx].set(t_new),
                lambda s: s,
                saved_t
            )
            out_idx_new = jax.lax.cond(
                on_output,
                lambda idx: idx + 1,
                lambda idx: idx,
                out_idx
            )

            return t_new, u_new, step_idx + 1, out_idx_new, saved_t, saved_u

        final_t, final_u, total_steps, _, saved_t, saved_u = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        return saved_t, saved_u, total_steps

    def solve(self):
        dx = self.pars.dx
        x = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)
        u0 = self.eqn.initial_data(x)

        saved_t, saved_u, steps = self.solve_jit(u0)

        return {
            "x": x,
            "t": saved_t,
            "u_n": saved_u,
            "steps": steps
        }


class FastSolverWithAllLayersWithoutExtends1d:
    """
    Оптимизированный 1D солвер с сохранением всех временных слоёв.
    Использует jax.lax.while_loop для выполнения на устройстве.
    Сохраняет snapshots через интервалы dt_out.
    НЕТ наследования от Solver1d.
    """

    def __init__(self, pars: Pars1d, eqn: Equation1d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        self.pars = pars
        self.eqn = eqn

        # Настройка лимитера
        limiter_map = {
            "minmod": limiters.minmod,
            "superbee": limiters.superbee,
            "mc": limiters.monotonized_central,
            "van_leer": limiters.van_leer
        }
        self.limiter = limiter_map.get(limiter_name, limiters.minmod)

        # Настройка схемы
        if scheme_name == "sd2":
            def _rhs(t, u):
                return schemes.compute_rhs_sd2(t, u, self.pars, self.eqn, self.limiter)

            self.rhs_fn = _rhs
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented yet.")

        self.step_fn = time_integration.step_ssp_rk3

        # Вычисляем количество snapshots заранее
        self.num_snapshots = int(jnp.ceil(self.pars.t_final / self.pars.dt_out)) + 1

        # JIT-компиляция всего цикла
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """
        Внутренний цикл с сохранением временных слоёв.

        Возвращает:
            saved_times: массив времён snapshots
            saved_states: массив состояний в эти моменты
            total_steps: количество временных шагов
        """
        # Предаллоцируем массивы для сохранения
        # saved_states[i] — состояние в момент saved_times[i]
        saved_states = jnp.zeros((self.num_snapshots,) + u0.shape)
        saved_times = jnp.zeros(self.num_snapshots)

        # Сохраняем начальное состояние
        saved_states = saved_states.at[0].set(u0)
        saved_times = saved_times.at[0].set(0.0)

        def cond_fun(state):
            t, _, _, snapshot_idx, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, next_output_time, snapshot_idx, step_count = state

            # Вычисление адаптивного шага
            max_speed = jnp.max(self.eqn.spectral_radius(u))
            safe_speed = jnp.maximum(max_speed, 1e-6)
            dt = self.pars.cfl * self.pars.dx / safe_speed

            # Корректируем dt для попадания в next_output_time
            dt = jnp.minimum(dt, next_output_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            # Один шаг интегрирования
            u_new = self.step_fn(t, u, dt, self.rhs_fn)
            t_new = t + dt

            # Проверяем, достигли ли момента сохранения
            should_save = t_new >= next_output_time - 1e-9

            # Сохраняем snapshot если нужно
            snapshot_idx_new = jnp.where(should_save, snapshot_idx + 1, snapshot_idx)
            saved_states_upd = jnp.where(
                should_save,
                saved_states.at[snapshot_idx_new].set(u_new),
                saved_states
            )
            saved_times_upd = jnp.where(
                should_save,
                saved_times.at[snapshot_idx_new].set(t_new),
                saved_times
            )
            next_output_time_new = jnp.where(
                should_save,
                next_output_time + self.pars.dt_out,
                next_output_time
            )

            return (
                t_new,
                u_new,
                next_output_time_new,
                snapshot_idx_new,
                step_count + 1
            )

        init_state = (
            0.0,  # t
            u0,  # u
            self.pars.dt_out,  # next_output_time
            0,  # snapshot_idx (начинаем с 0, т.к. u0 уже сохранён)
            0  # step_count
        )

        final_t, final_u, _, final_snapshot_idx, total_steps = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        # Обрезаем массивы до реально использованного размера
        # +1 потому что snapshot_idx указывает на последний записанный индекс
        actual_snapshots = final_snapshot_idx + 1
        saved_states_trimmed = saved_states[:actual_snapshots]
        saved_times_trimmed = saved_times[:actual_snapshots]

        return saved_times_trimmed, saved_states_trimmed, total_steps

    def solve(self) -> Dict[str, Any]:
        """
        Запускает решение задачи.

        Возвращает словарь с:
            - x: координатная сетка
            - t: массив времён snapshots
            - u_n: массив состояний (num_snapshots, J)
            - steps: количество временных шагов
        """
        dx = self.pars.dx
        x = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)
        u0 = self.eqn.initial_data(x)

        saved_times, saved_states, total_steps = self.solve_jit(u0)

        return {
            "x": x,
            "t": saved_times,
            "u_n": saved_states,
            "steps": total_steps
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
            self.step_fn = time_integration.step_ssp_rk2
        elif scheme_name == "sd3":
            def _rhs(t, u):
                return schemes.compute_rhs_sd3_2d(t, u, self.pars, self.eqn)

            self.rhs_fn = _rhs
            self.step_fn = time_integration.step_ssp_rk3
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented for 2D yet.")

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

        return {
            "t": jnp.array(saved_t),
            "u": jnp.stack(saved_u),
            "X": X,
            "Y": Y
        }


class FastSolver2d(Solver2d):
    """
    Оптимизированная JAX-версия (через lax.while_loop).
    Для бенчмарков.
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


class FastSolverWithAllLayers2d(Solver2d):
    """
    Оптимизированная JAX-версия (через lax.while_loop).
    Сохраняет временные слои через интервал dt_out (как Solver2d),
    записывая их в предаллоцированный буфер.
    """

    def __init__(self, pars: Pars2d, eqn: Equation2d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        super().__init__(pars, eqn, scheme_name, limiter_name)
        self.n_out = int(round(pars.t_final / pars.dt_out))
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0):
        n_out = self.n_out
        Jx, Jy = self.pars.Jx, self.pars.Jy
        # u0 может быть shape (Jx, Jy) или (n_comp, Jx, Jy) — берём форму динамически
        u_shape = u0.shape

        saved_u = jnp.zeros((n_out + 1,) + u_shape)
        saved_t = jnp.zeros(n_out + 1)

        saved_u = saved_u.at[0].set(u0)
        saved_t = saved_t.at[0].set(0.0)

        init_state = (0.0, u0, 0, 1, saved_t, saved_u)

        def cond_fun(state):
            t, u, step_idx, out_idx, saved_t, saved_u = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, step_idx, out_idx, saved_t, saved_u = state

            max_speed_x = jnp.max(self.eqn.spectral_radius_x(u))
            max_speed_y = jnp.max(self.eqn.spectral_radius_y(u))
            safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
            safe_speed_y = jnp.maximum(max_speed_y, 1e-6)
            dt = self.pars.cfl / (safe_speed_x / self.pars.dx + safe_speed_y / self.pars.dy)

            next_out_time = out_idx * self.pars.dt_out
            dt = jnp.minimum(dt, next_out_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            u_new = self.step_fn(t, u, dt, self.rhs_fn)
            t_new = t + dt

            on_output = t_new >= next_out_time - 1e-9
            saved_u = jax.lax.cond(
                on_output,
                lambda s: s.at[out_idx].set(u_new),
                lambda s: s,
                saved_u
            )
            saved_t = jax.lax.cond(
                on_output,
                lambda s: s.at[out_idx].set(t_new),
                lambda s: s,
                saved_t
            )
            out_idx_new = jax.lax.cond(
                on_output,
                lambda idx: idx + 1,
                lambda idx: idx,
                out_idx
            )

            return t_new, u_new, step_idx + 1, out_idx_new, saved_t, saved_u

        final_t, final_u, total_steps, _, saved_t, saved_u = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        return saved_t, saved_u, total_steps

    def solve(self):
        x_1d = jnp.linspace(self.pars.x_init + self.pars.dx / 2, self.pars.x_final - self.pars.dx / 2, self.pars.Jx)
        y_1d = jnp.linspace(self.pars.y_init + self.pars.dy / 2, self.pars.y_final - self.pars.dy / 2, self.pars.Jy)
        X, Y = jnp.meshgrid(x_1d, y_1d, indexing='ij')

        u0 = self.eqn.initial_data(X, Y)
        saved_t, saved_u, total_steps = self.solve_jit(u0)

        return {
            "t": saved_t,
            "u": saved_u,
            "X": X,
            "Y": Y,
            "steps": total_steps
        }


class FastSolverWithAllLayersWithoutExtends2d:
    """
    Оптимизированный 2D солвер с сохранением всех временных слоёв.
    Использует jax.lax.while_loop для выполнения на устройстве.
    Сохраняет snapshots через интервалы dt_out.
    НЕТ наследования от Solver2d.
    """

    def __init__(self, pars: Pars2d, eqn: Equation2d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        self.pars = pars
        self.eqn = eqn

        # Настройка лимитера
        limiter_map = {
            "minmod": limiters.minmod,
            "superbee": limiters.superbee,
            "mc": limiters.monotonized_central,
            "van_leer": limiters.van_leer,
            "average": limiters.average
        }
        self.limiter = limiter_map.get(limiter_name, limiters.minmod)

        # Настройка схемы
        if scheme_name == "sd2":
            def _rhs(t, u):
                return schemes.compute_rhs_sd2_2d(t, u, self.pars, self.eqn, self.limiter)

            self.rhs_fn = _rhs
        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented for 2D yet.")

        self.step_fn = time_integration.step_ssp_rk2

        # Вычисляем количество snapshots
        self.num_snapshots = int(jnp.ceil(self.pars.t_final / self.pars.dt_out)) + 1

        # JIT-компиляция всего цикла
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """
        Внутренний цикл с сохранением временных слоёв.

        Возвращает:
            saved_times: массив времён snapshots
            saved_states: массив состояний в эти моменты
            total_steps: количество временных шагов
        """
        # Предаллоцируем массивы для сохранения
        saved_states = jnp.zeros((self.num_snapshots,) + u0.shape)
        saved_times = jnp.zeros(self.num_snapshots)

        # Сохраняем начальное состояние
        saved_states = saved_states.at[0].set(u0)
        saved_times = saved_times.at[0].set(0.0)

        def cond_fun(state):
            t, _, _, snapshot_idx, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, next_output_time, snapshot_idx, step_count = state

            # Вычисление адаптивного шага
            max_speed_x = jnp.max(self.eqn.spectral_radius_x(u))
            max_speed_y = jnp.max(self.eqn.spectral_radius_y(u))
            safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
            safe_speed_y = jnp.maximum(max_speed_y, 1e-6)

            dt = self.pars.cfl / (safe_speed_x / self.pars.dx + safe_speed_y / self.pars.dy)

            # Корректируем dt
            dt = jnp.minimum(dt, next_output_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            # Один шаг интегрирования
            u_new = self.step_fn(t, u, dt, self.rhs_fn)
            t_new = t + dt

            # Проверяем, достигли ли момента сохранения
            should_save = t_new >= next_output_time - 1e-9

            # Сохраняем snapshot если нужно
            snapshot_idx_new = jnp.where(should_save, snapshot_idx + 1, snapshot_idx)
            saved_states_upd = jnp.where(
                should_save,
                saved_states.at[snapshot_idx_new].set(u_new),
                saved_states
            )
            saved_times_upd = jnp.where(
                should_save,
                saved_times.at[snapshot_idx_new].set(t_new),
                saved_times
            )
            next_output_time_new = jnp.where(
                should_save,
                next_output_time + self.pars.dt_out,
                next_output_time
            )

            return (
                t_new,
                u_new,
                next_output_time_new,
                snapshot_idx_new,
                step_count + 1
            )

        init_state = (
            0.0,  # t
            u0,  # u
            self.pars.dt_out,  # next_output_time
            0,  # snapshot_idx
            0  # step_count
        )

        final_t, final_u, _, final_snapshot_idx, total_steps = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        # Обрезаем до реального размера
        actual_snapshots = final_snapshot_idx + 1
        saved_states_trimmed = saved_states[:actual_snapshots]
        saved_times_trimmed = saved_times[:actual_snapshots]

        return saved_times_trimmed, saved_states_trimmed, total_steps

    def solve(self) -> Dict[str, Any]:
        """
        Запускает решение 2D задачи.

        Возвращает словарь с:
            - X, Y: координатные сетки
            - t: массив времён snapshots
            - u: массив состояний (num_snapshots, Jx, Jy, num_vars)
            - steps: количество временных шагов
        """
        # Генерация 2D сетки
        x_1d = jnp.linspace(self.pars.x_init + self.pars.dx / 2, self.pars.x_final - self.pars.dx / 2, self.pars.Jx)
        y_1d = jnp.linspace(self.pars.y_init + self.pars.dy / 2, self.pars.y_final - self.pars.dy / 2, self.pars.Jy)
        X, Y = jnp.meshgrid(x_1d, y_1d, indexing='ij')

        u0 = self.eqn.initial_data(X, Y)

        saved_times, saved_states, total_steps = self.solve_jit(u0)

        return {
            "X": X,
            "Y": Y,
            "t": saved_times,
            "u": saved_states,
            "steps": total_steps
        }