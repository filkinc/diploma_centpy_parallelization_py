import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Callable, Any

from .core import Pars1d, Equation1d, Pars2d, Equation2d
from . import schemes
from . import limiters
from . import time_integration


def compute_dt(pars: Pars1d, eqn: Equation1d, u: jnp.ndarray) -> float:
    max_speed = jnp.max(eqn.spectral_radius(u))
    safe_speed = jnp.maximum(max_speed, 1e-6)

    return pars.cfl * pars.dx / safe_speed


class Solver1d:
    """
    Оптимизированный 1D солвер с сохранением всех временных слоёв.
    Использует jax.lax.while_loop + динамическое обновление массивов.
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
            self.step_fn = time_integration.step_ssp_rk2
            self.is_fd2 = False

        elif scheme_name == "fd2":
            # FD2: полностью дискретная схема Нессияху–Тадмора
            self.rhs_fn = None
            self.step_fn = None
            self.is_fd2 = True

            # JIT‑обёртка для шага FD2 (как в Solver1d, но здесь без наследования)
            self.fd2_step_jit = jax.jit(
                lambda u, dt, odd: schemes.compute_step_fd2_1d(
                    u, dt, self.pars, self.eqn, self.limiter, odd
                )
            )

        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented yet.")

        # Вычисляем максимальное количество snapshots
        self.max_snapshots = int(jnp.ceil(self.pars.t_final / self.pars.dt_out)) + 2

        # JIT-компиляция
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0: jnp.ndarray):
        """
        Внутренний цикл с сохранением временных слоёв через динамическое обновление.

        Возвращает:
            saved_times: массив времён (фиксированной длины)
            saved_states: массив состояний (фиксированной длины)
            actual_count: реальное количество сохранённых snapshots
            total_steps: количество временных шагов
        """
        # Предаллоцируем массивы фиксированного размера
        saved_states = jnp.zeros((self.max_snapshots,) + u0.shape)
        saved_times = jnp.zeros(self.max_snapshots)

        # Сохраняем начальное состояние (индекс 0)
        saved_states = saved_states.at[0].set(u0)
        saved_times = saved_times.at[0].set(0.0)

        def cond_fun(state):
            t, _, _, _, _, _, _, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, next_output_time, snapshot_idx, step_count, saved_states, saved_times, odd = state

            # Вычисление адаптивного шага
            max_speed = jnp.max(self.eqn.spectral_radius(u))
            safe_speed = jnp.maximum(max_speed, 1e-6)
            dt = self.pars.cfl * self.pars.dx / safe_speed

            # Корректируем dt
            dt = jnp.minimum(dt, next_output_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            # Один шаг интегрирования
            if self.is_fd2:
                u_new = self.fd2_step_jit(u, dt, odd)
                odd_new = jnp.logical_not(odd)
            else:
                u_new = self.step_fn(t, u, dt, self.rhs_fn)
                odd_new = odd

            t_new = t + dt

            # Проверяем, нужно ли сохранять
            should_save = t_new >= next_output_time - 1e-9
            next_idx = snapshot_idx + 1

            def save_snapshot(args):
                ss, st, idx = args
                ss_new = ss.at[idx].set(u_new)
                st_new = st.at[idx].set(t_new)
                return ss_new, st_new

            def no_save(args):
                ss, st, _ = args
                return ss, st

            saved_states_new, saved_times_new = jax.lax.cond(
                should_save,
                save_snapshot,
                no_save,
                (saved_states, saved_times, next_idx),
            )

            snapshot_idx_new = jnp.where(should_save, next_idx, snapshot_idx)
            next_output_time_new = jnp.where(
                should_save,
                next_output_time + self.pars.dt_out,
                next_output_time,
            )

            return (
                t_new,
                u_new,
                next_output_time_new,
                snapshot_idx_new,
                step_count + 1,
                saved_states_new,
                saved_times_new,
                odd_new,
            )

        init_state = (
            0.0,  # t
            u0,  # u
            self.pars.dt_out,  # next_output_time
            0,  # snapshot_idx
            0,  # step_count
            saved_states,
            saved_times,
            jnp.array(False),  # odd
        )

        (final_t, final_u, _, final_snapshot_idx,
         total_steps, saved_states_final, saved_times_final, final_odd) = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        # Возвращаем всё + количество реально сохранённых snapshots
        actual_count = final_snapshot_idx + 1

        return saved_times_final, saved_states_final, actual_count, total_steps, final_odd

    def solve(self) -> Dict[str, Any]:
        """
        Запускает решение 1D задачи.

        Возвращает словарь с:
            - x: координатная сетка
            - t: массив времён snapshots
            - u_n: массив состояний (num_snapshots, J)
            - steps: количество временных шагов
        """
        dx = self.pars.dx
        x = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)
        u0 = self.eqn.initial_data(x)

        saved_times, saved_states, actual_count, total_steps, final_odd = self.solve_jit(u0)

        # Обрезаем массивы НА HOST-СТОРОНЕ (после JIT)
        actual_count_int = int(actual_count)
        saved_times_trimmed = saved_times[:actual_count_int]
        saved_states_trimmed = saved_states[:actual_count_int]

        # FD2: компенсация шахматной сетки
        if self.is_fd2 and bool(final_odd):
            x_output = x - dx / 2
            print(f"[FD2] Финальный шаг на шахматной сетке, x сдвинут на +dx/2")
        else:
            x_output = x

        return {
            "x": x_output,
            "t": saved_times_trimmed,
            "u_n": saved_states_trimmed,
            "steps": total_steps
        }


def compute_dt_2d(pars: Pars2d, eqn: Equation2d, u: jnp.ndarray) -> float:
    max_speed_x = jnp.max(eqn.spectral_radius_x(u))
    max_speed_y = jnp.max(eqn.spectral_radius_y(u))

    safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
    safe_speed_y = jnp.maximum(max_speed_y, 1e-6)

    return pars.cfl / (safe_speed_x / pars.dx + safe_speed_y / pars.dy)


class Solver2d:
    """
    Оптимизированный 2D солвер с сохранением всех временных слоёв.
    Использует jax.lax.while_loop + динамическое обновление массивов.
    """

    def __init__(self, pars: Pars2d, eqn: Equation2d, scheme_name: str = "sd2", limiter_name: str = "minmod"):
        self.pars = pars
        self.eqn = eqn

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
            self.is_fd2 = False

        elif scheme_name == "fd2":
            # FD2 (Нессияху–Тадмор) 2D — полностью дискретная схема.
            # Не использует rhs_fn/step_fn.
            self.rhs_fn = None
            self.step_fn = None
            self.is_fd2 = True

            # JIT‑обёртка для шага FD2 2D
            self.fd2_step_jit = jax.jit(
                lambda u, dt, odd: schemes.compute_step_fd2_2d(
                    u, dt, self.pars, self.eqn, self.limiter, odd
                )
            )

        else:
            raise NotImplementedError(f"Scheme {scheme_name} not implemented for 2D yet.")

        # Вычисляем максимальное количество snapshots
        self.max_snapshots = int(jnp.ceil(self.pars.t_final / self.pars.dt_out)) + 2

        # JIT-компиляция
        self.solve_jit = jax.jit(self._solve_internal)

    def _solve_internal(self, u0: jnp.ndarray):
        """
        Внутренний цикл с сохранением временных слоёв через динамическое обновление.
        """
        # Предаллоцируем массивы фиксированного размера
        saved_states = jnp.zeros((self.max_snapshots,) + u0.shape)
        saved_times = jnp.zeros(self.max_snapshots)

        # Сохраняем начальное состояние (индекс 0)
        saved_states = saved_states.at[0].set(u0)
        saved_times = saved_times.at[0].set(0.0)

        def cond_fun(state):
            t, _, _, _, _, _, _, _ = state
            return t < self.pars.t_final

        def body_fun(state):
            t, u, next_output_time, snapshot_idx, step_count, saved_states, saved_times, odd = state

            # Вычисление адаптивного шага
            max_speed_x = jnp.max(self.eqn.spectral_radius_x(u))
            max_speed_y = jnp.max(self.eqn.spectral_radius_y(u))
            safe_speed_x = jnp.maximum(max_speed_x, 1e-6)
            safe_speed_y = jnp.maximum(max_speed_y, 1e-6)

            dt = self.pars.cfl / (safe_speed_x / self.pars.dx + safe_speed_y / self.pars.dy)
            dt = jnp.minimum(dt, next_output_time - t)
            dt = jnp.minimum(dt, self.pars.t_final - t)

            # Один шаг интегрирования
            if self.is_fd2:
                u_new = self.fd2_step_jit(u, dt, odd)
                odd_new = jnp.logical_not(odd)
            else:
                u_new = self.step_fn(t, u, dt, self.rhs_fn)
                odd_new = odd

            t_new = t + dt

            # Проверяем, нужно ли сохранять
            should_save = t_new >= next_output_time - 1e-9

            # Индекс для следующего сохранения
            next_idx = snapshot_idx + 1

            # ИСПОЛЬЗУЕМ lax.cond вместо jnp.where для условного обновления массивов
            def save_snapshot(args):
                ss, st, idx = args
                # Используем динамическое обновление через .at[idx]
                ss_new = ss.at[idx].set(u_new)
                st_new = st.at[idx].set(t_new)
                return ss_new, st_new

            def no_save(args):
                ss, st, _ = args
                return ss, st

            saved_states_new, saved_times_new = jax.lax.cond(
                should_save,
                save_snapshot,
                no_save,
                (saved_states, saved_times, next_idx)
            )

            # Обновляем остальные переменные
            snapshot_idx_new = jnp.where(should_save, next_idx, snapshot_idx)
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
                step_count + 1,
                saved_states_new,
                saved_times_new,
                odd_new,
            )

        init_state = (
            0.0,  # t
            u0,  # u
            self.pars.dt_out,  # next_output_time
            0,  # snapshot_idx
            0,  # step_count
            saved_states,
            saved_times,
            jnp.array(False),  # odd
        )

        (final_t, final_u, _, final_snapshot_idx,
         total_steps, saved_states_final, saved_times_final, _) = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )

        # Возвращаем всё + количество реально сохранённых snapshots
        actual_count = final_snapshot_idx + 1

        return saved_times_final, saved_states_final, actual_count, total_steps

    def solve(self) -> Dict[str, Any]:
        """Запускает решение 2D задачи."""
        # Генерация 2D сетки
        x_1d = jnp.linspace(self.pars.x_init + self.pars.dx / 2,
                            self.pars.x_final - self.pars.dx / 2, self.pars.Jx)
        y_1d = jnp.linspace(self.pars.y_init + self.pars.dy / 2,
                            self.pars.y_final - self.pars.dy / 2, self.pars.Jy)
        X, Y = jnp.meshgrid(x_1d, y_1d, indexing='ij')

        u0 = self.eqn.initial_data(X, Y)

        saved_times, saved_states, actual_count, total_steps = self.solve_jit(u0)

        # Обрезаем массивы НА HOST-СТОРОНЕ (после JIT)
        actual_count_int = int(actual_count)
        saved_times_trimmed = saved_times[:actual_count_int]
        saved_states_trimmed = saved_states[:actual_count_int]

        return {
            "X": X,
            "Y": Y,
            "t": saved_times_trimmed,
            "u": saved_states_trimmed,
            "steps": total_steps
        }
