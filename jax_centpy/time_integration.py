import jax
import jax.numpy as jnp
from typing import Callable

# Тип функции правой части: (t, u, **kwargs) -> du/dt
RHS_FUNC = Callable[[float, jnp.ndarray], jnp.ndarray]


def step_euler(t: float, u: jnp.ndarray, dt: float, rhs_fn: RHS_FUNC) -> jnp.ndarray:
    L = rhs_fn(t, u)
    return u + dt * L


def step_ssp_rk2(t: float, u: jnp.ndarray, dt: float, rhs_fn: RHS_FUNC) -> jnp.ndarray:
    L1 = rhs_fn(t, u)
    u_1 = u + dt * L1

    L2 = rhs_fn(t + dt, u_1)

    return 0.5 * u + 0.5 * u_1 + 0.5 * dt * L2


def step_ssp_rk3(t: float, u: jnp.ndarray, dt: float, rhs_fn: RHS_FUNC) -> jnp.ndarray:
    L1 = rhs_fn(t, u)
    u_1 = u + dt * L1

    L2 = rhs_fn(t + dt, u_1)
    u_2 = 0.75 * u + 0.25 * u_1 + 0.25 * dt * L2

    L3 = rhs_fn(t + 0.5 * dt, u_2)
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * u_2 + (2.0 / 3.0) * dt * L3

    return u_new
