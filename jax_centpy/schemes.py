
import jax.numpy as jnp
from typing import Callable, Tuple
from core import Equation1d, Pars1d
from limiters import minmod

LimiterFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def reconstruction_sd2(u: jnp.ndarray, limiter: LimiterFunc = minmod, theta: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:

    diff_plus = u[2:] - u[1:-1]
    diff_minus = u[1:-1] - u[:-2]
    slopes = limiter(theta * diff_minus, theta * diff_plus)

    u_center = u[1:-1]
    u_east = u_center + 0.5 * slopes
    u_west = u_center - 0.5 * slopes
    return u_east, u_west


def compute_rhs_sd2(t: float, u_inner: jnp.ndarray, pars: Pars1d, eqn: Equation1d, limiter: LimiterFunc = minmod) -> jnp.ndarray:

    n_ghost = 2
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    u_R_all, u_L_all = reconstruction_sd2(u_padded, limiter=limiter)

    u_minus = u_R_all[:-1]
    u_plus = u_L_all[1:]

    a_minus = eqn.spectral_radius(u_minus)
    a_plus = eqn.spectral_radius(u_plus)
    a = jnp.maximum(a_minus, a_plus)

    f_minus = eqn.flux(u_minus)
    f_plus = eqn.flux(u_plus)
    flux = 0.5 * (f_minus + f_plus - a * (u_plus - u_minus))

    flux_diff = flux[1:] - flux[:-1]
    rhs = -flux_diff / pars.dx

    return rhs
