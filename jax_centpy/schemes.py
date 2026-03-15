import jax.numpy as jnp
from typing import Callable, Tuple
from core import Equation1d, Pars1d, Equation2d, Pars2d
from limiters import minmod

LimiterFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def reconstruction_sd2(u: jnp.ndarray, limiter: LimiterFunc = minmod, theta: float = 1.0, axis: int = 0) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    u_moved = jnp.moveaxis(u, axis, 0)

    diff_plus = u_moved[2:] - u_moved[1:-1]
    diff_minus = u_moved[1:-1] - u_moved[:-2]
    slopes = limiter(theta * diff_minus, theta * diff_plus)

    u_center = u[1:-1]
    u_east = u_center + 0.5 * slopes
    u_west = u_center - 0.5 * slopes
    return u_east, u_west


def compute_rhs_sd2(t: float, u_inner: jnp.ndarray, pars: Pars1d, eqn: Equation1d, limiter: LimiterFunc = minmod,
                    theta: float = 1.0) -> jnp.ndarray:
    n_ghost = 2
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    u_R_all, u_L_all = reconstruction_sd2(u_padded, limiter=limiter, theta=theta, axis=0)

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


def compute_rhs_sd2_2d(t: float, u_inner: jnp.ndarray, pars: Pars2d, eqn: Equation2d, limiter: LimiterFunc = minmod,
                       theta: float = 1.0) -> jnp.ndarray:
    n_ghost = 2
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    u_strip_x = u_padded[:, n_ghost:-n_ghost, ...]
    u_east_x, u_west_x = reconstruction_sd2(u_strip_x, limiter, theta, axis=0)

    u_L_x = u_east_x[:-1, ...]
    u_R_x = u_west_x[1:, ...]

    a_x = jnp.maximum(eqn.spectral_radius_x(u_L_x), eqn.spectral_radius_x(u_R_x))
    flux_x = 0.5 * (eqn.flux_x(u_L_x) + eqn.flux_x(u_R_x) - a_x * (u_R_x - u_L_x))
    rhs_x = -(flux_x[1:, ...] - flux_x[:-1, ...]) / pars.dx

    u_strip_y = u_padded[n_ghost:-n_ghost, :, ...]
    u_north_y, u_south_y = reconstruction_sd2(u_strip_y, limiter, theta, axis=1)

    u_L_y = u_north_y[:, :-1, ...]
    u_R_y = u_south_y[:, 1:, ...]

    a_y = jnp.maximum(eqn.spectral_radius_y(u_L_y), eqn.spectral_radius_y(u_R_y))
    flux_y = 0.5 * (eqn.flux_y(u_L_y) + eqn.flux_y(u_R_y) - a_y * (u_R_y - u_L_y))
    rhs_y = -(flux_y[:, 1:, ...] - flux_y[:, :-1, ...]) / pars.dy

    return rhs_x + rhs_y
