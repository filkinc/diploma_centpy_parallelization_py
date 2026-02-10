from core import Equation1d
from boundaries import periodic_bc, neumann_bc, dirichlet_zero_bc
import jax.numpy as jnp


def make_burgers_1d(periodic=True):
    return Equation1d(
        flux=lambda u: 0.5 * u**2,
        spectral_radius=lambda u: jnp.abs(u),
        initial_data=lambda x: jnp.sin(x) + 0.5 * jnp.sin(0.5 * x),
        boundary_handler=periodic_bc if periodic else neumann_bc,
        name="Burgers 1D"
    )


def make_linear_convection_1d(a=1.0, periodic=True):
    """du/dt + a * du/dx = 0"""
    return Equation1d(
        flux=lambda u: a * u,
        spectral_radius=lambda u: jnp.full_like(u, jnp.abs(a)),
        initial_data=lambda x: jnp.exp(-100 * (x - 0.5)**2), # Гауссиан
        boundary_handler=periodic_bc if periodic else dirichlet_zero_bc,
        name=f"Linear Convection (a={a})"
    )


def make_linear_advection_1d(velocity: float = 1.0):
    """
    Уравнение линейного переноса: u_t + v * u_x = 0
    Flux: f(u) = v * u
    Spectral radius: |v|
    """
    return Equation1d(
        flux=lambda u: velocity * u,
        spectral_radius=lambda u: jnp.full_like(u, jnp.abs(velocity)),
        initial_data=lambda x: jnp.sin(x),
        boundary_handler=periodic_bc,
        name=f"Linear Advection (v={velocity})"
    )
