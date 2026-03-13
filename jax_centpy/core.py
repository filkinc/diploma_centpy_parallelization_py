from typing import NamedTuple, Callable, Any
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


class Pars1d(NamedTuple):
    x_init: float
    x_final: float
    t_final: float
    dt_out: float
    J: int
    cfl: float
    scheme: str = "sd2"

    @property
    def dx(self):
        return (self.x_final - self.x_init) / self.J


class Equation1d(NamedTuple):
    flux: Callable[[jnp.ndarray], jnp.ndarray]
    spectral_radius: Callable[[jnp.ndarray], jnp.ndarray]
    initial_data: Callable[[jnp.ndarray], jnp.ndarray]
    boundary_handler: Callable[[jnp.ndarray, int], jnp.ndarray]
    name: str = "GenericEquation"
