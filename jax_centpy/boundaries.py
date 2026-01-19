import jax.numpy as jnp


def periodic_bc(u, n_ghost):
    return jnp.pad(u, (n_ghost, n_ghost), mode='wrap')


def neumann_bc(u, n_ghost):
    # "Свободный выход" или нулевая производная: u[-1] = u[0]
    return jnp.pad(u, (n_ghost, n_ghost), mode='edge')


def dirichlet_zero_bc(u, n_ghost):
    # "Ноль на границах"
    return jnp.pad(u, (n_ghost, n_ghost), mode='constant', constant_values=0.0)
