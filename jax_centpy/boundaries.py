import jax.numpy as jnp


def periodic_bc(u, n_ghost):
    return jnp.pad(u, (n_ghost, n_ghost), mode='wrap')


def neumann_bc(u, n_ghost):
    # "Свободный выход" или нулевая производная: u[-1] = u[0]
    return jnp.pad(u, (n_ghost, n_ghost), mode='edge')


def dirichlet_zero_bc(u, n_ghost):
    # "Ноль на границах"
    return jnp.pad(u, (n_ghost, n_ghost), mode='constant', constant_values=0.0)


def periodic_bc_2d(u, n_ghost):
    """
    Периодические граничные условия для 2D массивов.
    Поддерживает как скалярные (Nx, Ny), так и векторные (Nx, Ny, num_vars) переменные.
    """
    if u.ndim == 2:
        pad_width = ((n_ghost, n_ghost), (n_ghost, n_ghost))
    else:
        # Не добавляем паддинг по оси физических переменных (num_vars)
        pad_width = ((n_ghost, n_ghost), (n_ghost, n_ghost), (0, 0))

    return jnp.pad(u, pad_width, mode='wrap')


def neumann_bc_2d(u, n_ghost):
    """Свободный выход для 2D"""
    if u.ndim == 2:
        pad_width = ((n_ghost, n_ghost), (n_ghost, n_ghost))
    else:
        pad_width = ((n_ghost, n_ghost), (n_ghost, n_ghost), (0, 0))
    return jnp.pad(u, pad_width, mode='edge')
