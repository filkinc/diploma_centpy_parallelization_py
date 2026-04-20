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


def dirichlet_riemann_bc_2d(u, n_ghost):
    """
    Фиксированные граничные условия Дирихле для 2D Задачи Римана.
    Замораживает границы значениями из 4-х начальных квадрантов.
    """
    Nx, Ny = u.shape[0], u.shape[1]
    mid_x = Nx // 2
    mid_y = Ny // 2

    pad_width = ((n_ghost, n_ghost), (n_ghost, n_ghost), (0, 0))
    u_padded = jnp.pad(u, pad_width, mode='constant')

    # Задаем физические константы для 4-х квадрантов
    # Вектор состояния q = [rho, rho*vx, rho*vy, E]
    gamma = 1.4

    # Upper Right (UR): x > 0.5, y > 0.5
    ur = jnp.array([1.5, 0.0, 0.0, 1.5 / (gamma - 1.0)])

    # Upper Left (UL): x <= 0.5, y > 0.5
    ul = jnp.array([0.5323, 0.5323 * 1.206, 0.0, 0.3 / (gamma - 1.0) + 0.5 * 0.5323 * (1.206 ** 2)])

    # Lower Left (LL): x <= 0.5, y <= 0.5
    ll = jnp.array(
        [0.138, 0.138 * 1.206, 0.138 * 1.206, 0.029 / (gamma - 1.0) + 0.5 * 0.138 * (1.206 ** 2 + 1.206 ** 2)])

    # Lower Right (LR): x > 0.5, y <= 0.5
    lr = jnp.array([0.5323, 0.0, 0.5323 * 1.206, 0.3 / (gamma - 1.0) + 0.5 * 0.5323 * (1.206 ** 2)])

    # 3. Заполняем фиктивные ячейки (ghost cells) жесткими значениями

    # Левая граница (включая углы)
    u_padded = u_padded.at[:n_ghost, :mid_y + n_ghost].set(ll)  # Нижняя часть левой границы
    u_padded = u_padded.at[:n_ghost, mid_y + n_ghost:].set(ul)  # Верхняя часть левой границы

    # Правая граница (включая углы)
    u_padded = u_padded.at[Nx + n_ghost:, :mid_y + n_ghost].set(lr)  # Нижняя часть правой границы
    u_padded = u_padded.at[Nx + n_ghost:, mid_y + n_ghost:].set(ur)  # Верхняя часть правой границы

    # Нижняя граница (центр, без углов)
    u_padded = u_padded.at[n_ghost: n_ghost + mid_x, :n_ghost].set(ll)
    u_padded = u_padded.at[n_ghost + mid_x: n_ghost + Nx, :n_ghost].set(lr)

    # Верхняя граница (центр, без углов)
    u_padded = u_padded.at[n_ghost: n_ghost + mid_x, Ny + n_ghost:].set(ul)
    u_padded = u_padded.at[n_ghost + mid_x: n_ghost + Nx, Ny + n_ghost:].set(ur)

    return u_padded
