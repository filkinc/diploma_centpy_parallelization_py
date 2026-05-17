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

def wall_bc_1d(u, n_ghost):
    """
    Условие непротекания (твёрдая стенка) для 1D.
    u: shape (J, num_vars), где вектор состояния [rho, rho*v, E]
    Нормальная скорость (компонента импульса) отражается по знаку.
    Скаляры (rho, E) — копируются.
    """
    # Фиктивные ячейки слева: зеркальное отражение
    left_ghost = u[:n_ghost][::-1]                         # копируем значения
    left_ghost = left_ghost.at[:, 1].set(-left_ghost[:, 1])  # отражаем rho*v

    # Фиктивные ячейки справа
    right_ghost = u[-n_ghost:][::-1]
    right_ghost = right_ghost.at[:, 1].set(-right_ghost[:, 1])

    return jnp.concatenate([left_ghost, u, right_ghost], axis=0)


def wall_bc_2d(u, n_ghost):
    """
    Условие непротекания (твёрдая стенка) для 2D.
    u: shape (Jx, Jy, num_vars), где вектор состояния [rho, rho*vx, rho*vy, E]

    На каждой границе:
    - Скаляры (rho, E) копируются из ближайшей внутренней ячейки
    - Нормальная компонента скорости (импульс) отражается по знаку
    - Тангенциальная компонента скорости копируется

    Возвращает u с добавленными фиктивными ячейками размера n_ghost со всех сторон.
    """
    Jx, Jy, num_vars = u.shape

    # Создаём массив с фиктивными ячейками
    u_padded = jnp.zeros((Jx + 2 * n_ghost, Jy + 2 * n_ghost, num_vars))

    # Копируем внутреннюю область
    u_padded = u_padded.at[n_ghost:Jx + n_ghost, n_ghost:Jy + n_ghost].set(u)

    # === ЛЕВАЯ ГРАНИЦА (x_min): отражаем x-компоненту импульса ===
    # Берём ближайшие внутренние ячейки и отражаем по оси x
    for i in range(n_ghost):
        # Индекс фиктивной ячейки слева
        ghost_idx = n_ghost - 1 - i
        # Индекс соответствующей внутренней ячейки (зеркальное отражение)
        real_idx = n_ghost + i

        # Копируем все переменные
        u_padded = u_padded.at[ghost_idx, n_ghost:Jy + n_ghost].set(
            u_padded[real_idx, n_ghost:Jy + n_ghost]
        )
        # Отражаем rho*vx (компонента 1)
        u_padded = u_padded.at[ghost_idx, n_ghost:Jy + n_ghost, 1].set(
            -u_padded[real_idx, n_ghost:Jy + n_ghost, 1]
        )

    # === ПРАВАЯ ГРАНИЦА (x_max): отражаем x-компоненту импульса ===
    for i in range(n_ghost):
        ghost_idx = Jx + n_ghost + i
        real_idx = Jx + n_ghost - 1 - i

        u_padded = u_padded.at[ghost_idx, n_ghost:Jy + n_ghost].set(
            u_padded[real_idx, n_ghost:Jy + n_ghost]
        )
        u_padded = u_padded.at[ghost_idx, n_ghost:Jy + n_ghost, 1].set(
            -u_padded[real_idx, n_ghost:Jy + n_ghost, 1]
        )

    # === НИЖНЯЯ ГРАНИЦА (y_min): отражаем y-компоненту импульса ===
    for j in range(n_ghost):
        ghost_idx = n_ghost - 1 - j
        real_idx = n_ghost + j

        # Копируем ВСЮ строку (включая уже заполненные углы)
        u_padded = u_padded.at[:, ghost_idx].set(u_padded[:, real_idx])
        # Отражаем rho*vy (компонента 2)
        u_padded = u_padded.at[:, ghost_idx, 2].set(-u_padded[:, real_idx, 2])

    # === ВЕРХНЯЯ ГРАНИЦА (y_max): отражаем y-компоненту импульса ===
    for j in range(n_ghost):
        ghost_idx = Jy + n_ghost + j
        real_idx = Jy + n_ghost - 1 - j

        u_padded = u_padded.at[:, ghost_idx].set(u_padded[:, real_idx])
        u_padded = u_padded.at[:, ghost_idx, 2].set(-u_padded[:, real_idx, 2])

    return u_padded