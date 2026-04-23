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

    u_center = u_moved[1:-1]

    u_east = u_center + 0.5 * slopes
    u_west = u_center - 0.5 * slopes

    return jnp.moveaxis(u_east, 0, axis), jnp.moveaxis(u_west, 0, axis)


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

    is_system = u_inner.ndim == 3

    u_strip_x = u_padded[:, n_ghost:-n_ghost, ...]
    u_east_x, u_west_x = reconstruction_sd2(u_strip_x, limiter, theta, axis=0)

    u_L_x = u_east_x[:-1, ...]
    u_R_x = u_west_x[1:, ...]

    a_x = jnp.maximum(eqn.spectral_radius_x(u_L_x), eqn.spectral_radius_x(u_R_x))

    if is_system:
        a_x = a_x[..., None]

    flux_x = 0.5 * (eqn.flux_x(u_L_x) + eqn.flux_x(u_R_x) - a_x * (u_R_x - u_L_x))
    rhs_x = -(flux_x[1:, ...] - flux_x[:-1, ...]) / pars.dx

    u_strip_y = u_padded[n_ghost:-n_ghost, :, ...]
    u_north_y, u_south_y = reconstruction_sd2(u_strip_y, limiter, theta, axis=1)

    u_L_y = u_north_y[:, :-1, ...]
    u_R_y = u_south_y[:, 1:, ...]

    a_y = jnp.maximum(eqn.spectral_radius_y(u_L_y), eqn.spectral_radius_y(u_R_y))

    if is_system:
        a_y = a_y[..., None]

    flux_y = 0.5 * (eqn.flux_y(u_L_y) + eqn.flux_y(u_R_y) - a_y * (u_R_y - u_L_y))
    rhs_y = -(flux_y[:, 1:, ...] - flux_y[:, :-1, ...]) / pars.dy

    return rhs_x + rhs_y


def reconstruction_sd3(u: jnp.ndarray, axis: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Кусочно-параболическая реконструкция третьего порядка (PPM).
    Использует 5-точечный шаблон: требует минимум 2 ghost-ячейки с каждой стороны.

    Для ячейки i квадратичный полином подбирается по условию сохранения среднего
    на трёх ячейках (i-1, i, i+1) и согласования с соседними средними.
    Формула для граничных значений:
        u_{i+1/2} = (7/12)*(u_i + u_{i+1}) - (1/12)*(u_{i-1} + u_{i+2})
    Затем применяется ограничитель монотонности PPM.

    Возвращает (u_east, u_west) — правое и левое граничное значение каждой ячейки.
    """
    m = jnp.moveaxis(u, axis, 0)  # shape: [N, ...]

    # ── Высокоточная интерполяция на грани i+1/2 (4-й порядок без ограничителя) ──
    # Требует индексы [1:-2] для получения граней между [1:-3] и [2:-2]
    # face[k] = значение на грани между m[k] и m[k+1], k in [1, N-3]
    face = (7.0 / 12.0) * (m[1:-2] + m[2:-1]) \
           - (1.0 / 12.0) * (m[:-3] + m[3:])
    # face имеет длину N-3; face[k] — грань справа от ячейки k+1 (0-based в m)
    # Для ячеек m[1:-2] (индексы 1..N-3 в исходном массиве):
    #   u_east[i] = face[i]       (правая грань ячейки i)
    #   u_west[i] = face[i-1]     (левая грань ячейки i)
    # face[k] соответствует грани между m[k+1] и m[k+2]

    u_east_raw = face[1:]  # правая грань ячеек m[2:-2]
    u_west_raw = face[:-1]  # левая  грань ячеек m[2:-2]
    u_center = m[2:-2]

    # ── Ограничитель монотонности PPM (Colella & Woodward 1984) ──
    # Шаг 1: если ячейка является локальным экстремумом — сделать реконструкцию постоянной
    is_extremum = (u_east_raw - u_center) * (u_center - u_west_raw) <= 0.0
    u_east = jnp.where(is_extremum, u_center, u_east_raw)
    u_west = jnp.where(is_extremum, u_center, u_west_raw)

    # Шаг 2: ограничить выброс — граничное значение не должно выходить за пределы соседей
    u_min = jnp.minimum(jnp.minimum(m[1:-3], m[2:-2]), m[3:-1])
    u_max = jnp.maximum(jnp.maximum(m[1:-3], m[2:-2]), m[3:-1])
    u_east = jnp.clip(u_east, u_min, u_max)
    u_west = jnp.clip(u_west, u_min, u_max)

    # Шаг 3: PPM-условие — устранение нефизичных параболических выбросов
    # Если (u_east - u_west) * 6*(u_center - 0.5*(u_east+u_west)) > (u_east - u_west)^2
    # то «придавить» один из концов
    delta = u_east - u_west
    avg = 0.5 * (u_east + u_west)
    overshoot = 6.0 * (u_center - avg)
    # Случай 1: левый конец выбивается
    cond1 = delta * (u_center - avg - delta / 3.0) < -(delta ** 2) / 6.0
    u_west = jnp.where(cond1, 3.0 * u_center - 2.0 * u_east, u_west)
    # Случай 2: правый конец выбивается
    cond2 = delta * (u_center - avg + delta / 3.0) > (delta ** 2) / 6.0
    u_east = jnp.where(cond2, 3.0 * u_center - 2.0 * u_west, u_east)

    return jnp.moveaxis(u_east, 0, axis), jnp.moveaxis(u_west, 0, axis)


def compute_rhs_sd3(
        t: float,
        u_inner: jnp.ndarray,
        pars: Pars1d,
        eqn: Equation1d,
) -> jnp.ndarray:
    """
    Правая часть для схемы SD3 (1D).
    Ghost-зон нужно 3 (reconstruction_sd3 «съедает» по 2 с каждой стороны,
    плюс 1 для разностей потоков).
    """
    n_ghost = 3
    u_padded = eqn.boundary_handler(u_inner, n_ghost)  # [J + 2*n_ghost, ...]

    u_R_all, u_L_all = reconstruction_sd3(u_padded, axis=0)
    # u_R_all, u_L_all имеют длину J + 2*n_ghost - 4 = J + 2
    # Нам нужны J+1 граней (между J ячейками), поэтому берём [:-1] и [1:]
    u_minus = u_R_all[:-1]  # левое значение на грани
    u_plus = u_L_all[1:]  # правое значение на грани

    a = jnp.maximum(eqn.spectral_radius(u_minus), eqn.spectral_radius(u_plus))

    flux = 0.5 * (eqn.flux(u_minus) + eqn.flux(u_plus) - a * (u_plus - u_minus))

    return -(flux[1:] - flux[:-1]) / pars.dx


def compute_rhs_sd3_2d(
        t: float,
        u_inner: jnp.ndarray,
        pars: Pars2d,
        eqn: Equation2d,
) -> jnp.ndarray:
    """
    Правая часть для схемы SD3 (2D), dimension-by-dimension splitting.
    Структура полностью аналогична compute_rhs_sd2_2d.
    """
    n_ghost = 3
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    is_system = u_inner.ndim == 3

    # ── Направление X ──
    u_strip_x = u_padded[:, n_ghost:-n_ghost, ...]  # убираем ghost по Y
    u_east_x, u_west_x = reconstruction_sd3(u_strip_x, axis=0)

    u_L_x = u_east_x[:-1, ...]
    u_R_x = u_west_x[1:, ...]

    a_x = jnp.maximum(eqn.spectral_radius_x(u_L_x), eqn.spectral_radius_x(u_R_x))
    if is_system:
        a_x = a_x[..., None]

    flux_x = 0.5 * (eqn.flux_x(u_L_x) + eqn.flux_x(u_R_x) - a_x * (u_R_x - u_L_x))
    rhs_x = -(flux_x[1:, ...] - flux_x[:-1, ...]) / pars.dx

    # ── Направление Y ──
    u_strip_y = u_padded[n_ghost:-n_ghost, :, ...]  # убираем ghost по X
    u_north_y, u_south_y = reconstruction_sd3(u_strip_y, axis=1)

    u_L_y = u_north_y[:, :-1, ...]
    u_R_y = u_south_y[:, 1:, ...]

    a_y = jnp.maximum(eqn.spectral_radius_y(u_L_y), eqn.spectral_radius_y(u_R_y))
    if is_system:
        a_y = a_y[..., None]

    flux_y = 0.5 * (eqn.flux_y(u_L_y) + eqn.flux_y(u_R_y) - a_y * (u_R_y - u_L_y))
    rhs_y = -(flux_y[:, 1:, ...] - flux_y[:, :-1, ...]) / pars.dy

    return rhs_x + rhs_y
