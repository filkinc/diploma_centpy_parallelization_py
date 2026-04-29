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
    m = jnp.moveaxis(u, axis, 0)

    face = (7.0 / 12.0) * (m[1:-2] + m[2:-1]) - (1.0 / 12.0) * (m[:-3] + m[3:])

    u_L = face[:-1]
    u_R = face[1:]
    u_c = m[2:-2]

    u_im1 = m[1:-3]
    u_ip1 = m[3:-1]

    u_min = jnp.minimum(u_im1, jnp.minimum(u_c, u_ip1))
    u_max = jnp.maximum(u_im1, jnp.maximum(u_c, u_ip1))

    is_extremum = (u_R - u_c) * (u_c - u_L) <= 0.0
    u_R = jnp.where(is_extremum, u_c, u_R)
    u_L = jnp.where(is_extremum, u_c, u_L)

    u_R = jnp.clip(u_R, u_min, u_max)
    u_L = jnp.clip(u_L, u_min, u_max)

    delta = u_R - u_L
    u_mid = 0.5 * (u_R + u_L)

    cond1 = (delta * (u_c - u_mid)) > (delta * delta / 6.0)
    u_L_c1 = 3.0 * u_c - 2.0 * u_R
    u_L = jnp.where(~is_extremum & cond1, jnp.clip(u_L_c1, u_min, u_max), u_L)

    delta2 = u_R - u_L
    u_mid2 = 0.5 * (u_R + u_L)
    cond2 = (-delta2 * delta2 / 6.0) > (delta2 * (u_c - u_mid2))
    u_R_c2 = 3.0 * u_c - 2.0 * u_L
    u_R = jnp.where(~is_extremum & cond2, jnp.clip(u_R_c2, u_min, u_max), u_R)

    return jnp.moveaxis(u_R, 0, axis), jnp.moveaxis(u_L, 0, axis)


def compute_rhs_sd3(
        t: float,
        u_inner: jnp.ndarray,
        pars: Pars1d,
        eqn: Equation1d,
) -> jnp.ndarray:
    n_ghost = 3
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    u_R_all, u_L_all = reconstruction_sd3(u_padded, axis=0)
    u_minus = u_R_all[:-1]
    u_plus = u_L_all[1:]

    a = jnp.maximum(eqn.spectral_radius(u_minus), eqn.spectral_radius(u_plus))

    flux = 0.5 * (eqn.flux(u_minus) + eqn.flux(u_plus) - a * (u_plus - u_minus))

    return -(flux[1:] - flux[:-1]) / pars.dx


def compute_rhs_sd3_2d(
        t: float,
        u_inner: jnp.ndarray,
        pars: Pars2d,
        eqn: Equation2d,
) -> jnp.ndarray:
    n_ghost = 3
    u_padded = eqn.boundary_handler(u_inner, n_ghost)

    is_system = u_inner.ndim == 3

    u_strip_x = u_padded[:, n_ghost:-n_ghost, ...]
    u_east_x, u_west_x = reconstruction_sd3(u_strip_x, axis=0)

    u_L_x = u_east_x[:-1, ...]
    u_R_x = u_west_x[1:, ...]

    a_x = jnp.maximum(eqn.spectral_radius_x(u_L_x), eqn.spectral_radius_x(u_R_x))
    if is_system:
        a_x = a_x[..., None]

    flux_x = 0.5 * (eqn.flux_x(u_L_x) + eqn.flux_x(u_R_x) - a_x * (u_R_x - u_L_x))
    rhs_x = -(flux_x[1:, ...] - flux_x[:-1, ...]) / pars.dx

    u_strip_y = u_padded[n_ghost:-n_ghost, :, ...]
    u_north_y, u_south_y = reconstruction_sd3(u_strip_y, axis=1)

    u_L_y = u_north_y[:, :-1, ...]
    u_R_y = u_south_y[:, 1:, ...]

    a_y = jnp.maximum(eqn.spectral_radius_y(u_L_y), eqn.spectral_radius_y(u_R_y))
    if is_system:
        a_y = a_y[..., None]

    flux_y = 0.5 * (eqn.flux_y(u_L_y) + eqn.flux_y(u_R_y) - a_y * (u_R_y - u_L_y))
    rhs_y = -(flux_y[:, 1:, ...] - flux_y[:, :-1, ...]) / pars.dy

    return rhs_x + rhs_y


def compute_step_fd2_1d(
        u_inner: jnp.ndarray,
        dt: float,
        pars: Pars1d,
        eqn: Equation1d,
        limiter: LimiterFunc,
        odd: jnp.ndarray,
) -> jnp.ndarray:
    """
    Один шаг схемы Нессияху-Тадмора (NT/FD2) для 1D.

    Полностью дискретная схема 2-го порядка с шахматной сеткой.
    odd=True  → corrector[1:]  (сдвиг сетки на +0.5 ячейки)
    odd=False → corrector[:-1] (сетка возвращена)

    Устойчивость: pars.cfl <= 0.5
    """
    n_ghost = 2
    u = eqn.boundary_handler(u_inner, n_ghost)  # [J+4]

    # Уклоны решения, размер J+2
    sigma = limiter(u[1:-1] - u[:-2], u[2:] - u[1:-1])

    f = eqn.flux(u)  # [J+4]

    # Уклоны потока для предиктора, размер J+2
    f_sigma = limiter(f[1:-1] - f[:-2], f[2:] - f[1:-1])

    # Предиктор: u_half_inner имеет размер J
    u_half_inner = (u[1:-1] - 0.5 * (dt / pars.dx) * f_sigma)[1:-1]
    u_half = eqn.boundary_handler(u_half_inner, n_ghost)  # [J+4]
    f_half = eqn.flux(u_half)  # [J+4]

    # Корректор (одинаков для обеих веток), размер J+1
    corrector = (
            0.5 * (u[2:-1] + u[1:-2])
            + 0.125 * (sigma[:-1] - sigma[1:])
            - (dt / pars.dx) * (f_half[2:-1] - f_half[1:-2])
    )

    # odd=True  → corrector[1:]  (J значений, правый сдвиг)
    # odd=False → corrector[:-1] (J значений, исходные позиции)
    return jnp.where(odd, corrector[1:], corrector[:-1])


def compute_step_fd2_2d(
        u_inner: jnp.ndarray,
        dt: float,
        pars: Pars2d,
        eqn: Equation2d,
        limiter: LimiterFunc,
        odd: jnp.ndarray,
) -> jnp.ndarray:
    """
    Один шаг 2D схемы Нессияху-Тадмора (NT/FD2).

    Полностью дискретная схема 2-го порядка на шахматной сетке.
    Алгоритм:
      1. sigma_x, sigma_y — уклоны решения по x и y (slope-limited)
      2. un_half — среднее по 4 угловым ячейкам + поправка уклонами (Predictor 1)
      3. u_star  — предиктор по потокам: u - 0.5*dt*(df_x/dx + dg_y/dy) (Predictor 2)
      4. Корректор: un_half - 0.5*dt/dx*(avg_x flux) - 0.5*dt/dy*(avg_y flux)

    Сдвиг сетки:
      odd=True  → corrector_rhs[1:, 1:]  (Jx×Jy значений, сдвиг (+0.5, +0.5))
      odd=False → corrector_rhs[:-1, :-1] (Jx×Jy значений, исходные позиции)

    Устойчивость: pars.cfl <= 0.5
    Работает для скалярных уравнений и систем (u_inner.ndim == 3).
    """
    n_ghost = 2
    u = eqn.boundary_handler(u_inner, n_ghost)  # (Jx+4, Jy+4[, nq])

    # ── Уклоны решения по x и y, форма (Jx+2, Jy+2[, nq]) ──
    # sigma_x[i,j] = limiter(u[i,j]-u[i-1,j], u[i+1,j]-u[i,j])
    sigma_x = limiter(
        u[1:-1, 1:-1, ...] - u[:-2, 1:-1, ...],
        u[2:, 1:-1, ...] - u[1:-1, 1:-1, ...]
    )
    # sigma_y[i,j] = limiter(u[i,j]-u[i,j-1], u[i,j+1]-u[i,j])
    sigma_y = limiter(
        u[1:-1, 1:-1, ...] - u[1:-1, :-2, ...],
        u[1:-1, 2:, ...] - u[1:-1, 1:-1, ...]
    )

    # ── Predictor 1: un_half — взвешенное среднее по 4 угловым ячейкам ──
    # Размер (Jx+1, Jy+1[, nq]) = (N-3, M-3)
    #
    # Маппинг: sigma_x[:-1, :-1] = u_prime_x[1:-2, 1:-2] оригинального кода
    #          sigma_x[1:,  :-1] = u_prime_x[2:-1, 1:-2]
    #          sigma_x[:-1, 1:]  = u_prime_x[1:-2, 2:-1]
    #          sigma_x[1:,  1:]  = u_prime_x[2:-1, 2:-1]
    un_half_rhs = 0.25 * (
        (u[1:-2, 1:-2, ...] + u[2:-1, 1:-2, ...] + u[1:-2, 2:-1, ...] + u[2:-1, 2:-1, ...])
        + 0.25 * (
            (sigma_x[:-1, :-1, ...] - sigma_x[1:, :-1, ...])
            + (sigma_x[:-1, 1:, ...] - sigma_x[1:, 1:, ...])
            + (sigma_y[:-1, :-1, ...] - sigma_y[:-1, 1:, ...])
            + (sigma_y[1:, :-1, ...] - sigma_y[1:, 1:, ...])
        )
    )  # (Jx+1, Jy+1[, nq])

    # ── Predictor 2: u_star — предиктор для вычисления потоков ──
    f = eqn.flux_x(u)  # (Jx+4, Jy+4[, nq])
    g = eqn.flux_y(u)  # (Jx+4, Jy+4[, nq])

    # Уклоны потоков по соответствующим направлениям, форма (Jx+2, Jy+2[, nq])
    f_sigma_x = limiter(
        f[1:-1, 1:-1, ...] - f[:-2, 1:-1, ...],
        f[2:, 1:-1, ...] - f[1:-1, 1:-1, ...]
    )
    g_sigma_y = limiter(
        g[1:-1, 1:-1, ...] - g[1:-1, :-2, ...],
        g[1:-1, 2:, ...] - g[1:-1, 1:-1, ...]
    )

    # u_star в точках [1:-1, 1:-1], форма (Jx+2, Jy+2[, nq])
    u_star_all = u[1:-1, 1:-1, ...] - 0.5 * dt * (
        f_sigma_x / pars.dx + g_sigma_y / pars.dy
    )
    # Берём только физические внутренние ячейки и применяем граничные условия
    u_star_inner = u_star_all[1:-1, 1:-1, ...]  # (Jx, Jy[, nq])
    u_star = eqn.boundary_handler(u_star_inner, n_ghost)  # (Jx+4, Jy+4[, nq])

    f_star = eqn.flux_x(u_star)  # (Jx+4, Jy+4[, nq])
    g_star = eqn.flux_y(u_star)  # (Jx+4, Jy+4[, nq])

    # ── Корректор ──
    # Все слагаемые имеют форму (Jx+1, Jy+1[, nq]) = (N-3, M-3)
    #
    # Поток по x: среднее двух горизонтальных граней
    flux_x_term = 0.5 * (dt / pars.dx) * (
        (f_star[2:-1, 1:-2, ...] - f_star[1:-2, 1:-2, ...])
        + (f_star[2:-1, 2:-1, ...] - f_star[1:-2, 2:-1, ...])
    )
    # Поток по y: среднее двух вертикальных граней
    flux_y_term = 0.5 * (dt / pars.dy) * (
        (g_star[1:-2, 2:-1, ...] - g_star[1:-2, 1:-2, ...])
        + (g_star[2:-1, 2:-1, ...] - g_star[2:-1, 1:-2, ...])
    )

    corrector_rhs = un_half_rhs - flux_x_term - flux_y_term  # (Jx+1, Jy+1[, nq])

    # odd=True  → corrector_rhs[1:, 1:]  (сдвиг (+0.5, +0.5): берём правый-верхний угол)
    # odd=False → corrector_rhs[:-1, :-1] (исходные позиции: берём левый-нижний угол)
    return jnp.where(odd, corrector_rhs[1:, 1:, ...], corrector_rhs[:-1, :-1, ...])
