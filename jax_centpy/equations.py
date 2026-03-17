from core import Equation1d, Equation2d
from boundaries import periodic_bc, neumann_bc, dirichlet_zero_bc, periodic_bc_2d, neumann_bc_2d
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


def make_euler_explosion_2d(gamma: float = 1.4, periodic: bool = True):
    """
    Двумерные уравнения Эйлера для идеального газа.
    U = [rho, rho*u, rho*v, E]
    """

    def _compute_pressure(q):
        rho = q[..., 0]
        u = q[..., 1] / rho
        v = q[..., 2] / rho
        E = q[..., 3]
        # p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
        return (gamma - 1.0) * (E - 0.5 * rho * (u ** 2 + v ** 2))

    def flux_x(q):
        rho = q[..., 0]
        rhou = q[..., 1]
        rhov = q[..., 2]
        E = q[..., 3]

        u = rhou / rho
        p = _compute_pressure(q)

        # F(U) = [rho*u, rho*u^2 + p, rho*u*v, u*(E + p)]
        return jnp.stack([
            rhou,
            rhou * u + p,
            rhou * (rhov / rho),
            u * (E + p)
        ], axis=-1)

    def flux_y(q):
        rho = q[..., 0]
        rhou = q[..., 1]
        rhov = q[..., 2]
        E = q[..., 3]

        v = rhov / rho
        p = _compute_pressure(q)

        # G(U) = [rho*v, rho*u*v, rho*v^2 + p, v*(E + p)]
        return jnp.stack([
            rhov,
            rhov * (rhou / rho),
            rhov * v + p,
            v * (E + p)
        ], axis=-1)

    def spectral_radius_x(q):
        rho = q[..., 0]
        u = q[..., 1] / rho
        p = _compute_pressure(q)
        c = jnp.sqrt(gamma * p / rho)  # скорость звука
        return jnp.abs(u) + c

    def spectral_radius_y(q):
        rho = q[..., 0]
        v = q[..., 2] / rho
        p = _compute_pressure(q)
        c = jnp.sqrt(gamma * p / rho)
        return jnp.abs(v) + c

    def initial_explosion(x, y):
        """Задача 2D взрыва. В центре высокое давление и плотность."""
        r = jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

        # Внутренняя область r < 0.25, внешняя r >= 0.25
        rho = jnp.where(r < 0.25, 1.0, 0.125)
        p = jnp.where(r < 0.25, 1.0, 0.1)
        u = jnp.zeros_like(x)
        v = jnp.zeros_like(x)

        E = p / (gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)

        return jnp.stack([rho, rho * u, rho * v, E], axis=-1)

    return Equation2d(
        flux_x=flux_x,
        flux_y=flux_y,
        spectral_radius_x=spectral_radius_x,
        spectral_radius_y=spectral_radius_y,
        initial_data=initial_explosion,
        boundary_handler=periodic_bc_2d,
        name="Euler 2D (Explosion)"
    )


def make_euler_isentropic_vortex_2d(gamma: float = 1.4):
    """
    Гладкое точное решение: Изоэнтропийный вихрь.
    Идеально для проверки сходимости 2D Эйлера. Периодические граничные условия.
    Домен: x, y в [0, 10]. Центр вихря (5, 5). Скорость потока (1, 1).
    """

    def _compute_pressure(q):
        rho, u, v, E = q[..., 0], q[..., 1] / q[..., 0], q[..., 2] / q[..., 0], q[..., 3]
        return (gamma - 1.0) * (E - 0.5 * rho * (u ** 2 + v ** 2))

    def flux_x(q):
        rho, rhou, rhov, E = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        u = rhou / rho;
        p = _compute_pressure(q)
        return jnp.stack([rhou, rhou * u + p, rhou * (rhov / rho), u * (E + p)], axis=-1)

    def flux_y(q):
        rho, rhou, rhov, E = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        v = rhov / rho;
        p = _compute_pressure(q)
        return jnp.stack([rhov, rhov * (rhou / rho), rhov * v + p, v * (E + p)], axis=-1)

    def spectral_radius_x(q):
        rho, u, p = q[..., 0], q[..., 1] / q[..., 0], _compute_pressure(q)
        return jnp.abs(u) + jnp.sqrt(gamma * p / rho)

    def spectral_radius_y(q):
        rho, v, p = q[..., 0], q[..., 2] / q[..., 0], _compute_pressure(q)
        return jnp.abs(v) + jnp.sqrt(gamma * p / rho)

    # Функция генерирует точное решение для любого момента t
    def exact_solution(x, y, t=0.0):
        beta = 5.0
        # Смещение вихря с учетом скорости потока u=1, v=1 и периодичности
        dx = (x - 5.0 - t) % 10.0
        dy = (y - 5.0 - t) % 10.0
        dx = jnp.where(dx > 5.0, dx - 10.0, dx)
        dy = jnp.where(dy > 5.0, dy - 10.0, dy)

        r2 = dx ** 2 + dy ** 2
        du = - (beta / (2 * jnp.pi)) * jnp.exp(0.5 * (1.0 - r2)) * dy
        dv = (beta / (2 * jnp.pi)) * jnp.exp(0.5 * (1.0 - r2)) * dx

        u, v = 1.0 + du, 1.0 + dv
        T = 1.0 - ((gamma - 1.0) * beta ** 2 / (8 * gamma * jnp.pi ** 2)) * jnp.exp(1.0 - r2)
        rho = T ** (1.0 / (gamma - 1.0))
        p = rho ** gamma
        E = p / (gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
        return jnp.stack([rho, rho * u, rho * v, E], axis=-1)

    return Equation2d(
        flux_x=flux_x, flux_y=flux_y,
        spectral_radius_x=spectral_radius_x, spectral_radius_y=spectral_radius_y,
        initial_data=lambda x, y: exact_solution(x, y, 0.0),
        boundary_handler=periodic_bc_2d,
        name="Euler 2D (Isentropic Vortex)"
    ), exact_solution


def make_euler_riemann_2d(gamma: float = 1.4):
    def _compute_pressure(q):
        rho = q[..., 0]
        u = q[..., 1] / rho
        v = q[..., 2] / rho
        E = q[..., 3]
        return (gamma - 1.0) * (E - 0.5 * rho * (u ** 2 + v ** 2))

    def flux_x(q):
        rho, rhou, rhov, E = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        u = rhou / rho
        p = _compute_pressure(q)
        return jnp.stack([rhou, rhou * u + p, rhou * (rhov / rho), u * (E + p)], axis=-1)

    def flux_y(q):
        rho, rhou, rhov, E = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        v = rhov / rho
        p = _compute_pressure(q)
        return jnp.stack([rhov, rhov * (rhou / rho), rhov * v + p, v * (E + p)], axis=-1)

    def spectral_radius_x(q):
        rho = q[..., 0]
        u = q[..., 1] / rho
        c = jnp.sqrt(gamma * _compute_pressure(q) / rho)
        return jnp.abs(u) + c

    def spectral_radius_y(q):
        rho = q[..., 0]
        v = q[..., 2] / rho
        c = jnp.sqrt(gamma * _compute_pressure(q) / rho)
        return jnp.abs(v) + c

    def initial_riemann(x, y):
        # Воссоздаем 4 квадранта точно как в CPU версии
        rho = jnp.where((x > 0.5) & (y > 0.5), 1.5,
                        jnp.where((x <= 0.5) & (y > 0.5), 0.5323,
                                  jnp.where((x <= 0.5) & (y <= 0.5), 0.138, 0.5323)))  # LR

        vx = jnp.where((x > 0.5) & (y > 0.5), 0.0,
                       jnp.where((x <= 0.5) & (y > 0.5), 1.206,
                                 jnp.where((x <= 0.5) & (y <= 0.5), 1.206, 0.0)))

        vy = jnp.where((x > 0.5) & (y > 0.5), 0.0,
                       jnp.where((x <= 0.5) & (y > 0.5), 0.0,
                                 jnp.where((x <= 0.5) & (y <= 0.5), 1.206, 1.206)))

        p = jnp.where((x > 0.5) & (y > 0.5), 1.5,
                      jnp.where((x <= 0.5) & (y > 0.5), 0.3,
                                jnp.where((x <= 0.5) & (y <= 0.5), 0.029, 0.3)))

        E = p / (gamma - 1.0) + 0.5 * rho * (vx ** 2 + vy ** 2)
        return jnp.stack([rho, rho * vx, rho * vy, E], axis=-1)

    return Equation2d(
        flux_x=flux_x, flux_y=flux_y,
        spectral_radius_x=spectral_radius_x, spectral_radius_y=spectral_radius_y,
        initial_data=initial_riemann,
        boundary_handler=neumann_bc_2d,
        name="Euler 2D (Riemann)"
    )
