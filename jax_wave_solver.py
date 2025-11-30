# jax_wave_solver.py

import jax
import jax.numpy as jnp
from jax import jit
import matplotlib
matplotlib.use("TkAgg")  # или "Qt5Agg"
import matplotlib.pyplot as plt

# ---- ЧИСТЫЕ ФУНКЦИИ БЕЗ КЛАССОВ (JAX-дружественные) ----

@jit
def laplacian(u, dx):
    ux = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx) ** 2
    uy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dx) ** 2
    return ux + uy

@jit
def step(u_n, u_nm1, dx, dt, c):
    lap_u = laplacian(u_n, dx)
    u_np1 = 2.0 * u_n - u_nm1 + (c * dt) ** 2 * lap_u
    return u_np1, u_n

# ---- КЛАСС-ОБЁРТКА ДЛЯ УДОБСТВА ----

class JAXWave2D:
    def __init__(self, nx=256, ny=256, dx=0.01, dt=0.004, c=1.0):
        self.nx, self.ny = nx, ny
        self.dx, self.dt, self.c = dx, dt, c

        x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)
        y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

    def initial_conditions(self):
        r = jnp.sqrt((self.X - jnp.pi) ** 2 + (self.Y - jnp.pi) ** 2)
        u0 = jnp.exp(-50.0 * (r - 0.5) ** 2)
        return u0, jnp.zeros_like(u0)

    def solve(self, steps=500, device=None):
        u0, ut0 = self.initial_conditions()

        if device is not None:
            u0 = jax.device_put(u0, device)
            ut0 = jax.device_put(ut0, device)

        u_nm1, u_n = ut0, u0
        dx, dt, c = self.dx, self.dt, self.c

        for _ in range(steps):
            u_n, u_nm1 = step(u_n, u_nm1, dx, dt, c)

        return u_n

    def plot(self, u):
        plt.figure(figsize=(8, 6))
        plt.imshow(
            jnp.asarray(u),
            cmap="viridis",
            origin="lower",
            extent=[0.0, 2.0 * jnp.pi, 0.0, 2.0 * jnp.pi],
        )
        plt.colorbar()
        plt.title("2D Wave Equation Solution (JAX)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
