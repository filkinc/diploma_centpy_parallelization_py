# jax_wave_solver.py

import time
import numpy as onp  # обычный NumPy
import jax
import jax.numpy as jnp
from jax import jit
import matplotlib
import matplotlib.pyplot as plt


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

    def solve_with_history(self, steps=200, every=2, device=None):
        u0, ut0 = self.initial_conditions()
        if device is not None:
            u0 = jax.device_put(u0, device)
            ut0 = jax.device_put(ut0, device)

        u_nm1, u_n = ut0, u0
        dx, dt, c = self.dx, self.dt, self.c

        frames = [u_n]
        for i in range(1, steps + 1):
            u_n, u_nm1 = step(u_n, u_nm1, dx, dt, c)
            if i % every == 0:
                frames.append(u_n)
        return frames

    def animate(self, steps=200, every=2, device=None, interval=50):
        from matplotlib.animation import FuncAnimation

        frames = self.solve_with_history(steps=steps, every=every, device=device)
        frames_np = [onp.asarray(f) for f in frames]

        fig, ax = plt.subplots(figsize=(6, 5))
        img = ax.imshow(
            frames_np[0],
            cmap="viridis",
            origin="lower",
            extent=[0.0, 2.0 * onp.pi, 0.0, 2.0 * onp.pi],
            animated=True,
        )
        plt.colorbar(img, ax=ax)
        ax.set_title("2D Wave Equation (JAX animation)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(frame_idx):
            img.set_data(frames_np[frame_idx])
            ax.set_title(f"2D Wave Equation (step {frame_idx * every})")
            return [img]

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames_np),
            interval=interval,
            blit=True,
        )
        plt.tight_layout()
        plt.show()

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


def numpy_wave_solve(nx=256, ny=256, dx=0.01, dt=0.004, c=1.0, steps=500):
    x = onp.linspace(0.0, 2.0 * onp.pi, nx)
    y = onp.linspace(0.0, 2.0 * onp.pi, ny)
    X, Y = onp.meshgrid(x, y, indexing="ij")
    r = onp.sqrt((X - onp.pi) ** 2 + (Y - onp.pi) ** 2)
    u0 = onp.exp(-50.0 * (r - 0.5) ** 2)
    ut0 = onp.zeros_like(u0)

    u_nm1, u_n = ut0, u0

    def lap_np(u):
        ux = (onp.roll(u, -1, axis=0) - onp.roll(u, 1, axis=0)) / (2 * dx) ** 2
        uy = (onp.roll(u, -1, axis=1) - onp.roll(u, 1, axis=1)) / (2 * dx) ** 2
        return ux + uy

    for _ in range(steps):
        lap_u = lap_np(u_n)
        u_np1 = 2.0 * u_n - u_nm1 + (c * dt) ** 2 * lap_u
        u_nm1, u_n = u_n, u_np1

    return u_n


def benchmark(nx=256, ny=256, steps=500, device=None):
    solver = JAXWave2D(nx=nx, ny=ny)

    if device is None:
        device = jax.devices()[0]  # первый доступный (CPU или GPU)

    # NumPy (всегда на CPU)
    t0 = time.time()
    u_np = numpy_wave_solve(nx=nx, ny=ny, steps=steps)
    t_np = time.time() - t0

    # JAX: первый прогон
    t0 = time.time()
    u_jax = solver.solve(steps=steps, device=device)
    jax.block_until_ready(u_jax)
    t_jax_first = time.time() - t0

    # JAX: второй прогон
    t0 = time.time()
    u_jax2 = solver.solve(steps=steps, device=device)
    jax.block_until_ready(u_jax2)
    t_jax_second = time.time() - t0

    print(f"Device: {device}")
    print(f"NumPy: {t_np:.4f} s")
    print(f"JAX 1st (compile+run): {t_jax_first:.4f} s")
    print(f"JAX 2nd (warm):        {t_jax_second:.4f} s")
    print(f"Speedup NumPy/JAX:     {t_np / t_jax_second:.2f}x")

    return u_np, u_jax2
