import time
import numpy as onp  # обычный NumPy
import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation


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
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(k):
            img.set_data(frames_np[k])
            ax.set_title(f"Step {k * every}")
            return [img]

        anim = FuncAnimation(
            fig, update, frames=len(frames_np), interval=interval, blit=True
        )
        plt.close(fig)  # не показывать автоматически
        return anim

    def plot(self, u, title="2D Wave Equation Solution (JAX)"):
        plt.figure(figsize=(8, 6))
        plt.imshow(
            jnp.asarray(u),
            cmap="viridis",
            origin="lower",
            extent=[0.0, 2.0 * jnp.pi, 0.0, 2.0 * jnp.pi],
        )
        plt.colorbar()
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


@jit(static_argnums=1)
def solve_one_wave(params_batch, steps):
    nx, ny = params_batch.shape
    dx = 2 * jnp.pi / (nx - 1)
    dt = 0.004
    c = 1.0

    u0 = params_batch
    ut0 = jnp.zeros_like(u0)

    def one_step(carry, _):
        u_nm1, u_n = carry
        u_np1, u_new_nm1 = step(u_n, u_nm1, dx, dt, c)
        return (u_np1, u_n), None

    final_carry, _ = lax.scan(one_step, (ut0, u0), None, length=steps)
    u_final, _ = final_carry
    return u_final


solve_batch = jax.vmap(solve_one_wave, in_axes=(0, None))


def benchmark_vmap(nx=128, ny=128, batch_size=8, steps=200, device=None):
    if device is None:
        device = jax.devices()[0]

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Grid: {nx}x{ny}, Steps: {steps}")

    # Генерируем batch_size разных начальных условий
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    centers_x = jnp.linspace(0.5, 1.5, batch_size)
    centers_y = jnp.linspace(0.5, 1.5, batch_size)

    u0_batch = jnp.stack([
        jnp.exp(-50.0 * ((X - cx) ** 2 + (Y - cy) ** 2))
        for cx, cy in zip(centers_x, centers_y)
    ])

    if device is not None:
        u0_batch = jax.device_put(u0_batch, device)

    t0 = time.time()
    results_loop = []
    for i in range(batch_size):
        result = solve_one_wave(u0_batch[i], steps)
        results_loop.append(result)
    results_loop = jnp.stack(results_loop)
    jax.block_until_ready(results_loop)
    t_loop = time.time() - t0

    t0 = time.time()
    results_vmap = solve_batch(u0_batch, steps)
    jax.block_until_ready(results_vmap)
    t_vmap = time.time() - t0

    print(f"Python loop:  {t_loop:.4f} s")
    print(f"vmap batch:   {t_vmap:.4f} s")
    print(f"Speedup:      {t_loop / t_vmap:.2f}x")

    return results_loop, results_vmap