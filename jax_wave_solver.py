# jax_wave_solver.py

import time
import numpy as onp  # обычный NumPy
import jax
import jax.numpy as jnp
from jax import jit
import matplotlib
matplotlib.use("TkAgg")  # или "Qt5Agg"
import matplotlib.pyplot as plt

"""
    Аннотация jit работает по следующему принципу:
    При первой компиляции  она создает\строит вычислительный граф (схему или шаблон) операций, 
    (где вершины это операции (операции из JAX), а ребра это данные)
    который оптимизируется (фьюжн, автодополнение векторизации, раскладка памяти и т.д.) компилятором XLA.
    Этот граф затем реализуется как один кусок скомпилированного кода на CPU или GPU, 
    и работает для любых входных данных с той же формой и dtypes (важно: если форма изменится, 
    JAX сделает ещё одну компиляцию под новый случай)
    Именно поэтому тяжеловесные методы laplacian и step вынесены отдельно, 
    чтобы оставить только примитивы JAX и обернуть в jit, для разовой компиляции и создания граф, который затем будет 
    использоваться на каждом временном слое для других числовых данных такого же типа.
    
    Идея построения вычислительного графа:
    Вычислительный граф строится из примитивных операций JAX (jnp.add, jnp.mul, jnp.exp, jnp.roll, матричное умножение, свёртки)
    Когда мы вызываем метод с аннотацией jit, JAX один раз прогоняет этот метод на специальных абстрактных массивах.
    Во время этого прогона все что реализовано через примитивы JAX записывается в вычислительный граф,
    а все остальное (что не относится к JAX отбрасывается при трассировке, либо делает функцию не‑jittable) 
    не попадает в граф и не оптимизируется. (строится внутреннее представление jaxpr, 
    а на основе jaxpr уже строится XLA‑граф и компилируется в машинный код)
    
    Вместо множества маленьких Python‑вызовов NumPy‑функций мы описываем вычисление 
    как одну большую функцию на jax.numpy‑массивах
"""

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

def benchmark(nx=256, ny=256, steps=500):
    solver = JAXWave2D(nx=nx, ny=ny)
    device = jax.devices("cpu")[0]

    # NumPy
    t0 = time.time()
    u_np = numpy_wave_solve(nx=nx, ny=ny, steps=steps)
    t_np = time.time() - t0

    # JAX: первый прогон (компиляция + выполнение)
    t0 = time.time()
    u_jax = solver.solve(steps=steps, device=device)
    jax.block_until_ready(u_jax)
    t_jax_first = time.time() - t0

    # JAX: чистое время после компиляции
    t0 = time.time()
    u_jax2 = solver.solve(steps=steps, device=device)
    jax.block_until_ready(u_jax2)
    t_jax_second = time.time() - t0

    print(f"NumPy: {t_np:.4f} s")
    print(f"JAX (1-й запуск, с компиляцией): {t_jax_first:.4f} s")
    print(f"JAX (2-й запуск, без компиляции): {t_jax_second:.4f} s")
    print(f"Ускорение JAX/NumPy (2-й запуск): {t_np / t_jax_second:.2f}x")

    return u_np, u_jax2
