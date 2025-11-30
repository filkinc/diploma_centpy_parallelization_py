from jax_wave_solver import JAXWave2D, benchmark
import jax

if __name__ == "__main__":
    # 1. Простое решение и статичный график
    solver = JAXWave2D(nx=128, ny=128)
    device = jax.devices("cpu")[0]
    u = solver.solve(steps=200, device=device)
    print("Решение на устройстве:", u.device)
    solver.plot(u)

    # 2. Анимация
    solver.animate(steps=200, every=5, device=device, interval=50)

    # 3. Бенчмарк NumPy vs JAX
    print("\n== Бенчмарк NumPy vs JAX ==")
    u_np, u_jax = benchmark(nx=128, ny=128, steps=300)
