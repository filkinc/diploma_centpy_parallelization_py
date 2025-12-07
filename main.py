from jax_wave_solver_local import JAXWave2D, benchmark
import jax

if __name__ == "__main__":
    # 1. Простое решение и статичный график
    solver = JAXWave2D(nx=256, ny=256)
    device = jax.devices("cpu")[0]
    u = solver.solve(steps=500, device=device)
    print("Решение на устройстве:", u.device)
    solver.plot(u)

    # 2. Анимация
    solver.animate(steps=200, every=5, device=device, interval=50)

    # 3. Бенчмарк NumPy vs JAX
    print("\n== Бенчмарк NumPy vs JAX ==")
    u_np, u_jax = benchmark(nx=256, ny=256, steps=500)

    """
    Результат для сравнения:
    NumPy: 79.2147 s
    JAX (1-й запуск, с компиляцией): 4.4407 s
    JAX (2-й запуск, без компиляции): 4.0477 s
    Ускорение JAX/NumPy (2-й запуск): 19.57x
    """
