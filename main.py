# main.py

from jax_wave_solver import JAXWave2D
import jax

if __name__ == "__main__":
    solver = JAXWave2D(nx=128, ny=128)
    device = jax.devices("cpu")[0]  # или jax.devices("gpu")[0], если есть GPU

    u = solver.solve(steps=200, device=device)
    print("Решение на устройстве:", u.device)
    solver.plot(u)
