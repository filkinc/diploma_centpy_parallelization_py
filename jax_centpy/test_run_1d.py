import jax
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from core import Pars1d
from equations import make_burgers_1d
from solver import Solver1d
from visualization import create_animation

jax.config.update("jax_enable_x64", True)


def run_test():
    pars = Pars1d(
        x_init=0.0,
        x_final=2.0 * 3.14159,  # 2*pi
        t_final=2.0,
        dt_out=0.1,
        J=200,
        cfl=0.475,
        scheme="sd2"
    )

    # Уравнение
    eqn = make_burgers_1d()

    # Солвер
    solver = Solver1d(pars, eqn, scheme_name="sd2", limiter_name="minmod")

    # Запуск
    results = solver.solve()

    # График
    x = results['x']
    u_all = results['u_n']

    plt.figure(figsize=(10, 6))
    plt.plot(x, u_all[0], label='t=0.0 (Initial)', linestyle='--')
    plt.plot(x, u_all[-1], label=f't={results["t"][-1]:.2f} (Final)')
    plt.title("Burgers Equation (JAX SD2)")
    plt.legend()
    plt.grid(True)
    plt.show()

    create_animation(results, "test_run_1d.gif")


if __name__ == "__main__":
    run_test()
