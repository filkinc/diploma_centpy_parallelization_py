import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from jax_centpy.core import Pars1d
from jax_centpy.equations import make_linear_advection_1d
from experiments.parallel_solver import ParallelSolver1d
from experiments.visualization import create_animation


def run():
    print(f"Devices available: {jax.device_count()}")

    pars = Pars1d(
        x_init=0.0, x_final=2 * np.pi, t_final=2.0 * np.pi, dt_out=0.5,
        J=400, cfl=0.45, scheme="sd2"
    )
    eqn = make_linear_advection_1d()

    solver = ParallelSolver1d(pars, eqn, limiter_name="mc")
    res = solver.solve()

    # Рисуем
    plt.plot(res['x'], res['u_n'][0], label='Start')
    plt.plot(res['x'], res['u_n'][-1], 'o-', label='End (Parallel)')
    plt.legend()
    plt.show()

    create_animation(res, "parallel_test.gif")


if __name__ == "__main__":
    run()
