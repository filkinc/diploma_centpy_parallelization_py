import jax
import jax.numpy as jnp
import pandas as pd
from core import Pars2d
from equations import make_euler_isentropic_vortex_2d
from solver import FastSolver2d

jax.config.update("jax_enable_x64", True)


def compute_errors(u_num, u_exact, dx, dy):
    # Ошибка по плотности (компонент [..., 0])
    err = jnp.abs(u_num[..., 0] - u_exact[..., 0])
    l1 = jnp.sum(err) * dx * dy
    return float(l1)


def run_convergence_test():
    resolutions = [20, 40, 80, 160]
    results = []

    eqn, exact_solution_fn = make_euler_isentropic_vortex_2d()
    t_final = 2.0  # Время переноса

    for J in resolutions:
        pars = Pars2d(
            x_init=0.0, x_final=10.0, y_init=0.0, y_final=10.0,
            t_final=t_final, dt_out=t_final, Jx=J, Jy=J, cfl=0.45, scheme="sd2"
        )

        solver = FastSolver2d(pars, eqn, scheme_name="sd2")

        x_1d = jnp.linspace(pars.x_init + pars.dx / 2, pars.x_final - pars.dx / 2, J)
        y_1d = jnp.linspace(pars.y_init + pars.dy / 2, pars.y_final - pars.dy / 2, J)
        X, Y = jnp.meshgrid(x_1d, y_1d, indexing='ij')

        u_init = eqn.initial_data(X, Y)
        print(f"Running grid {J}x{J}...")

        u_final, _ = solver.solve_jit(u_init)
        u_exact = exact_solution_fn(X, Y, t_final)

        l1_err = compute_errors(u_final, u_exact, pars.dx, pars.dy)
        results.append({"J": J, "L1_Error": l1_err})

    df = pd.DataFrame(results)
    df["Order_L1"] = jnp.nan
    for i in range(1, len(df)):
        df.loc[i, "Order_L1"] = jnp.log2(df.loc[i - 1, "L1_Error"] / df.loc[i, "L1_Error"])

    print("\nConvergence Results (Euler 2D - Isentropic Vortex):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_convergence_test()
