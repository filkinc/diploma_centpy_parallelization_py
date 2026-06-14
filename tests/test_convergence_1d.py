import jax
import jax.numpy as jnp
import pandas as pd  # Для красивой таблички
import numpy as np

from jax_centpy.core import Pars1d
from jax_centpy.equations import make_linear_advection_1d
from jax_centpy.solver import Solver1d

jax.config.update("jax_enable_x64", True)


def compute_errors(u_num, u_exact, dx):
    """
    Вычисляет L1 и L_inf ошибки.
    L1 = sum(|err|) * dx
    L_inf = max(|err|)
    """
    err = jnp.abs(u_num - u_exact)
    l1 = jnp.sum(err) * dx
    l_inf = jnp.max(err)
    return float(l1), float(l_inf)


def run_convergence_test():
    print("=== Convergence Test: SD2 + Monotonized_central ===")

    resolutions = [50, 100, 200, 400, 800]
    t_final = 2.0 * jnp.pi
    velocity = 1.0

    results = []

    prev_l1 = None
    prev_linf = None

    for J in resolutions:
        pars = Pars1d(
            x_init=0.0,
            x_final=2.0 * jnp.pi,
            t_final=t_final,
            dt_out=t_final,
            J=J,
            cfl=0.45,
            scheme="sd2"
        )

        eqn = make_linear_advection_1d(velocity=velocity)

        solver = Solver1d(pars, eqn, scheme_name="sd2", limiter_name="mc")

        sol_data = solver.solve()

        u_num = sol_data['u_n'][-1]
        x = sol_data['x']

        u_exact = jnp.sin(x - velocity * t_final)

        l1, l_inf = compute_errors(u_num, u_exact, pars.dx)

        eoc_l1 = np.log2(prev_l1 / l1) if prev_l1 else np.nan
        eoc_linf = np.log2(prev_linf / l_inf) if prev_linf else np.nan

        results.append({
            "J": J,
            "L1 Error": l1,
            "L1 Order": eoc_l1,
            "Linf Error": l_inf,
            "Linf Order": eoc_linf
        })

        prev_l1 = l1
        prev_linf = l_inf

        print(f"J={J}: L1={l1:.2e}, Order={eoc_l1:.2f}")

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df.to_string(formatters={
        'L1 Error': '{:.2e}'.format,
        'L1 Order': '{:.2f}'.format,
        'Linf Error': '{:.2e}'.format,
        'Linf Order': '{:.2f}'.format
    }))


if __name__ == "__main__":
    run_convergence_test()
