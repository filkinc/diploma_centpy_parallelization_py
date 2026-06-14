import jax
import matplotlib

matplotlib.use('TkAgg')
from jax_centpy.core import Pars2d
from jax_centpy.equations import make_euler_explosion_2d, make_euler_isentropic_vortex_2d
from jax_centpy.solver import Solver2d
from experiments.visualization import create_animation_2d

jax.config.update("jax_enable_x64", True)


def run_test():
    # Параметры расчетной области (взрыв происходит в кубе [0, 1]x[0, 1])
    pars = Pars2d(
        x_init=0.0, x_final=1.0,
        y_init=0.0, y_final=1.0,
        t_final=0.2,  # Короткое время, чтобы волны не дошли до границ
        dt_out=0.01,  # Сохраняем 20 кадров
        Jx=100, Jy=100,
        cfl=0.45,
        scheme="sd2"
    )

    eqn = make_euler_explosion_2d()
    solver = Solver2d(pars, eqn, scheme_name="sd2", limiter_name="superbee")

    results = solver.solve()

    # Рендерим анимацию (var_idx=0 означает плотность)
    create_animation_2d(results, filename="euler_explosion_2d.mp4", fps=15, var_idx=0)


if __name__ == "__main__":
    run_test()
