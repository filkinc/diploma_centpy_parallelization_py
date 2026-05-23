import numpy as np
import jax.numpy as jnp
from solver import FastSolverWithAllLayersWithoutExtends1d, FastSolverWithAllLayersWithoutExtends2d   # или нужный класс
from core import Pars1d, Equation1d, Pars2d, Equation2d


# ================================================================
# УТИЛИТЫ
# ================================================================

def to_np(x):
    return np.array(x)

def errors(num, ref):
    diff = to_np(num) - to_np(ref)
    return np.mean(np.abs(diff)), np.sqrt(np.mean(diff**2))  # L1, L2

def convergence_order(e_coarse, e_fine, r=2.0):
    if e_coarse < 1e-15 or e_fine < 1e-15:
        return float('nan')
    return np.log(e_coarse / e_fine) / np.log(r)

def prim_from_cons_1d(U, gamma=1.4):
    """U: (J, 3) -> rho(J,), u(J,), p(J,)"""
    rho = U[:, 0]
    u   = U[:, 1] / rho
    E   = U[:, 2] / rho
    p   = (gamma - 1.0) * rho * (E - 0.5 * u**2)
    return rho, u, p

def prim_from_cons_2d(U, gamma=1.4):
    """U: (Jx, Jy, 4) -> rho, u, v, p каждый (Jx, Jy)"""
    rho = U[:, :, 0]
    u   = U[:, :, 1] / rho
    v   = U[:, :, 2] / rho
    E   = U[:, :, 3] / rho
    p   = (gamma - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    return rho, u, v, p


# ================================================================
# ТОЧНЫЕ РЕШЕНИЯ
# ================================================================

def exact_sine_1d(J, t, gamma=1.4):
    """rho, u, p — каждый shape (J,)"""
    dx = 1.0 / J
    x  = np.linspace(0.5*dx, 1.0 - 0.5*dx, J)
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * (x - t))
    u   = np.ones(J)
    p   = np.ones(J)
    return rho, u, p

def exact_vortex_2d(Jx, t, gamma=1.4, eps=5.0, L=10.0):
    """rho, u, v, p — каждый shape (Jx, Jx)"""
    dx  = L / Jx
    x1d = np.linspace(0.5*dx, L - 0.5*dx, Jx)
    x, y = np.meshgrid(x1d, x1d, indexing='ij')
    xc = (5.0 + t) % L
    yc = (5.0 + t) % L
    dx_ = (x - xc + L/2) % L - L/2
    dy_ = (y - yc + L/2) % L - L/2
    r2  = dx_**2 + dy_**2
    coeff = eps / (2.0 * np.pi)
    exp_h = np.exp(0.5 * (1.0 - r2))
    dT    = -(gamma-1)*eps**2 / (8*gamma*np.pi**2) * np.exp(1.0 - r2)
    rho = (1.0 + dT) ** (1.0 / (gamma-1))
    u   = 1.0 - coeff * dy_ * exp_h
    v   = 1.0 + coeff * dx_ * exp_h
    p   = (1.0 + dT) ** (gamma / (gamma-1))
    return rho, u, v, p


# ================================================================
# ПЕЧАТЬ ТАБЛИЦ
# ================================================================

def print_convergence_table(results, var_names):
    """
    results: список (N, [(L1_v0, L2_v0), (L1_v1, L2_v1), ...])
    """
    for vi, vname in enumerate(var_names):
        print(f"\n  {vname}:")
        print(f"  {'N':>8} | {'L1':>11} {'p_L1':>6} | {'L2':>11} {'p_L2':>6}")
        print("  " + "-" * 48)
        for k, (N, errs) in enumerate(results):
            L1, L2 = errs[vi]
            if k == 0:
                print(f"  {N:>8} | {L1:>11.4e}    --- | {L2:>11.4e}    ---")
            else:
                p1 = convergence_order(results[k-1][1][vi][0], L1)
                p2 = convergence_order(results[k-1][1][vi][1], L2)
                print(f"  {N:>8} | {L1:>11.4e} {p1:>6.2f} | {L2:>11.4e} {p2:>6.2f}")


# ================================================================
# 1D: ГЛАДКИЙ СИНУС
# ================================================================

def make_sine_pars(J, cfl=0.45, t_final=1.0):
    """Создаёт Pars1d для теста гладкого синуса."""
    return Pars1d(
        x_init=0.0,
        x_final=1.0,
        J=J,
        cfl=cfl,
        t_final=t_final,
        dt_out=t_final,   # сохраняем только финальный слой
    )

def verify_sine_1d(sine_equation_cls, scheme_name="sd2", limiter="minmod",
                   grids=(50, 100, 200, 400), t_final=1.0, gamma=1.4):
    """
    sine_equation_cls: класс уравнения Эйлера с начальными условиями синуса.
    Создаётся как sine_equation_cls(pars) или sine_equation_cls().
    Уточни сигнатуру под свой Equation1d.
    """
    print("=" * 55)
    print(f"ВЕРИФИКАЦИЯ 1D: ГЛАДКИЙ СИНУС  (схема: {scheme_name})")
    print("=" * 55)

    results = []
    for J in grids:

        pars = Pars1d(
            x_init=0.0,
            x_final=1.0,
            t_final=1,  # Время моделирования (достаточно для сдвига на 1/4 фазы)
            dt_out=0.25,  # Выводим только финальное состояние для экономии памяти
            J=J,  # Количество ячеек (N)
            cfl=0.6,  # Число Куранта
            scheme=scheme_name
        )

        solver = FastSolverWithAllLayersWithoutExtends1d(pars, sine_equation_cls, scheme_name=scheme_name, limiter_name=limiter)
        out    = solver.solve()

        # u_n shape: (N_times, J, 3)  — берём последний слой
        U_final = to_np(out["u_n"][-1])  # (J, 3)

        # Восстанавливаем примитивные переменные
        rho_n, u_n, p_n = prim_from_cons_1d(U_final, gamma)

        # Точное решение
        rho_r, u_r, p_r = exact_sine_1d(J, t_final, gamma)

        errs = [
            errors(rho_n, rho_r),
            errors(u_n, u_r),
            errors(p_n, p_r),
        ]
        results.append((J, errs))
        print(f"  J={J:>5} готово")

        print_convergence_table(results, var_names=["rho", "u", "p"])
    return results


# ================================================================
# 2D: ИЗОЭНТРОПИЙНЫЙ ВИХРЬ
# ================================================================

def make_vortex_pars(N, cfl=0.45, t_final=10.0, L=10.0):
    """Создаёт Pars2d для теста вихря."""
    return Pars2d(
        x_init=0.0, x_final=L, Jx=N,
        y_init=0.0, y_final=L, Jy=N,
        cfl=cfl,
        t_final=t_final,
        dt_out=t_final,   # сохраняем только финальный слой
    )

def verify_vortex_2d(vortex_equation_cls, pars, scheme_name="sd2", limiter="minmod",
                     grids=(20, 40, 80, 160), t_final=10.0, gamma=1.4):
    """
    vortex_equation_cls: класс уравнения Эйлера с начальными условиями вихря.
    """
    print("=" * 55)
    print(f"ВЕРИФИКАЦИЯ 2D: ИЗОЭНТРОПИЙНЫЙ ВИХРЬ  (схема: {scheme_name})")
    print("=" * 55)

    results = []
    for N in grids:

        solver = FastSolverWithAllLayersWithoutExtends2d(pars, vortex_equation_cls, scheme_name=scheme_name, limiter_name=limiter)
        out = solver.solve()

        # u shape: (N_times, Jx, Jy, 4) — берём последний слой
        U_final = to_np(out["u"][-1])           # (N, N, 4)

        rho_n, u_n, v_n, p_n = prim_from_cons_2d(U_final, gamma)
        rho_r, u_r, v_r, p_r = exact_vortex_2d(N, t_final, gamma)

        errs = [
            errors(rho_n, rho_r),
            errors(u_n,   u_r),
            errors(v_n,   v_r),
            errors(p_n,   p_r),
        ]
        results.append((N, errs))
        print(f"  N={N:>4}×{N:<4} готово")

    print_convergence_table(results, var_names=["rho", "u", "v", "p"])
    return results


# ================================================================
# ЗАПУСК
# ================================================================

# verify_sine_1d(MySineEquation1d,   scheme_name="sd2", grids=[50, 100, 200, 400])
# verify_sine_1d(MySineEquation1d,   scheme_name="sd3", grids=[50, 100, 200, 400])
# verify_vortex_2d(MyVortexEquation2d, scheme_name="sd2", grids=[20, 40, 80, 160])
# verify_vortex_2d(MyVortexEquation2d, scheme_name="sd3", grids=[20, 40, 80, 160])