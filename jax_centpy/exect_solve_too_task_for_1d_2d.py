import numpy as np
import jax.numpy as jnp
from solver import FastSolverWithAllLayersWithoutExtends1d, FastSolverWithAllLayersWithoutExtends2d # или нужный класс
from core import Pars1d, Equation1d, Pars2d, Equation2d

# ================================================================
# УТИЛИТЫ
# ================================================================

def to_np(x):
    return np.array(x)

def errors(num, ref):
    diff = to_np(num) - to_np(ref)
    return np.mean(np.abs(diff)), np.sqrt(np.mean(diff**2)) # L1, L2

def convergence_order(e_coarse, e_fine, r=2.0):
    if e_coarse < 1e-15 or e_fine < 1e-15:
        return float('nan')
    return np.log(e_coarse / e_fine) / np.log(r)

def prim_from_cons_1d(U, gamma=1.4):
    """U: (J, 3) -> rho(J,), u(J,), p(J,)"""
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1.0) * rho * (E - 0.5 * u**2)
    return rho, u, p

def prim_from_cons_2d(U, gamma=1.4):
    """U: (Jx, Jy, 4) -> rho, u, v, p каждый (Jx, Jy)"""
    rho = U[:, :, 0]
    u = U[:, :, 1] / rho
    v = U[:, :, 2] / rho
    E = U[:, :, 3] / rho
    p = (gamma - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    return rho, u, v, p

# ================================================================
# ТОЧНЫЕ РЕШЕНИЯ
# ================================================================

def exact_sine_1d(J, t, gamma=1.4):
    """rho, u, p — каждый shape (J,)"""
    dx = 1.0 / J
    x = np.linspace(0.5*dx, 1.0 - 0.5*dx, J)
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * (x - t))
    u = np.ones(J)
    p = np.ones(J)
    return rho, u, p

def exact_vortex_2d(Jx, t, gamma=1.4, eps=5.0, L=10.0):
    """rho, u, v, p — каждый shape (Jx, Jx)"""
    dx = L / Jx
    x1d = np.linspace(0.5*dx, L - 0.5*dx, Jx)
    x, y = np.meshgrid(x1d, x1d, indexing='ij')
    xc = (5.0 + t) % L
    yc = (5.0 + t) % L
    dx_ = (x - xc + L/2) % L - L/2
    dy_ = (y - yc + L/2) % L - L/2
    r2 = dx_**2 + dy_**2
    coeff = eps / (2.0 * np.pi)
    exp_h = np.exp(0.5 * (1.0 - r2))
    dT = -(gamma-1)*eps**2 / (8*gamma*np.pi**2) * np.exp(1.0 - r2)
    rho = (1.0 + dT) ** (1.0 / (gamma-1))
    u = 1.0 - coeff * dy_ * exp_h
    v = 1.0 + coeff * dx_ * exp_h
    p = (1.0 + dT) ** (gamma / (gamma-1))
    return rho, u, v, p

# ================================================================
# ТОЧНОЕ РЕШЕНИЕ: УРАВНЕНИЕ БЮРГЕРСА 1D
# ================================================================

def exact_burgers_1d(J, t, x_init=0.0, x_final=2.0 * np.pi, tol=1e-12, max_iter=50):
    """
    Точное решение уравнения Бюргерса:
        u_t + (u^2/2)_x = 0,  u(x, 0) = 0.5 * sin(x)
    на периодической области [x_init, x_final].

    Решение задаётся неявно через метод характеристик:
        u = u0(xi),  x = xi + u0(xi) * t,  u0(xi) = 0.5 * sin(xi)

    Для каждого x находим xi итерационно (метод Ньютона).
    Валидно только до образования удара при T_shock = 1 / max|u0'(x)| = 2.0.

    Parameters
    ----------
    J        : int   — число ячеек
    t        : float — момент времени (t < 2.0 для гладкого решения)
    x_init   : float — левая граница
    x_final  : float — правая граница
    tol      : float — допуск сходимости итераций Ньютона
    max_iter : int   — максимальное число итераций

    Returns
    -------
    x  : ndarray (J,)  — координаты центров ячеек
    u  : ndarray (J,)  — точное решение u(x, t)
    """
    dx = (x_final - x_init) / J
    x = np.linspace(x_init + 0.5 * dx, x_final - 0.5 * dx, J)

    # Начальное приближение xi = x (при t=0 совпадает точно)
    xi = x.copy()

    for _ in range(max_iter):
        u0 = 0.5 * np.sin(xi)
        F  = xi + u0 * t - x          # F(xi) = 0
        dF = 1.0 + 0.5 * np.cos(xi) * t  # F'(xi)
        delta = F / dF
        xi -= delta
        if np.max(np.abs(delta)) < tol:
            break

    u = 0.5 * np.sin(xi)
    return x, u

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
                print(f"  {N:>8} | {L1:>11.4e} --- | {L2:>11.4e} ---")
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
        dt_out=t_final, # сохраняем только финальный слой
    )

def verify_sine_1d(sine_equation_cls, scheme_name="sd2", limiter="minmod",
                   grids=(50, 100, 200, 400), t_final=1.0, cfl=0.6, gamma=1.4):
    """
    sine_equation_cls: класс уравнения Эйлера с начальными условиями синуса.
    Создаётся как sine_equation_cls(pars) или sine_equation_cls().
    Уточни сигнатуру под свой Equation1d.
    """
    print("=" * 55)
    print(f"ВЕРИФИКАЦИЯ 1D: ГЛАДКИЙ СИНУС (схема: {scheme_name})")
    print("=" * 55)

    results = []
    for J in grids:

        pars = Pars1d(
            x_init=0.0,
            x_final=1.0,
            t_final=t_final,
            dt_out=0.25,
            J=J,
            cfl=cfl,
            scheme=scheme_name
        )

        solver = FastSolverWithAllLayersWithoutExtends1d(pars, sine_equation_cls, scheme_name=scheme_name, limiter_name=limiter)
        out = solver.solve()

        U_final = to_np(out["u_n"][-1]) # (J, 3)

        rho_n, u_n, p_n = prim_from_cons_1d(U_final, gamma)

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
# 1D: УРАВНЕНИЕ БЮРГЕРСА (порядок точности до удара)
# ================================================================

def verify_burgers_1d(burgers_equation_factory, scheme_name="sd2", limiter="minmod",
                      grids=(50, 100, 200, 400), t_final=0.5, cfl=0.45,
                      x_init=0.0, x_final=2.0 * np.pi):
    """
    Верификация порядка точности на уравнении Бюргерса 1D:
        u_t + (u^2/2)_x = 0,  u(x,0) = 0.5 * sin(x)
    с периодическими граничными условиями.

    Точное решение вычисляется через метод характеристик (итерации Ньютона).
    Валидно до образования удара при T_shock = 2.0.

    Parameters
    ----------
    burgers_equation_factory : callable
        Фабрика уравнения: принимает Pars1d, возвращает Equation1d.
        Пример: lambda pars: make_burgers_1d(periodic=True)
    scheme_name : str   — "sd2" или "fd2"
    limiter     : str   — "minmod", "superbee", "mc", "vanleer"
    grids       : tuple — последовательность чисел ячеек (каждый в 2x крупнее)
    t_final     : float — время (< 2.0 для гладкого решения, иначе TVD-тест)
    cfl         : float — число Куранта
    x_init      : float — левая граница области
    x_final     : float — правая граница области

    Returns
    -------
    results : list of (J, [(L1_u, L2_u)])
    """
    print("=" * 55)
    print(f"ВЕРИФИКАЦИЯ 1D: УРАВНЕНИЕ БЮРГЕРСА (схема: {scheme_name})")
    print(f"  t_final = {t_final:.2f}  "
          f"({'гладкое решение' if t_final < 2.0 else 'ПОСЛЕ удара — только TVD!'})")
    print("=" * 55)

    results = []
    for J in grids:
        pars = Pars1d(
            x_init=x_init,
            x_final=x_final,
            t_final=t_final,
            dt_out=t_final,   # сохраняем только финальный слой
            J=J,
            cfl=cfl,
            scheme=scheme_name,
        )

        eqn = burgers_equation_factory(pars)
        solver = FastSolverWithAllLayersWithoutExtends1d(
            pars, eqn, scheme_name=scheme_name, limiter_name=limiter
        )
        out = solver.solve()

        # u_n shape: (N_times, J, 1) — скалярное уравнение, одна компонента
        U_final = to_np(out["u_n"][-1])   # (J,) или (J, 1)
        u_num = U_final.reshape(-1)        # приводим к (J,)

        # Точное решение через метод характеристик
        _, u_ref = exact_burgers_1d(J, t_final, x_init=x_init, x_final=x_final)

        errs = [errors(u_num, u_ref)]
        results.append((J, errs))
        print(f"  J={J:>5} готово")

    print_convergence_table(results, var_names=["u"])
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
        dt_out=t_final, # сохраняем только финальный слой
    )

def verify_vortex_2d(vortex_equation_cls, scheme_name="sd2", limiter="minmod",
                     grids=(20, 40, 80, 160), t_final=10.0, cfl=0.475, gamma=1.4):
    """
    vortex_equation_cls: класс уравнения Эйлера с начальными условиями вихря.
    """
    print("=" * 55)
    print(f"ВЕРИФИКАЦИЯ 2D: ИЗОЭНТРОПИЙНЫЙ ВИХРЬ (схема: {scheme_name})")
    print("=" * 55)

    results = []
    for N in grids:

        pars = Pars2d(
            x_init=0.0,
            x_final=10.0,
            y_init=0.0,
            y_final=10.0,
            t_final=t_final,
            dt_out=0.005,
            Jx=N,
            Jy=N,
            cfl=cfl,
            scheme="sd2"
        )

        solver = FastSolverWithAllLayersWithoutExtends2d(pars, vortex_equation_cls, scheme_name=scheme_name, limiter_name=limiter)
        out = solver.solve()

        U_final = to_np(out["u"][-1]) # (N, N, 4)

        rho_n, u_n, v_n, p_n = prim_from_cons_2d(U_final, gamma)
        rho_r, u_r, v_r, p_r = exact_vortex_2d(N, t_final, gamma)

        errs = [
            errors(rho_n, rho_r),
            errors(u_n, u_r),
            errors(v_n, v_r),
            errors(p_n, p_r),
        ]
        results.append((N, errs))
        print(f"  N={N:>4}×{N:<4} готово")

    print_convergence_table(results, var_names=["rho", "u", "v", "p"])
    return results

# ================================================================
# ЗАПУСК
# ================================================================

# from equations import make_burgers_1d, make_euler_sine_1d, make_euler_isentropic_vortex_2d

# --- 1D Гладкий синус (Эйлер) ---
# verify_sine_1d(make_euler_sine_1d, scheme_name="sd2", grids=[50, 100, 200, 400])

# --- 1D Бюргерс (порядок точности до удара) ---
# verify_burgers_1d(make_burgers_1d, scheme_name="sd2", t_final=0.5, grids=[50, 100, 200, 400])
# verify_burgers_1d(make_burgers_1d, scheme_name="fd2", t_final=0.5, grids=[50, 100, 200, 400])

# --- 2D Изоэнтропийный вихрь ---
# verify_vortex_2d(make_euler_isentropic_vortex_2d, scheme_name="sd2", grids=[20, 40, 80, 160])