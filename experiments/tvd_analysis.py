"""
tvd_analysis.py
Инструменты для проверки TVD-свойства численных схем.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (регистрирует стиль 'science')
from typing import Dict, Tuple, Optional
from pathlib import Path

# Единый стиль оформления графиков (как у остальных рисунков работы)
_SCIENCE_STYLE = ['science', 'no-latex']
_PRIMARY_COLOR = '#C62828'   # основная линия (красный)
_SECONDARY_COLOR = '#0C5DA5' # вспомогательная линия (синий)


def compute_tv_1d(u: jnp.ndarray, variable_idx: int = 0) -> float:
    """
    Вычисляет Total Variation для 1D массива.

    Args:
        u: массив состояния (J,) или (J, nvars)
        variable_idx: индекс переменной (по умолчанию 0 = плотность)

    Returns:
        TV(u): полная вариация
    """
    if u.ndim == 1:
        var = u
    else:
        var = u[:, variable_idx]

    return jnp.sum(jnp.abs(jnp.diff(var)))


def compute_tv_2d(u: jnp.ndarray, variable_idx: int = 0) -> float:
    """
    Вычисляет Total Variation для 2D массива.

    Args:
        u: массив состояния (Jx, Jy, nvars)
        variable_idx: индекс переменной (по умолчанию 0 = плотность)

    Returns:
        TV(u): полная вариация в 2D
    """
    var = u[..., variable_idx]

    # TV_x: вариация по x-направлению
    tv_x = jnp.sum(jnp.abs(jnp.diff(var, axis=0)))

    # TV_y: вариация по y-направлению
    tv_y = jnp.sum(jnp.abs(jnp.diff(var, axis=1)))

    return tv_x + tv_y


def compute_tv_evolution_1d(
    solution: Dict,
    variable_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет эволюцию TV для всех временных слоёв (1D).

    Args:
        solution: результат solver.solve() с ключами 't', 'u_n'
        variable_idx: индекс переменной

    Returns:
        times: массив времён
        tv_values: значения TV(t)
    """
    times = np.array(solution['t'])
    u_snapshots = solution['u_n']

    tv_values = []
    for u in u_snapshots:
        tv = compute_tv_1d(u, variable_idx)
        tv_values.append(float(tv))

    return times, np.array(tv_values)


def compute_tv_evolution_2d(
    solution: Dict,
    variable_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет эволюцию TV для всех временных слоёв (2D).

    Args:
        solution: результат solver.solve() с ключами 't', 'u'
        variable_idx: индекс переменной

    Returns:
        times: массив времён
        tv_values: значения TV(t)
    """
    times = np.array(solution['t'])
    u_snapshots = solution['u']

    tv_values = []
    for u in u_snapshots:
        tv = compute_tv_2d(u, variable_idx)
        tv_values.append(float(tv))

    return times, np.array(tv_values)


def check_tvd_property(
    tv_values: np.ndarray,
    tolerance: float = 0.05,
    strict_mode: bool = False
) -> Tuple[bool, Dict]:
    """
    Проверяет выполнение TVD-свойства.

    Args:
        tv_values: массив значений TV(t)
        tolerance: допустимое относительное увеличение TV (по умолчанию 5%)
        strict_mode: если True, проверяет строгую монотонность TV

    Returns:
        is_tvd: True если схема TVD
        diagnostics: словарь с диагностикой
    """
    tv_initial = tv_values[0]
    tv_final = tv_values[-1]

    # Относительное изменение TV от начала до конца
    relative_change = (tv_final - tv_initial) / tv_initial

    # Максимальный локальный скачок между snapshots
    tv_diffs = np.diff(tv_values)
    max_jump = np.max(tv_diffs)
    min_jump = np.min(tv_diffs)
    max_relative_jump = max_jump / tv_initial

    # Проверка монотонности (TV никогда не растёт более чем на tolerance)
    violations = np.sum(tv_diffs > tolerance * tv_initial)

    if strict_mode:
        # Строгая проверка: TV должна монотонно убывать или оставаться постоянной
        is_tvd = np.all(tv_diffs <= 0)
    else:
        # Практическая проверка:
        # 1. Финальная TV не больше начальной + tolerance
        # 2. Нет больших локальных скачков вверх
        is_tvd = (relative_change <= tolerance) and (violations == 0)

    diagnostics = {
        'tv_initial': tv_initial,
        'tv_final': tv_final,
        'relative_change': relative_change,
        'absolute_change': tv_final - tv_initial,
        'max_jump': max_jump,
        'min_jump': min_jump,
        'max_relative_jump': max_relative_jump,
        'violations_count': violations,
        'is_tvd': is_tvd,
        'verdict': _get_verdict(relative_change, violations, strict_mode)
    }

    return is_tvd, diagnostics


def _get_verdict(relative_change, violations, strict_mode):
    """Человекочитаемая оценка TVD-свойства."""
    if relative_change < -0.01:
        return "✓ Отлично: TV убывает (диссипативная схема)"
    elif relative_change < 0.001:
        return "✓ Отлично: TV почти постоянна"
    elif relative_change < 0.05 and violations == 0:
        return "✓ Хорошо: TV слегка растёт в допустимых пределах"
    elif violations == 0:
        return "⚠ Приемлемо: TV растёт, но без локальных скачков"
    else:
        return "✗ TVD нарушено: есть локальные скачки"


def plot_tv_evolution_only(
    solution: Dict,
    variable_idx: int = 0,
    figsize: Tuple = (6, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Отрисовка графика эволюции TV(t) в стиле scienceplots
    (единообразно с остальными рисунками работы).

    Args:
        solution: результат solver.solve()
        variable_idx: индекс переменной (0 = ρ)
        figsize: размер графика
        show: показать график (True) или только сохранить (False)
        save_path: путь для сохранения (None = не сохранять)
        dpi: разрешение

    Returns:
        None (график отображается или сохраняется)
    """
    # Вычисляем TV эволюцию
    tv_times, tv_values = compute_tv_evolution_1d(solution, variable_idx)

    with plt.style.context(_SCIENCE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Эволюция полной вариации
        ax.plot(tv_times, tv_values, color=_PRIMARY_COLOR, linewidth=1.8,
                label=r'$\mathrm{TV}(t)$')

        # Референсная линия TV(t=0)
        ax.axhline(tv_values[0], color='black', linestyle='--', linewidth=1.2,
                   label=rf'$\mathrm{{TV}}(t=0) = {tv_values[0]:.4f}$')

        ax.set_xlabel(r'Время $t$', fontsize=24)
        ax.set_ylabel(r'$\mathrm{TV}(u)$', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.legend(fontsize=22, loc='best')

        fig.tight_layout()

        # Сохранение (если указан путь)
        if save_path is not None:
            output_file = Path(save_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"✓ Сохранено: {output_file}")

        # Отображение (если show=True)
        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================================================================
# СТАРЫЕ ФУНКЦИИ (для обратной совместимости)
# ============================================================================

def plot_tvd_analysis_1d_separate(
    solution: Dict,
    exact_solution: Optional[Dict] = None,
    variable_idx: int = 0,
    variable_name: str = "Плотность ρ",
    snapshot_indices: Optional[list] = None,
    output_dir: str = "tvd_figures",
    prefix: str = "tvd_1d",
    figsize_profiles: Tuple = (12, 8),
    figsize_tv: Tuple = (10, 6),
    figsize_zoom: Tuple = (10, 6),
    dpi: int = 150,
    show_points: bool = True,
    point_step: int = 5
):
    """
    Создаёт TVD-анализ для 1D задачи с сохранением каждого графика отдельно.
    (СТАРАЯ ВЕРСИЯ - для совместимости)
    """
    # Создаём папку для результатов
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    x = np.array(solution['x'])
    times = np.array(solution['t'])
    u_snapshots = solution['u_n']

    # Выбор snapshots
    if snapshot_indices is None:
        n_snapshots = min(6, len(times))
        snapshot_indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)

    # Вычисляем TV эволюцию
    tv_times, tv_values = compute_tv_evolution_1d(solution, variable_idx)

    # Стиль линии
    line_style = 'o-' if show_points else '-'
    markersize = 3 if show_points else 0
    markevery = point_step if show_points else None

    saved_files = {}

    # ==================== График 1: Профили в разные моменты ====================
    fig1, ax1 = plt.subplots(figsize=figsize_profiles)

    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

    for idx, snap_idx in enumerate(snapshot_indices):
        u = u_snapshots[snap_idx]
        t = times[snap_idx]

        if u.ndim == 1:
            var = u
        else:
            var = u[:, variable_idx]

        ax1.plot(x, var, line_style, color=colors[idx],
                label=f't = {t:.3f}', markersize=markersize,
                linewidth=1.5, markevery=markevery)

        # Если есть точное решение
        if exact_solution is not None:
            u_exact = exact_solution['u_n'][snap_idx]
            if u_exact.ndim == 1:
                var_exact = u_exact
            else:
                var_exact = u_exact[:, variable_idx]
            ax1.plot(x, var_exact, '--', color=colors[idx],
                    linewidth=2, alpha=0.7)

    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel(variable_name, fontsize=14)
    ax1.set_title('Профили решения в разные моменты времени',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    # Сохраняем
    file1 = output_path / f"{prefix}_profiles.png"
    fig1.savefig(file1, dpi=dpi, bbox_inches='tight')
    saved_files['profiles'] = str(file1)
    plt.close(fig1)
    print(f"✓ Сохранено: {file1}")

    # ==================== График 2: Эволюция TV ====================
    fig2, ax2 = plt.subplots(figsize=figsize_tv)

    ax2.plot(tv_times, tv_values, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(tv_values[0], color='r', linestyle='--', linewidth=2,
               label=f'TV(t=0) = {tv_values[0]:.4f}')

    # Проверка монотонности
    tv_increase = tv_values[-1] - tv_values[0]
    tv_max_increase = np.max(np.diff(tv_values))

    textstr = f'ΔTV = {tv_increase:.2e}\nMax jump = {tv_max_increase:.2e}'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Время t', fontsize=24)
    ax2.set_ylabel('TV(u)', fontsize=24)
    ax2.set_title('Эволюция полной вариации', fontsize=24, fontweight='bold')
    ax2.legend(fontsize=22)
    ax2.grid(True, alpha=0.3)

    # Сохраняем
    file2 = output_path / f"{prefix}_tv_evolution.png"
    fig2.savefig(file2, dpi=dpi, bbox_inches='tight')
    saved_files['tv_evolution'] = str(file2)
    plt.close(fig2)
    print(f"✓ Сохранено: {file2}")

    # ==================== График 3: Zoom на разрыв ====================
    fig3, ax3 = plt.subplots(figsize=figsize_zoom)

    u_final = u_snapshots[-1]
    if u_final.ndim == 1:
        var_final = u_final
    else:
        var_final = u_final[:, variable_idx]

    # Находим область с максимальным градиентом
    grad = np.abs(np.diff(var_final))
    shock_idx = np.argmax(grad)

    # Zoom: ±20 точек вокруг разрыва
    zoom_range = 20
    idx_min = max(0, shock_idx - zoom_range)
    idx_max = min(len(x) - 1, shock_idx + zoom_range)

    x_zoom = x[idx_min:idx_max]
    var_zoom = var_final[idx_min:idx_max]

    ax3.plot(x_zoom, var_zoom, 'bo-', markersize=5, linewidth=1.5,
            label='Численное', markevery=1)

    if exact_solution is not None:
        u_exact_final = exact_solution['u_n'][-1]
        if u_exact_final.ndim == 1:
            var_exact_final = u_exact_final
        else:
            var_exact_final = u_exact_final[:, variable_idx]

        var_exact_zoom = var_exact_final[idx_min:idx_max]
        ax3.plot(x_zoom, var_exact_zoom, 'r--', linewidth=2,
                alpha=0.7, label='Точное')

    ax3.set_xlabel('x', fontsize=14)
    ax3.set_ylabel(variable_name, fontsize=14)
    ax3.set_title(f'Zoom на разрыв (t = {times[-1]:.3f})',
                  fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Сохраняем
    file3 = output_path / f"{prefix}_zoom.png"
    fig3.savefig(file3, dpi=dpi, bbox_inches='tight')
    saved_files['zoom'] = str(file3)
    plt.close(fig3)
    print(f"✓ Сохранено: {file3}")

    return saved_files


def plot_tv_evolution_only_2d(
        solution: Dict,
        variable_idx: int = 0,
        figsize: Tuple = (6, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300
):
    """
    Отрисовка графика эволюции TV(t) для 2D в стиле scienceplots
    (единообразно с остальными рисунками работы).

    Args:
        solution: результат solver.solve() для 2D
        variable_idx: индекс переменной (0 = ρ)
        figsize: размер графика
        show: показать график (True) или только сохранить (False)
        save_path: путь для сохранения (None = не сохранять)
        dpi: разрешение

    Returns:
        None (график отображается или сохраняется)
    """
    # Вычисляем TV эволюцию
    tv_times, tv_values = compute_tv_evolution_2d(solution, variable_idx)

    with plt.style.context(_SCIENCE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Эволюция полной вариации
        ax.plot(tv_times, tv_values, color=_PRIMARY_COLOR, linewidth=1.8,
                label=r'$\mathrm{TV}(t)$')

        # Референсная линия TV(t=0)
        ax.axhline(tv_values[0], color='black', linestyle='--', linewidth=1.2,
                   label=rf'$\mathrm{{TV}}(t=0) = {tv_values[0]:.4f}$')

        ax.set_xlabel(r'Время $t$', fontsize=24)
        ax.set_ylabel(r'$\mathrm{TV}(u)$', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.legend(fontsize=22, loc='best')

        fig.tight_layout()

        # Сохранение (если указан путь)
        if save_path is not None:
            output_file = Path(save_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"✓ Сохранено: {output_file}")

        # Отображение (если show=True)
        if show:
            plt.show()
        else:
            plt.close(fig)


# Пример использования 1D

# from tvd_analysis import (
#     plot_tv_evolution_only,  # НОВАЯ ФУНКЦИЯ!
#     compute_tv_evolution_1d,
#     check_tvd_property
# )
#
# # Инициализация задачи Гладкого синуса
# eqn = make_euler_sine_1d()
# pars = Pars1d(
#     x_init=0.0,
#     x_final=1.0,
#     t_final=1.0,
#     dt_out=0.25,
#     J=400,
#     cfl=0.475,
#     scheme="sd2"
# )
#
# # Решение задачи
# solver = FastSolverWithAllLayersWithoutExtends1d(pars, eqn, limiter_name="minmod", scheme_name="sd2")
# sol = solver.solve()
#
# # ==================== ПРОСТО ПОКАЗАТЬ ГРАФИК ====================
# print("\n" + "="*60)
# print("ГРАФИК ЭВОЛЮЦИИ TV(t)")
# print("="*60)
#
# plot_tv_evolution_only(
#     sol,
#     variable_idx=0,
#     figsize=(10, 6),
#     show=True,           # Показать на экране
#     save_path=None,      # НЕ сохранять в файл
#     dpi=150
# )
#
# # ==================== TVD-ДИАГНОСТИКА ====================
# tv_times, tv_values = compute_tv_evolution_1d(sol, variable_idx=0)
# is_tvd, diag = check_tvd_property(tv_values, tolerance=0.05)
#
# print("\n" + "="*60)
# print("TVD-ДИАГНОСТИКА")
# print("="*60)
# print(f"TV начальная:              {diag['tv_initial']:.6f}")
# print(f"TV финальная:              {diag['tv_final']:.6f}")
# print(f"Абсолютное изменение:      {diag['absolute_change']:.6e}")
# print(f"Относительное изменение:   {diag['relative_change']:.4e} ({diag['relative_change']*100:.2f}%)")
# print(f"Макс локальный скачок:     {diag['max_jump']:.4e}")
# print(f"Количество нарушений:      {diag['violations_count']}")
# print("-" * 60)
# print(f"ВЕРДИКТ: {diag['verdict']}")
# print(f"TVD выполняется: {'✓ ДА' if is_tvd else '✗ НЕТ'}")
# print("="*60)


# Пример использования 2D

# from tvd_analysis import (
#     plot_tv_evolution_only_2d,  # ДЛЯ 2D!
#     compute_tv_evolution_2d,
#     check_tvd_property
# )
#
# # Инициализация 2D задачи (например, вихрь)
# eqn_2d = make_isentropic_vortex_2d(
#     center=(5.0, 5.0),
#     strength=5.0,
#     velocity_inf=(1.0, 1.0)
# )
#
# pars_2d = Pars2d(
#     x_init=0.0,
#     x_final=10.0,
#     y_init=0.0,
#     y_final=10.0,
#     t_final=10.0,
#     dt_out=0.5,
#     Jx=80,
#     Jy=80,
#     cfl=0.475,
#     scheme="sd2"
# )
#
# # Решение задачи
# from solver import FastSolverWithAllLayersWithoutExtends2d
#
# solver_2d = FastSolverWithAllLayersWithoutExtends2d(
#     pars_2d, eqn_2d, limiter_name="minmod", scheme_name="sd2"
# )
# sol_2d = solver_2d.solve()
#
# # ==================== ПРОСТО ПОКАЗАТЬ ГРАФИК ====================
# print("\n" + "="*60)
# print("ГРАФИК ЭВОЛЮЦИИ TV(t) для 2D")
# print("="*60)
#
# plot_tv_evolution_only_2d(
#     sol_2d,
#     variable_idx=0,
#     figsize=(10, 6),
#     show=True,           # Показать на экране
#     save_path=None,      # НЕ сохранять в файл
#     dpi=150
# )
#
# # ==================== TVD-ДИАГНОСТИКА ====================
# tv_times, tv_values = compute_tv_evolution_2d(sol_2d, variable_idx=0)
# is_tvd, diag = check_tvd_property(tv_values, tolerance=0.05)
#
# print("\n" + "="*60)
# print("TVD-ДИАГНОСТИКА (2D)")
# print("="*60)
# print(f"TV начальная:              {diag['tv_initial']:.6f}")
# print(f"TV финальная:              {diag['tv_final']:.6f}")
# print(f"Абсолютное изменение:      {diag['absolute_change']:.6e}")
# print(f"Относительное изменение:   {diag['relative_change']:.4e} ({diag['relative_change']*100:.2f}%)")
# print(f"Макс локальный скачок:     {diag['max_jump']:.4e}")
# print(f"Количество нарушений:      {diag['violations_count']}")
# print("-" * 60)
# print(f"ВЕРДИКТ: {diag['verdict']}")
# print(f"TVD выполняется: {'✓ ДА' if is_tvd else '✗ НЕТ'}")
# print("="*60)