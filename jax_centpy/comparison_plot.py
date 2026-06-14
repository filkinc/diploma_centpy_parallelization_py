"""
comparison_plots.py
Функции для сравнения численного и аналитического решений (1D и 2D).
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def plot_1d_comparison(
        solution_numerical: Dict,
        solution_exact: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_index: int = -1,
        output_path: Optional[str] = None,
        figsize: Tuple = (12, 7),
        dpi: int = 150,
        show_points: bool = True,
        point_step: int = 3,
        title: Optional[str] = None
) -> str:
    """
    Сравнение численного и аналитического решений для 1D задачи.

    Args:
        solution_numerical: результат solver.solve() для численного решения
        solution_exact: результат solver.solve() для точного решения (или аналитика)
        variable_idx: индекс переменной (0=ρ, 1=ρu, 2=E)
        variable_name: название переменной для подписи осей
        snapshot_index: индекс временного слоя (-1 = последний)
        output_path: путь для сохранения (None = не сохранять)
        figsize: размер фигуры
        dpi: разрешение
        show_points: показывать точки численного решения
        point_step: шаг отображения точек
        title: заголовок графика (None = автоматический)

    Returns:
        path: путь к сохранённому файлу (если output_path указан)
    """
    # Извлечение данных
    x = np.array(solution_numerical['x'])
    t = np.array(solution_numerical['t'])[snapshot_index]

    u_num = solution_numerical['u_n'][snapshot_index]
    u_exact = solution_exact['u_n'][snapshot_index]

    # Извлечение нужной переменной
    if u_num.ndim == 1:
        var_num = u_num
        var_exact = u_exact
    else:
        var_num = u_num[:, variable_idx]
        var_exact = u_exact[:, variable_idx]

    # Создание графика
    fig, ax = plt.subplots(figsize=figsize)

    # Точное решение - жирная сплошная линия
    ax.plot(x, var_exact, 'r-', linewidth=2.5, label='Точное решение', zorder=1)

    # Численное решение
    if show_points:
        # С точками
        ax.plot(x, var_num, 'b-', linewidth=1.2, alpha=0.7,
                label='Численное решение', zorder=2)
        ax.plot(x[::point_step], var_num[::point_step], 'bo',
                markersize=4, zorder=3)
    else:
        # Без точек
        ax.plot(x, var_num, 'b--', linewidth=1.8, alpha=0.8,
                label='Численное решение', zorder=2)

    # Оформление
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel(variable_name, fontsize=14)

    if title is None:
        title = f'Сравнение численного и аналитического решений (t = {t:.3f})'
    ax.set_title(title, fontsize=16, fontweight='bold')

    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Подпись с параметрами
    N = len(x)
    textstr = f'N = {N}\nt = {t:.3f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Сохранение
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Сохранено: {output_file}")
        return str(output_file)
    else:
        plt.show()
        return ""


def plot_1d_comparison_error(
        solution_numerical: Dict,
        solution_exact: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_index: int = -1,
        output_path: Optional[str] = None,
        figsize: Tuple = (12, 10),
        dpi: int = 150,
        show_points: bool = True,
        point_step: int = 3
) -> str:
    """
    Сравнение + график ошибки (два subplot'а).

    Верхний график: численное vs точное
    Нижний график: абсолютная ошибка |u_num - u_exact|
    """
    # Извлечение данных
    x = np.array(solution_numerical['x'])
    t = np.array(solution_numerical['t'])[snapshot_index]

    u_num = solution_numerical['u_n'][snapshot_index]
    u_exact = solution_exact['u_n'][snapshot_index]

    if u_num.ndim == 1:
        var_num = u_num
        var_exact = u_exact
    else:
        var_num = u_num[:, variable_idx]
        var_exact = u_exact[:, variable_idx]

    error = np.abs(var_num - var_exact)

    # Создание двух subplot'ов
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # ========== Subplot 1: Сравнение ==========
    ax1.plot(x, var_exact, 'r-', linewidth=2.5, label='Точное', zorder=1)

    if show_points:
        ax1.plot(x, var_num, 'b-', linewidth=1.2, alpha=0.7,
                 label='Численное', zorder=2)
        ax1.plot(x[::point_step], var_num[::point_step], 'bo',
                 markersize=4, zorder=3)
    else:
        ax1.plot(x, var_num, 'b--', linewidth=1.8, alpha=0.8,
                 label='Численное', zorder=2)

    ax1.set_ylabel(variable_name, fontsize=14)
    ax1.set_title(f'Сравнение решений (t = {t:.3f})',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== Subplot 2: Ошибка ==========
    ax2.plot(x, error, 'g-', linewidth=1.5, label='|Численное - Точное|')
    ax2.fill_between(x, 0, error, alpha=0.3, color='green')

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('Абсолютная ошибка', fontsize=14)
    ax2.set_title('Распределение ошибки', fontsize=14)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Статистика ошибки
    max_error = np.max(error)
    l1_error = np.mean(error)
    l2_error = np.sqrt(np.mean(error ** 2))

    textstr = f'Max: {max_error:.2e}\n$L_1$: {l1_error:.2e}\n$L_2$: {l2_error:.2e}'
    ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Сохранение
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Сохранено: {output_file}")
        return str(output_file)
    else:
        plt.show()
        return ""


def plot_2d_comparison(
        solution_numerical: Dict,
        solution_exact: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_index: int = -1,
        output_path: Optional[str] = None,
        figsize: Tuple = (16, 6),
        dpi: int = 150,
        cmap: str = 'jet',
        levels: int = 50
) -> str:
    """
    Сравнение численного и аналитического решений для 2D задачи (contourf).

    Создаёт 3 subplot'а: численное | точное | ошибка

    Args:
        solution_numerical: результат solver.solve() для численного решения
        solution_exact: результат solver.solve() для точного решения
        variable_idx: индекс переменной (0=ρ, 1=ρu, 2=ρv, 3=E)
        variable_name: название переменной
        snapshot_index: индекс временного слоя (-1 = последний)
        output_path: путь для сохранения
        figsize: размер фигуры
        dpi: разрешение
        cmap: цветовая карта ('jet', 'viridis', 'coolwarm')
        levels: количество уровней в contourf

    Returns:
        path: путь к сохранённому файлу
    """
    # Извлечение данных
    X = np.array(solution_numerical['X'])
    Y = np.array(solution_numerical['Y'])
    t = np.array(solution_numerical['t'])[snapshot_index]

    u_num = solution_numerical['u'][snapshot_index]
    u_exact = solution_exact['u'][snapshot_index]

    var_num = u_num[..., variable_idx]
    var_exact = u_exact[..., variable_idx]

    error = np.abs(var_num - var_exact)

    # Создание 3 subplot'ов
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Общие пределы цветовой шкалы для численного и точного
    vmin = min(var_num.min(), var_exact.min())
    vmax = max(var_num.max(), var_exact.max())

    # ========== Subplot 1: Численное ==========
    im1 = axes[0].contourf(X, Y, var_num, levels=levels, cmap=cmap,
                           vmin=vmin, vmax=vmax)
    axes[0].set_title('Численное решение', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label=variable_name)

    # ========== Subplot 2: Точное ==========
    im2 = axes[1].contourf(X, Y, var_exact, levels=levels, cmap=cmap,
                           vmin=vmin, vmax=vmax)
    axes[1].set_title('Точное решение', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label=variable_name)

    # ========== Subplot 3: Ошибка ==========
    im3 = axes[2].contourf(X, Y, error, levels=levels, cmap='Reds')
    axes[2].set_title('Абсолютная ошибка', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], label=f'|{variable_name}_num - {variable_name}_exact|')

    # Статистика ошибки
    max_error = error.max()
    l1_error = error.mean()
    l2_error = np.sqrt((error ** 2).mean())

    textstr = f'Max: {max_error:.2e}\n$L_1$: {l1_error:.2e}\n$L_2$: {l2_error:.2e}'
    axes[2].text(0.02, 0.98, textstr, transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Общий заголовок
    fig.suptitle(f'Сравнение решений: {variable_name} (t = {t:.2f})',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Сохранение
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Сохранено: {output_file}")
        return str(output_file)
    else:
        plt.show()
        return ""


def plot_2d_comparison_profiles(
        solution_numerical: Dict,
        solution_exact: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_index: int = -1,
        slice_x: Optional[float] = None,
        slice_y: Optional[float] = None,
        output_path: Optional[str] = None,
        figsize: Tuple = (14, 6),
        dpi: int = 150,
        show_points: bool = True,
        point_step: int = 3
) -> str:
    """
    Сравнение 1D профилей вдоль сечений x=const и y=const для 2D задачи.

    Создаёт 2 subplot'а: профиль по x | профиль по y

    Args:
        solution_numerical: численное решение
        solution_exact: точное решение
        variable_idx: индекс переменной
        variable_name: название переменной
        snapshot_index: индекс временного слоя
        slice_x: x-координата вертикального сечения (None = центр)
        slice_y: y-координата горизонтального сечения (None = центр)
        output_path: путь для сохранения
        figsize: размер фигуры
        dpi: разрешение
        show_points: показывать точки
        point_step: шаг точек

    Returns:
        path: путь к сохранённому файлу
    """
    # Извлечение данных
    X = np.array(solution_numerical['X'])
    Y = np.array(solution_numerical['Y'])
    t = np.array(solution_numerical['t'])[snapshot_index]

    u_num = solution_numerical['u'][snapshot_index]
    u_exact = solution_exact['u'][snapshot_index]

    var_num = u_num[..., variable_idx]
    var_exact = u_exact[..., variable_idx]

    # Определение индексов сечений
    if slice_x is None:
        # Центр по x
        Jx = X.shape[1]
        idx_x = Jx // 2
        slice_x = X[0, idx_x]
    else:
        # Найти ближайший индекс
        idx_x = np.argmin(np.abs(X[0, :] - slice_x))
        slice_x = X[0, idx_x]

    if slice_y is None:
        # Центр по y
        Jy = Y.shape[0]
        idx_y = Jy // 2
        slice_y = Y[idx_y, 0]
    else:
        # Найти ближайший индекс
        idx_y = np.argmin(np.abs(Y[:, 0] - slice_y))
        slice_y = Y[idx_y, 0]

    # Извлечение профилей
    # Вдоль y (при x = const)
    y_coords = Y[:, idx_x]
    profile_num_y = var_num[:, idx_x]
    profile_exact_y = var_exact[:, idx_x]

    # Вдоль x (при y = const)
    x_coords = X[idx_y, :]
    profile_num_x = var_num[idx_y, :]
    profile_exact_x = var_exact[idx_y, :]

    # Создание графика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ========== Subplot 1: Профиль вдоль y (x = const) ==========
    ax1.plot(y_coords, profile_exact_y, 'r-', linewidth=2.5,
             label='Точное', zorder=1)

    if show_points:
        ax1.plot(y_coords, profile_num_y, 'b-', linewidth=1.2, alpha=0.7,
                 label='Численное', zorder=2)
        ax1.plot(y_coords[::point_step], profile_num_y[::point_step], 'bo',
                 markersize=4, zorder=3)
    else:
        ax1.plot(y_coords, profile_num_y, 'b--', linewidth=1.8, alpha=0.8,
                 label='Численное', zorder=2)

    ax1.set_xlabel('y', fontsize=14)
    ax1.set_ylabel(variable_name, fontsize=14)
    ax1.set_title(f'Профиль вдоль y (x = {slice_x:.2f})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== Subplot 2: Профиль вдоль x (y = const) ==========
    ax2.plot(x_coords, profile_exact_x, 'r-', linewidth=2.5,
             label='Точное', zorder=1)

    if show_points:
        ax2.plot(x_coords, profile_num_x, 'b-', linewidth=1.2, alpha=0.7,
                 label='Численное', zorder=2)
        ax2.plot(x_coords[::point_step], profile_num_x[::point_step], 'bo',
                 markersize=4, zorder=3)
    else:
        ax2.plot(x_coords, profile_num_x, 'b--', linewidth=1.8, alpha=0.8,
                 label='Численное', zorder=2)

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel(variable_name, fontsize=14)
    ax2.set_title(f'Профиль вдоль x (y = {slice_y:.2f})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Общий заголовок
    fig.suptitle(f'Сравнение профилей: {variable_name} (t = {t:.2f})',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Сохранение
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Сохранено: {output_file}")
        return str(output_file)
    else:
        plt.show()
        return ""


def plot_2d_side_by_side(
        solution_numerical: Dict,
        solution_exact: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_index: int = -1,
        output_dir: str = "comparison_figures",
        prefix: str = "vortex",
        figsize: Tuple = (8, 7),
        dpi: int = 150,
        cmap: str = 'jet',
        levels: int = 50
) -> Dict[str, str]:
    """
    Создаёт отдельные файлы для численного и точного решений (для LaTeX subfigure).

    Удобно для вставки в диплом как:
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics{vortex_numerical.png}
        \\caption{Численное решение}
    \\end{subfigure}
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics{vortex_exact.png}
        \\caption{Точное решение}
    \\end{subfigure}

    Returns:
        paths: {'numerical': path1, 'exact': path2}
    """
    # Извлечение данных
    X = np.array(solution_numerical['X'])
    Y = np.array(solution_numerical['Y'])
    t = np.array(solution_numerical['t'])[snapshot_index]

    u_num = solution_numerical['u'][snapshot_index]
    u_exact = solution_exact['u'][snapshot_index]

    var_num = u_num[..., variable_idx]
    var_exact = u_exact[..., variable_idx]

    # Общие пределы цветовой шкалы
    vmin = min(var_num.min(), var_exact.min())
    vmax = max(var_num.max(), var_exact.max())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # ========== График 1: Численное ==========
    fig1, ax1 = plt.subplots(figsize=figsize)
    im1 = ax1.contourf(X, Y, var_num, levels=levels, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    ax1.set_title('Численное решение', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, label=variable_name)
    cbar1.ax.tick_params(labelsize=11)

    file1 = output_path / f"{prefix}_numerical_2d.png"
    fig1.savefig(file1, dpi=dpi, bbox_inches='tight')
    plt.close(fig1)
    paths['numerical'] = str(file1)
    print(f"✓ Сохранено: {file1}")

    # ========== График 2: Точное ==========
    fig2, ax2 = plt.subplots(figsize=figsize)
    im2 = ax2.contourf(X, Y, var_exact, levels=levels, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    ax2.set_title('Точное решение', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, label=variable_name)
    cbar2.ax.tick_params(labelsize=11)

    file2 = output_path / f"{prefix}_exact_2d.png"
    fig2.savefig(file2, dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    paths['exact'] = str(file2)
    print(f"✓ Сохранено: {file2}")

    return paths

# Пример использования
# from solver import FastSolverWithAllLayersWithoutExtends1d
# from equations import make_smooth_sine_wave, make_isentropic_vortex_2d
# from comparison_plots import (
#     plot_1d_comparison,
#     plot_1d_comparison_error,
#     plot_2d_comparison,
#     plot_2d_comparison_profiles,
#     plot_2d_side_by_side
# )
#
# # ==================== 1D: Гладкий синус ====================
# print("\n" + "="*60)
# print("1D: ГЛАДКИЙ СИНУС")
# print("="*60)
#
# # Параметры
# pars_1d = {
#     'J': 400,
#     'T': 1.0,
#     'C': 0.475,
#     'x_range': (0.0, 1.0),
#     'boundary': 'periodic'
# }
#
# # Уравнение и начальные условия
# eqn_1d = make_smooth_sine_wave(amplitude=0.2, wavelength=1.0)
#
# # Численное решение
# solver_num = FastSolverWithAllLayersWithoutExtends1d(
#     pars_1d, eqn_1d, limiter_name="minmod"
# )
# sol_num = solver_num.solve()
#
# # Точное решение (аналитическое)
# solver_exact = FastSolverWithAllLayersWithoutExtends1d(
#     pars_1d, eqn_1d, limiter_name="minmod"
# )
# # Для синуса точное решение = начальное условие сдвинутое на u*t
# # (Если у тебя есть функция для аналитики, используй её)
# sol_exact = solver_exact.solve()  # Замени на аналитику!
#
# # Графики
# print("\n--- Создание графиков 1D ---")
#
# plot_1d_comparison(
#     sol_num, sol_exact,
#     variable_idx=0,
#     variable_name="Плотность ρ",
#     output_path="diploma_figures/smooth_sine_comparison_1d.png",
#     dpi=300
# )
#
# plot_1d_comparison_error(
#     sol_num, sol_exact,
#     variable_idx=0,
#     variable_name="Плотность ρ",
#     output_path="diploma_figures/smooth_sine_comparison_error_1d.png",
#     dpi=300
# )
#
# # ==================== 2D: Вихрь Ху-Шу ====================
# print("\n" + "="*60)
# print("2D: ВИХРЬ ХУ-ШУ")
# print("="*60)
#
# # Параметры
# pars_2d = {
#     'Jx': 80,
#     'Jy': 80,
#     'T': 10.0,
#     'C': 0.475,
#     'x_range': (0.0, 10.0),
#     'y_range': (0.0, 10.0),
#     'boundary': 'periodic'
# }
#
# # Уравнение
# eqn_2d = make_isentropic_vortex_2d(
#     center=(5.0, 5.0),
#     strength=5.0,
#     velocity_inf=(1.0, 1.0)
# )
#
# # Численное решение
# from solver import FastSolverWithAllLayersWithoutExtends2d
#
# solver_num_2d = FastSolverWithAllLayersWithoutExtends2d(
#     pars_2d, eqn_2d, limiter_name="minmod"
# )
# sol_num_2d = solver_num_2d.solve()
#
# # Точное решение (аналитическое)
# sol_exact_2d = solver_num_2d.solve()  # Замени на аналитику!
#
# # Графики
# print("\n--- Создание графиков 2D ---")
#
# plot_2d_comparison(
#     sol_num_2d, sol_exact_2d,
#     variable_idx=0,
#     variable_name="Плотность ρ",
#     output_path="diploma_figures/vortex_comparison_2d.png",
#     dpi=300,
#     cmap='jet'
# )
#
# plot_2d_comparison_profiles(
#     sol_num_2d, sol_exact_2d,
#     variable_idx=0,
#     variable_name="Плотность ρ",
#     slice_x=5.0,
#     slice_y=5.0,
#     output_path="diploma_figures/vortex_profiles_2d.png",
#     dpi=300
# )
#
# # Отдельные файлы для subfigure
# plot_2d_side_by_side(
#     sol_num_2d, sol_exact_2d,
#     variable_idx=0,
#     variable_name="Плотность ρ",
#     output_dir="diploma_figures",
#     prefix="vortex",
#     dpi=300
# )
#
# print("\n" + "="*60)
# print("ГОТОВО! Все графики сохранены в diploma_figures/")
# print("="*60)