"""
tvd_analysis.py
Инструменты для проверки TVD-свойства численных схем.
Каждый график сохраняется отдельно для удобства вставки в диплом.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pathlib import Path


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

    Args:
        solution: результат solver.solve()
        exact_solution: (опционально) точное решение
        variable_idx: индекс переменной для анализа
        variable_name: название переменной
        snapshot_indices: индексы snapshots (None = автовыбор)
        output_dir: папка для сохранения графиков
        prefix: префикс имён файлов
        figsize_profiles: размер графика профилей
        figsize_tv: размер графика TV(t)
        figsize_zoom: размер графика zoom
        dpi: разрешение сохраняемых изображений
        show_points: показывать точки на графиках
        point_step: шаг для отображения точек (каждая N-я точка)

    Returns:
        paths: словарь с путями к сохранённым файлам
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

    ax2.set_xlabel('Время t', fontsize=14)
    ax2.set_ylabel('TV(u)', fontsize=14)
    ax2.set_title('Эволюция полной вариации', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
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


def plot_tvd_analysis_2d_separate(
        solution: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_indices: Optional[list] = None,
        slice_axis: str = 'y',
        slice_position: float = 0.5,
        output_dir: str = "tvd_figures",
        prefix: str = "tvd_2d",
        figsize_contour: Tuple = (8, 6),
        figsize_profiles: Tuple = (12, 7),
        figsize_tv: Tuple = (10, 6),
        figsize_schlieren: Tuple = (8, 6),
        dpi: int = 150,
        show_points: bool = True,
        point_step: int = 5
):
    """
    Создаёт TVD-анализ для 2D задачи с сохранением каждого графика отдельно.

    Args:
        solution: результат solver.solve()
        variable_idx: индекс переменной
        variable_name: название переменной
        snapshot_indices: индексы snapshots (None = автовыбор 4 шт)
        slice_axis: направление среза ('x' или 'y')
        slice_position: позиция среза в относительных координатах [0, 1]
        output_dir: папка для сохранения
        prefix: префикс файлов
        figsize_*: размеры графиков
        dpi: разрешение
        show_points: показывать точки
        point_step: шаг точек

    Returns:
        paths: словарь с путями к файлам
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    times = np.array(solution['t'])
    u_snapshots = solution['u']
    X = np.array(solution['X'])
    Y = np.array(solution['Y'])

    # Выбор snapshots
    if snapshot_indices is None:
        n_snapshots = min(4, len(times))
        snapshot_indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)

    # TV эволюция
    tv_times, tv_values = compute_tv_evolution_2d(solution, variable_idx)

    saved_files = {}

    # ==================== Цветовые карты для каждого момента ====================
    for col_idx, snap_idx in enumerate(snapshot_indices):
        fig, ax = plt.subplots(figsize=figsize_contour)

        u = u_snapshots[snap_idx]
        var = u[..., variable_idx]
        t = times[snap_idx]

        im = ax.contourf(X, Y, var, levels=50, cmap='jet')
        ax.set_title(f'{variable_name} при t = {t:.3f}',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        plt.colorbar(im, ax=ax, label=variable_name)

        file_contour = output_path / f"{prefix}_contour_t{col_idx:02d}.png"
        fig.savefig(file_contour, dpi=dpi, bbox_inches='tight')
        saved_files[f'contour_t{col_idx}'] = str(file_contour)
        plt.close(fig)
        print(f"✓ Сохранено: {file_contour}")

    # ==================== Профили вдоль среза ====================
    if slice_axis == 'y':
        Jy = X.shape[1]
        slice_idx = int(slice_position * (Jy - 1))
        x_slice = X[:, slice_idx]
        slice_label = f'y = {Y[0, slice_idx]:.2f}'
    else:
        Jx = X.shape[0]
        slice_idx = int(slice_position * (Jx - 1))
        x_slice = Y[slice_idx, :]
        slice_label = f'x = {X[slice_idx, 0]:.2f}'

    fig_profile, ax_profile = plt.subplots(figsize=figsize_profiles)

    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))
    line_style = 'o-' if show_points else '-'
    markersize = 3 if show_points else 0
    markevery = point_step if show_points else None

    for idx, snap_idx in enumerate(snapshot_indices):
        u = u_snapshots[snap_idx]
        var = u[..., variable_idx]
        t = times[snap_idx]

        if slice_axis == 'y':
            profile = var[:, slice_idx]
        else:
            profile = var[slice_idx, :]

        ax_profile.plot(x_slice, profile, line_style, color=colors[idx],
                        label=f't = {t:.3f}', markersize=markersize,
                        linewidth=1.5, markevery=markevery)

    ax_profile.set_xlabel(slice_axis, fontsize=14)
    ax_profile.set_ylabel(variable_name, fontsize=14)
    ax_profile.set_title(f'Профили вдоль среза {slice_label}',
                         fontsize=16, fontweight='bold')
    ax_profile.legend(fontsize=11, ncol=2)
    ax_profile.grid(True, alpha=0.3)

    file_profile = output_path / f"{prefix}_profiles.png"
    fig_profile.savefig(file_profile, dpi=dpi, bbox_inches='tight')
    saved_files['profiles'] = str(file_profile)
    plt.close(fig_profile)
    print(f"✓ Сохранено: {file_profile}")

    # ==================== TV эволюция ====================
    fig_tv, ax_tv = plt.subplots(figsize=figsize_tv)

    ax_tv.plot(tv_times, tv_values, 'b-o', linewidth=2, markersize=4)
    ax_tv.axhline(tv_values[0], color='r', linestyle='--', linewidth=2,
                  label=f'TV(t=0) = {tv_values[0]:.4f}')

    tv_increase = tv_values[-1] - tv_values[0]
    textstr = f'ΔTV = {tv_increase:.2e}'
    ax_tv.text(0.05, 0.95, textstr, transform=ax_tv.transAxes,
               verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_tv.set_xlabel('Время t', fontsize=14)
    ax_tv.set_ylabel('TV(u)', fontsize=14)
    ax_tv.set_title('Эволюция полной вариации (2D)',
                    fontsize=16, fontweight='bold')
    ax_tv.legend(fontsize=12)
    ax_tv.grid(True, alpha=0.3)

    file_tv = output_path / f"{prefix}_tv_evolution.png"
    fig_tv.savefig(file_tv, dpi=dpi, bbox_inches='tight')
    saved_files['tv_evolution'] = str(file_tv)
    plt.close(fig_tv)
    print(f"✓ Сохранено: {file_tv}")

    # ==================== Schlieren ====================
    fig_schlieren, ax_schlieren = plt.subplots(figsize=figsize_schlieren)

    u_final = u_snapshots[-1]
    var_final = u_final[..., variable_idx]

    grad_x = np.gradient(var_final, axis=0)
    grad_y = np.gradient(var_final, axis=1)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    im_schlieren = ax_schlieren.contourf(X, Y, np.log10(grad_magnitude + 1e-10),
                                         levels=50, cmap='gray')
    ax_schlieren.set_title(f'Schlieren (|∇ρ|) при t = {times[-1]:.3f}',
                           fontsize=16, fontweight='bold')
    ax_schlieren.set_xlabel('x', fontsize=14)
    ax_schlieren.set_ylabel('y', fontsize=14)
    plt.colorbar(im_schlieren, ax=ax_schlieren, label='log10(|∇ρ|)')

    file_schlieren = output_path / f"{prefix}_schlieren.png"
    fig_schlieren.savefig(file_schlieren, dpi=dpi, bbox_inches='tight')
    saved_files['schlieren'] = str(file_schlieren)
    plt.close(fig_schlieren)
    print(f"✓ Сохранено: {file_schlieren}")

    return saved_files


# Совместимость с предыдущим интерфейсом
def plot_tvd_analysis_1d(*args, **kwargs):
    """Обёртка для обратной совместимости (возвращает старый формат)."""
    return plot_tvd_analysis_1d_separate(*args, **kwargs)


def plot_tvd_analysis_2d(*args, **kwargs):
    """Обёртка для обратной совместимости."""
    return plot_tvd_analysis_2d_separate(*args, **kwargs)
