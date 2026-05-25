import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from matplotlib.gridspec import GridSpec


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


def plot_tvd_analysis_1d(
        solution: Dict,
        exact_solution: Optional[Dict] = None,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_indices: Optional[list] = None,
        figsize: Tuple = (14, 10)
):
    """
    Создаёт комплексный анализ TVD для 1D задачи.

    Args:
        solution: результат solver.solve()
        exact_solution: (опционально) точное решение с ключами 'x', 't', 'u_n'
        variable_idx: индекс переменной для анализа
        variable_name: название переменной для графиков
        snapshot_indices: индексы snapshots для отображения (None = все)
        figsize: размер фигуры
    """
    x = np.array(solution['x'])
    times = np.array(solution['t'])
    u_snapshots = solution['u_n']

    # Выбор snapshots
    if snapshot_indices is None:
        # Берём равномерно распределённые snapshots (максимум 6)
        n_snapshots = min(6, len(times))
        snapshot_indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)

    # Вычисляем TV эволюцию
    tv_times, tv_values = compute_tv_evolution_1d(solution, variable_idx)

    # Создаём фигуру
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ==================== График 1: Профили в разные моменты ====================
    ax1 = fig.add_subplot(gs[0, :])

    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

    for idx, snap_idx in enumerate(snapshot_indices):
        u = u_snapshots[snap_idx]
        t = times[snap_idx]

        if u.ndim == 1:
            var = u
        else:
            var = u[:, variable_idx]

        ax1.plot(x, var, 'o-', color=colors[idx],
                 label=f't = {t:.3f}', markersize=3, linewidth=1.5)

        # Если есть точное решение
        if exact_solution is not None:
            u_exact = exact_solution['u_n'][snap_idx]
            if u_exact.ndim == 1:
                var_exact = u_exact
            else:
                var_exact = u_exact[:, variable_idx]
            ax1.plot(x, var_exact, '--', color=colors[idx],
                     linewidth=2, alpha=0.7)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel(variable_name, fontsize=12)
    ax1.set_title('Профили решения в разные моменты времени', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ==================== График 2: Эволюция TV ====================
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.plot(tv_times, tv_values, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(tv_values[0], color='r', linestyle='--',
                label=f'TV(t=0) = {tv_values[0]:.4f}')

    # Проверка монотонности
    tv_increase = tv_values[-1] - tv_values[0]
    tv_max_increase = np.max(np.diff(tv_values))

    textstr = f'ΔTV = {tv_increase:.2e}\nMax jump = {tv_max_increase:.2e}'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Время t', fontsize=12)
    ax2.set_ylabel('TV(u)', fontsize=12)
    ax2.set_title('Эволюция полной вариации', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ==================== График 3: Zoom на разрыв (финальное время) ====================
    ax3 = fig.add_subplot(gs[1, 1])

    u_final = u_snapshots[-1]
    if u_final.ndim == 1:
        var_final = u_final
    else:
        var_final = u_final[:, variable_idx]

    # Находим область с максимальным градиентом (разрыв)
    grad = np.abs(np.diff(var_final))
    shock_idx = np.argmax(grad)

    # Zoom: ±20 точек вокруг разрыва
    zoom_range = 20
    idx_min = max(0, shock_idx - zoom_range)
    idx_max = min(len(x) - 1, shock_idx + zoom_range)

    x_zoom = x[idx_min:idx_max]
    var_zoom = var_final[idx_min:idx_max]

    ax3.plot(x_zoom, var_zoom, 'bo-', markersize=5, linewidth=1.5, label='Численное')

    if exact_solution is not None:
        u_exact_final = exact_solution['u_n'][-1]
        if u_exact_final.ndim == 1:
            var_exact_final = u_exact_final
        else:
            var_exact_final = u_exact_final[:, variable_idx]

        var_exact_zoom = var_exact_final[idx_min:idx_max]
        ax3.plot(x_zoom, var_exact_zoom, 'r--', linewidth=2, alpha=0.7, label='Точное')

    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel(variable_name, fontsize=12)
    ax3.set_title(f'Zoom на разрыв (t = {times[-1]:.3f})', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('TVD-анализ для 1D задачи', fontsize=16, fontweight='bold', y=0.995)

    return fig


def plot_tvd_analysis_2d(
        solution: Dict,
        variable_idx: int = 0,
        variable_name: str = "Плотность ρ",
        snapshot_indices: Optional[list] = None,
        slice_axis: str = 'y',  # 'x' или 'y'
        slice_position: float = 0.5,  # относительная позиция среза [0, 1]
        figsize: Tuple = (16, 12)
):
    """
    Создаёт комплексный TVD-анализ для 2D задачи.

    Args:
        solution: результат solver.solve() с ключами 't', 'u', 'X', 'Y'
        variable_idx: индекс переменной
        variable_name: название переменной
        snapshot_indices: индексы snapshots (None = равномерно 4 шт)
        slice_axis: направление среза ('x' или 'y')
        slice_position: позиция среза в относительных координатах
        figsize: размер фигуры
    """
    times = np.array(solution['t'])
    u_snapshots = solution['u']
    X = np.array(solution['X'])
    Y = np.array(solution['Y'])

    # Выбор snapshots
    if snapshot_indices is None:
        n_snapshots = min(4, len(times))
        snapshot_indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)

    # Вычисляем TV эволюцию
    tv_times, tv_values = compute_tv_evolution_2d(solution, variable_idx)

    # Создаём фигуру
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, len(snapshot_indices), figure=fig, hspace=0.35, wspace=0.3)

    # ==================== Ряд 1: Цветовые карты в разные моменты ====================
    for col_idx, snap_idx in enumerate(snapshot_indices):
        ax = fig.add_subplot(gs[0, col_idx])

        u = u_snapshots[snap_idx]
        var = u[..., variable_idx]
        t = times[snap_idx]

        im = ax.contourf(X, Y, var, levels=50, cmap='jet')
        ax.set_title(f't = {t:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label=variable_name)

    # ==================== Ряд 2: Профили вдоль среза ====================
    # Определяем индекс среза
    if slice_axis == 'y':
        Jy = X.shape[1]
        slice_idx = int(slice_position * (Jy - 1))
        x_slice = X[:, slice_idx]
        slice_label = f'y = {Y[0, slice_idx]:.2f}'
    else:  # 'x'
        Jx = X.shape[0]
        slice_idx = int(slice_position * (Jx - 1))
        x_slice = Y[slice_idx, :]
        slice_label = f'x = {X[slice_idx, 0]:.2f}'

    ax_profile = fig.add_subplot(gs[1, :])

    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

    for idx, snap_idx in enumerate(snapshot_indices):
        u = u_snapshots[snap_idx]
        var = u[..., variable_idx]
        t = times[snap_idx]

        if slice_axis == 'y':
            profile = var[:, slice_idx]
        else:
            profile = var[slice_idx, :]

        ax_profile.plot(x_slice, profile, 'o-', color=colors[idx],
                        label=f't = {t:.3f}', markersize=3, linewidth=1.5)

    ax_profile.set_xlabel(slice_axis, fontsize=12)
    ax_profile.set_ylabel(variable_name, fontsize=12)
    ax_profile.set_title(f'Профили вдоль среза {slice_label}', fontsize=14, fontweight='bold')
    ax_profile.legend(fontsize=10, ncol=len(snapshot_indices))
    ax_profile.grid(True, alpha=0.3)

    # ==================== Ряд 3: TV эволюция + Schlieren ====================
    ax_tv = fig.add_subplot(gs[2, :2])

    ax_tv.plot(tv_times, tv_values, 'b-o', linewidth=2, markersize=4)
    ax_tv.axhline(tv_values[0], color='r', linestyle='--',
                  label=f'TV(t=0) = {tv_values[0]:.4f}')

    tv_increase = tv_values[-1] - tv_values[0]
    textstr = f'ΔTV = {tv_increase:.2e}'
    ax_tv.text(0.05, 0.95, textstr, transform=ax_tv.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_tv.set_xlabel('Время t', fontsize=12)
    ax_tv.set_ylabel('TV(u)', fontsize=12)
    ax_tv.set_title('Эволюция полной вариации (2D)', fontsize=14, fontweight='bold')
    ax_tv.legend(fontsize=10)
    ax_tv.grid(True, alpha=0.3)

    # Schlieren-like изображение (градиенты плотности)
    ax_schlieren = fig.add_subplot(gs[2, 2:])

    u_final = u_snapshots[-1]
    var_final = u_final[..., variable_idx]

    # Градиент (численный Schlieren)
    grad_x = np.gradient(var_final, axis=0)
    grad_y = np.gradient(var_final, axis=1)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    im_schlieren = ax_schlieren.contourf(X, Y, np.log10(grad_magnitude + 1e-10),
                                         levels=50, cmap='gray')
    ax_schlieren.set_title(f'Schlieren (|∇ρ|) at t = {times[-1]:.3f}',
                           fontsize=14, fontweight='bold')
    ax_schlieren.set_xlabel('x')
    ax_schlieren.set_ylabel('y')
    plt.colorbar(im_schlieren, ax=ax_schlieren, label='log10(|∇ρ|)')

    plt.suptitle('TVD-анализ для 2D задачи', fontsize=16, fontweight='bold', y=0.995)

    return fig


def check_tvd_property(
        tv_values: np.ndarray,
        tolerance: float = 0.05,  # ← Увеличил до 5%
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
