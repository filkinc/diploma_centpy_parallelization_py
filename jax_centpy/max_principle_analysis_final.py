"""
max_principle_analysis.py

Инструменты для проверки принципа максимума (Maximum Principle) для 2D задач.

Принцип максимума: min(u^0) <= u^n <= max(u^0) для всех j, n.
Применяется вместо TVD для многомерных задач, где TVD недостижимо
для схем выше первого порядка (теорема Гудмана-Левека, 1985).

Использование:
    from max_principle_analysis import (
        compute_minmax_evolution_2d,
        check_max_principle,
        plot_max_principle_evolution_2d,
    )

    times, u_min, u_max = compute_minmax_evolution_2d(sol_2d, variable_idx=0)
    is_ok, diag = check_max_principle(times, u_min, u_max)
    plot_max_principle_evolution_2d(sol_2d, variable_idx=0, show=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (регистрирует стиль 'science')
from pathlib import Path
from typing import Dict, Tuple, Optional

# Единый стиль оформления графиков (как у остальных рисунков работы)
_SCIENCE_STYLE = ['science', 'no-latex']
_MAX_COLOR = '#0C5DA5'  # u_max(t) — синий
_MIN_COLOR = '#C62828'  # u_min(t) — красный


def compute_minmax_evolution_2d(
    solution: Dict,
    variable_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет эволюцию u_min(t) и u_max(t) по всем временным слоям (2D).

    Параметры
    ----------
    solution     : словарь solver.solve() с ключами 't', 'u'
    variable_idx : индекс компоненты (0 = скаляр или первая переменная)

    Возвращает
    ----------
    times  : (n_snapshots,) — моменты времени
    u_min  : (n_snapshots,) — минимум u по всей области
    u_max  : (n_snapshots,) — максимум u по всей области
    """
    times = np.array(solution["t"])
    snapshots = solution["u"]  # (n_snapshots, Jx, Jy, n_comp)

    u_min_arr = []
    u_max_arr = []

    for u in snapshots:
        var = u[..., variable_idx]  # (Jx, Jy)
        u_min_arr.append(float(np.min(var)))
        u_max_arr.append(float(np.max(var)))

    return times, np.array(u_min_arr), np.array(u_max_arr)


def check_max_principle(
    times: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    tolerance: float = 1e-10,
) -> Tuple[bool, Dict]:
    """
    Проверяет выполнение принципа максимума:
        u_min(0) - tol <= u_min(t) для всех t
        u_max(t) <= u_max(0) + tol для всех t

    Параметры
    ----------
    times     : массив времён
    u_min     : массив минимумов u(t)
    u_max     : массив максимумов u(t)
    tolerance : числовой допуск (по умолчанию 1e-10)

    Возвращает
    ----------
    is_ok        : True если принцип максимума выполнен
    diagnostics  : словарь с диагностикой
    """
    u_min_init = u_min[0]
    u_max_init = u_max[0]

    # Нарушения снизу: u_min стало меньше начального
    min_violations = np.sum(u_min < u_min_init - tolerance)
    # Нарушения сверху: u_max превысило начальный максимум
    max_violations = np.sum(u_max > u_max_init + tolerance)

    total_violations = min_violations + max_violations
    is_ok = total_violations == 0

    # Максимальное нарушение
    max_undershoot = float(np.max(np.maximum(0.0, u_min_init - u_min)))
    max_overshoot  = float(np.max(np.maximum(0.0, u_max - u_max_init)))

    diag = {
        "u_min_initial"    : u_min_init,
        "u_max_initial"    : u_max_init,
        "u_min_final"      : u_min[-1],
        "u_max_final"      : u_max[-1],
        "max_undershoot"   : max_undershoot,
        "max_overshoot"    : max_overshoot,
        "min_violations"   : int(min_violations),
        "max_violations"   : int(max_violations),
        "total_violations" : int(total_violations),
        "is_satisfied"     : is_ok,
        "verdict"          : _get_verdict(is_ok, max_undershoot, max_overshoot),
    }

    return is_ok, diag


def _get_verdict(is_ok, undershoot, overshoot):
    if is_ok:
        if undershoot < 1e-12 and overshoot < 1e-12:
            return "✓ Отлично: принцип максимума строго выполнен"
        else:
            return "✓ Хорошо: принцип максимума выполнен в пределах допуска"
    else:
        parts = []
        if undershoot > 0:
            parts.append(f"undershoot={undershoot:.2e}")
        if overshoot > 0:
            parts.append(f"overshoot={overshoot:.2e}")
        return "✗ Нарушение принципа максимума: " + ", ".join(parts)


def plot_max_principle_evolution_2d(
    solution: Dict,
    variable_idx: int = 0,
    figsize: Tuple = (6, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    График эволюции u_min(t) и u_max(t) для проверки принципа максимума
    в стиле scienceplots (единообразно с остальными рисунками работы).

    На графике:
      - синяя линия   : u_max(t)
      - красная линия : u_min(t)
      - пунктиры      : начальные значения u_max(0) и u_min(0)

    Параметры
    ----------
    solution      : словарь solver.solve() для 2D задачи
    variable_idx  : индекс компоненты (0 = скаляр или первая переменная)
    figsize       : размер фигуры
    show          : True — показать на экране
    save_path     : путь для сохранения (None — не сохранять)
    dpi           : разрешение
    """
    times, u_min, u_max = compute_minmax_evolution_2d(solution, variable_idx)

    with plt.style.context(_SCIENCE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(times, u_max, color=_MAX_COLOR, linewidth=1.8, label=r"$u_{\max}(t)$")
        ax.plot(times, u_min, color=_MIN_COLOR, linewidth=1.8, label=r"$u_{\min}(t)$")

        # Пунктиры — начальные значения
        ax.axhline(u_max[0], color=_MAX_COLOR, linestyle="--", linewidth=1.2,
                   label=rf"$u_{{\max}}(0) = {u_max[0]:.4f}$")
        ax.axhline(u_min[0], color=_MIN_COLOR, linestyle="--", linewidth=1.2,
                   label=rf"$u_{{\min}}(0) = {u_min[0]:.4f}$")

        ax.set_xlabel(r"Время $t$", fontsize=18)
        ax.set_ylabel(r"$u$", fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        ax.legend(fontsize=12, loc="center right")

        fig.tight_layout()

        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"✓ Сохранено: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def print_max_principle_diagnostics(diag: Dict):
    """Выводит диагностику принципа максимума в консоль."""
    print("=" * 60)
    print("ДИАГНОСТИКА ПРИНЦИПА МАКСИМУМА (2D)")
    print("=" * 60)
    print(f"u_min начальное:    {diag['u_min_initial']:.6f}")
    print(f"u_max начальное:    {diag['u_max_initial']:.6f}")
    print(f"u_min финальное:    {diag['u_min_final']:.6f}")
    print(f"u_max финальное:    {diag['u_max_final']:.6f}")
    print(f"Макс. undershoot:   {diag['max_undershoot']:.4e}")
    print(f"Макс. overshoot:    {diag['max_overshoot']:.4e}")
    print(f"Нарушений снизу:    {diag['min_violations']}")
    print(f"Нарушений сверху:   {diag['max_violations']}")
    print("-" * 60)
    print(f"ВЕРДИКТ: {diag['verdict']}")
    print(f"Принцип максимума: {'✓ ДА' if diag['is_satisfied'] else '✗ НЕТ'}")
    print("=" * 60)
