import numpy as np


def restrict_1d(u_fine):
    """
    Ограничение (restriction) решения с мелкой сетки на крупную для 1D.
    Усредняет каждые 2 соседние ячейки (коэффициент измельчения r=2).
    """
    return (u_fine[0::2] + u_fine[1::2]) / 2.0


def restrict_2d(u_fine):
    """
    Ограничение (restriction) решения с мелкой сетки на крупную для 2D.
    Усредняет блоки 2x2 ячейки (коэффициент измельчения r=2).
    Предполагается, что сетка равномерна по x и y.
    """
    return (u_fine[0::2, 0::2] + u_fine[1::2, 0::2] +
            u_fine[0::2, 1::2] + u_fine[1::2, 1::2]) / 4.0


def compute_errors_and_order(u_coarse, u_medium, u_fine, r=2.0, norm_type='L1'):
    """
    Вычисляет ошибки между сетками, наблюдаемый порядок точности (p)
    и экстраполированное по Ричардсону решение на крупной сетке.

    Параметры:
    - u_coarse: решение на крупной сетке (NxN)
    - u_medium: решение на средней сетке (2Nx2N)
    - u_fine: решение на мелкой сетке (4Nx4N)
    - r: коэффициент измельчения сетки (по умолчанию 2.0)
    - norm_type: тип нормы ('L1', 'L2', 'Linf')
    """
    ndim = u_coarse.ndim
    is_system = u_coarse.shape[-1] > 1 if ndim > 1 or (ndim == 1 and u_coarse.shape[0] != u_coarse.size) else False

    # Проекция на крупную сетку
    if ndim == 1 or (ndim == 2 and is_system):
        # Логика для 1D
        u_medium_rest = restrict_1d(u_medium)
        u_fine_rest = restrict_1d(restrict_1d(u_fine))
        spatial_axes = (0,)
    else:
        # Логика для 2D
        u_medium_rest = restrict_2d(u_medium)
        u_fine_rest = restrict_2d(restrict_2d(u_fine))
        spatial_axes = (0, 1)

    # Разности между сетками, спроецированными на одинаковую размерность
    diff_mc = u_medium_rest - u_coarse
    diff_fm = u_fine_rest - u_medium_rest

    num_cells = np.prod([u_coarse.shape[i] for i in spatial_axes])

    # Вычисление выбранной нормы
    if norm_type == 'L1':
        error_mc = np.sum(np.abs(diff_mc), axis=spatial_axes) / num_cells
        error_fm = np.sum(np.abs(diff_fm), axis=spatial_axes) / num_cells
    elif norm_type == 'L2':
        error_mc = np.sqrt(np.sum(diff_mc ** 2, axis=spatial_axes) / num_cells)
        error_fm = np.sqrt(np.sum(diff_fm ** 2, axis=spatial_axes) / num_cells)
    elif norm_type == 'Linf':
        error_mc = np.max(np.abs(diff_mc), axis=spatial_axes)
        error_fm = np.max(np.abs(diff_fm), axis=spatial_axes)
    else:
        raise ValueError(f"Неподдерживаемый тип нормы: {norm_type}")

    # Защита от деления на ноль
    epsilon = 1e-15
    error_mc_safe = np.maximum(error_mc, epsilon)
    error_fm_safe = np.maximum(error_fm, epsilon)

    # Порядок сходимости Ричардсона: p = ln(E_{coarse-medium} / E_{medium-fine}) / ln(r)
    p = np.log(error_mc_safe / error_fm_safe) / np.log(r)
    p_safe = np.maximum(p, 1e-5)

    if is_system:
        p_expand = p_safe if ndim == 1 else p_safe[np.newaxis, np.newaxis, :]
        factor = r ** p_expand - 1.0
    else:
        factor = r ** p_safe - 1.0

    u_extrapolated = u_fine_rest + diff_fm / factor

    return p, error_mc, error_fm, u_extrapolated


def self_convergence_analysis(u_coarse, u_medium, u_fine, r=2.0, var_names=None):
    """Обертка для удобного вывода результатов по всем нормам."""
    results = {}

    for norm in ['L1', 'L2', 'Linf']:
        p, err_mc, err_fm, _ = compute_errors_and_order(u_coarse, u_medium, u_fine, r, norm_type=norm)
        results[norm] = {'order': p, 'error_coarse_medium': err_mc, 'error_medium_fine': err_fm}

    print("-" * 50)
    print("Анализ самосходимости (Экстраполяция Ричардсона)")
    print(f"Коэффициент измельчения (r) = {r}")
    print("-" * 50)

    is_system = u_coarse.shape[-1] > 1 if u_coarse.ndim > 1 or (
                u_coarse.ndim == 1 and u_coarse.shape[0] != u_coarse.size) else False
    num_vars = u_coarse.shape[-1] if is_system else 1

    if var_names is None:
        var_names = [f"Var {i}" for i in range(num_vars)] if is_system else ["Скаляр"]

    for i, var in enumerate(var_names):
        print(f"\n--- {var} ---")
        for norm in ['L1', 'L2', 'Linf']:
            p_val = results[norm]['order'][i] if is_system else results[norm]['order']
            err_mc_val = results[norm]['error_coarse_medium'][i] if is_system else results[norm]['error_coarse_medium']
            err_fm_val = results[norm]['error_medium_fine'][i] if is_system else results[norm]['error_medium_fine']
            print(f"Норма {norm}:")
            print(f"  Ошибка (Крупная-Средняя): {err_mc_val:.4e}")
            print(f"  Ошибка (Средняя-Мелкая):  {err_fm_val:.4e}")
            print(f"  Наблюдаемый порядок (p):  {p_val:.4f}")

    print("-" * 50)
    return results
