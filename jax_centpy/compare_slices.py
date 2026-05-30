import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_trajectories(coords_jax, coords_centpy):
    """
    Отрисовывает графики с высококонтрастными цветами,
    пунктирными проекциями на оси и точными подписями координат.
    """
    x1, y1 = zip(*coords_jax)
    x2, y2 = zip(*coords_centpy)

    # Увеличим высоту графика, чтобы значения на оси Y не слипались
    plt.figure(figsize=(10, 8))

    color_jax = '#FF0000'  # Чистый красный
    color_centpy = '#0000FF'  # Чистый синий

    # Отрисовываем основные линии
    plt.plot(x1, y1, marker='o', markersize=8, linestyle='-', linewidth=3,
             color=color_jax, label='JAX', zorder=5)
    plt.plot(x2, y2, marker='s', markersize=8, linestyle='-', linewidth=3,
             color=color_centpy, label='centpy', zorder=5)

    # Рисуем пунктирные линии (проекции) для JAX
    for x, y in zip(x1, y1):
        # Вертикальная линия к оси X
        plt.plot([x, x], [0, y], color=color_jax, linestyle='--', linewidth=1.5, alpha=0.7)
        # Горизонтальная линия к оси Y
        plt.plot([0, x], [y, y], color=color_jax, linestyle='--', linewidth=1.5, alpha=0.7)

    # Рисуем пунктирные линии (проекции) для centpy
    for x, y in zip(x2, y2):
        plt.plot([x, x], [0, y], color=color_centpy, linestyle='--', linewidth=1.5, alpha=0.7)
        plt.plot([0, x], [y, y], color=color_centpy, linestyle='--', linewidth=1.5, alpha=0.7)

    # Собираем все уникальные значения координат, чтобы сделать из них подписи на осях
    all_x = sorted(list(set(x1 + x2)))
    all_y = sorted(list(set(y1 + y2)))

    # Принудительно устанавливаем эти значения как засечки на осях
    plt.xticks(all_x, fontsize=16)
    plt.yticks(all_y, fontsize=16)

    # Чтобы пунктиры ровно упирались в оси, начинаем оси строго с нуля
    plt.xlim(left=0, right=max(all_x) * 1.1)
    plt.ylim(bottom=0, top=max(all_y) * 1.1)

    # Оформление
    #plt.title('Сравнение траекторий JAX и centpy', fontsize=14, fontweight='bold')
    plt.xlabel('Плотность сетки', fontsize=16)
    plt.ylabel('Время работы solver сек.', fontsize=16)

    # Отключаем стандартную сетку, так как теперь есть цветные проекции
    plt.grid(True, linestyle='--', alpha=0.7)

    # Легенда
    plt.legend(fontsize=12, loc='upper left', framealpha=1.0, edgecolor='black')

    plt.tight_layout()
    plt.show()


def plot_trajectories_v2(coords_jax, coords_centpy):
    x1, y1 = zip(*coords_jax)
    x2, y2 = zip(*coords_centpy)

    # Меняем соотношение сторон: делаем график очень высоким (8 в ширину, 14 в высоту)
    # Это растянет ось Y и "приподнимет" линию JAX над нулем
    plt.figure(figsize=(8, 14))

    color_jax = '#FF0000'
    color_centpy = '#0000FF'

    # Отрисовываем всё на одной оси
    plt.plot(x1, y1, marker='o', markersize=8, linestyle='-', linewidth=3,
             color=color_jax, label='JAX', zorder=5)
    plt.plot(x2, y2, marker='s', markersize=8, linestyle='-', linewidth=3,
             color=color_centpy, label='centpy', zorder=5)

    # Пунктирные линии для JAX
    for x, y in zip(x1, y1):
        plt.plot([x, x], [0, y], color=color_jax, linestyle='--', linewidth=1.5, alpha=0.7)
        plt.plot([0, x], [y, y], color=color_jax, linestyle='--', linewidth=1.5, alpha=0.7)

    # Пунктирные линии для centpy
    for x, y in zip(x2, y2):
        plt.plot([x, x], [0, y], color=color_centpy, linestyle='--', linewidth=1.5, alpha=0.7)
        plt.plot([0, x], [y, y], color=color_centpy, linestyle='--', linewidth=1.5, alpha=0.7)

    all_x = sorted(list(set(x1 + x2)))
    all_y = sorted(list(set(y1 + y2)))

    plt.xticks(all_x, fontsize=11, fontweight='bold')
    # Устанавливаем все значения на одну ось
    plt.yticks(all_y, fontsize=10, fontweight='bold')

    plt.xlim(left=0, right=max(all_x) * 1.1)
    plt.ylim(bottom=0, top=max(all_y) * 1.05)

    plt.xlabel('Плотность сетки', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Время работы solver (сек.)', fontsize=12, fontweight='bold', labelpad=10)

    plt.legend(fontsize=12, loc='upper left', framealpha=1.0, edgecolor='black')

    # Убираем рамки сверху и справа для более чистого вида
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_trajectories_v3(coords1, coords2, coords3, coords4, coords5, labels=['JAX (CPU)', 'JAX (GPU)', 'centpy', 'JAX-FLUIDS (GPU)', 'JAX-FLUIDS (CPU)']):
    """
    Отрисовывает графики для 4 наборов данных с высококонтрастными цветами,
    пунктирными проекциями на оси и логарифмическим масштабом.
    """
    # Распаковываем координаты всех 4 массивов
    x1, y1 = zip(*coords1)
    x2, y2 = zip(*coords2)
    x3, y3 = zip(*coords3)
    x4, y4 = zip(*coords4)
    x5, y5 = zip(*coords5)

    plt.figure(figsize=(10, 8))

    # Задаем контрастные цвета и разные маркеры для 4 графиков
    colors = ['#FF0000', '#0000FF', '#008000', '#FFA500', '#FF00FF']  # Красный, Синий, Зеленый, Оранжевый
    markers = ['o', 's', '^', 'D', '*']  # Круг, Квадрат, Треугольник, Ромб

    # Собираем все уникальные значения координат для засечек
    all_x = sorted(list(set(x1 + x2 + x3 + x4 + x5)))
    all_y = sorted(list(set(y1 + y2 + y3 + y4 + y5)))

    # Задаем минимальные границы осей (в логарифмическом масштабе нельзя использовать 0)
    # Берем значение чуть меньше самого минимального в данных, чтобы пунктиры упирались в край графика
    x_min_limit = min(all_x) * 0.8
    y_min_limit = min(all_y) * 0.8

    # Объединяем данные для удобного прохода циклом
    datasets = [
        (x1, y1, colors[0], markers[0], labels[0]),
        (x2, y2, colors[1], markers[1], labels[1]),
        (x3, y3, colors[2], markers[2], labels[2]),
        (x4, y4, colors[3], markers[3], labels[3]),
        (x5, y5, colors[4], markers[4], labels[4]),
    ]

    # Отрисовываем основные линии и их проекции
    for x, y, color, marker, label in datasets:
        # Основная линия
        plt.plot(x, y, marker=marker, markersize=8, linestyle='-', linewidth=3,
                 color=color, label=label, zorder=5)

        # Рисуем пунктирные линии (проекции)
        for xi, yi in zip(x, y):
            # Вертикальная линия к оси X (идет до x_min_limit вместо 0)
            plt.plot([xi, xi], [y_min_limit, yi], color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            # Горизонтальная линия к оси Y (идет до y_min_limit вместо 0)
            plt.plot([x_min_limit, xi], [yi, yi], color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    # Включаем логарифмический масштаб
    plt.xscale('log')
    plt.yscale('log')

    # Форматируем оси, чтобы отключить научный формат (10^x) и вернуть обычные числа
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Принудительно устанавливаем уникальные значения как засечки
    # Шрифт немного уменьшен до 12, т.к. из 4 массивов соберется много уникальных чисел
    plt.xticks(all_x, fontsize=22)
    plt.yticks(all_y, fontsize=22)

    # Устанавливаем границы осей
    plt.xlim(left=x_min_limit, right=max(all_x) * 1.2)
    plt.ylim(bottom=y_min_limit, top=max(all_y) * 1.2)

    # Оформление
    plt.xlabel('Плотность сетки', fontsize=22)
    plt.ylabel('Время работы solver сек.', fontsize=22)

    # Делаем обычную сетку более бледной, чтобы пунктирные проекции выделялись лучше
    plt.grid(True, linestyle=':', alpha=0.4)

    plt.legend(fontsize=18, loc='upper left', framealpha=1.0, edgecolor='black')

    plt.tight_layout()
    plt.show()


import scienceplots  # Обязательный импорт для работы стилей


# def plot_trajectories_v4(coords1, coords2, coords3, coords4, coords5,
#                          labels=['JAX (CPU)', 'JAX (GPU)', 'centpy', 'JAX-FLUIDS (GPU)', 'JAX-FLUIDS (CPU)']):
#     """
#     Отрисовывает графики для 5 наборов данных в строгом академическом стиле.
#     Оптимизировано для печати на бумаге формата А4 в дипломной работе.
#     """
#     # Применяем контекст стилей:
#     # 'science' - базовый стиль для научных статей
#     # 'high-vis' - высококонтрастная палитра, которая отлично читается при цветной и ч/б печати
#     # 'grid' - добавляет аккуратную сетку
#     # 'russian-font' - включает поддержку кириллицы (требует установленного LaTeX).
#     # ВНИМАНИЕ: Если LaTeX не установлен, используйте ['science', 'no-latex', 'high-vis', 'grid']
#     with plt.style.context(['science', 'high-vis', 'grid', 'russian-font']):
#
#         # Размер 7x5 дюймов оптимален для вставки в текстовый документ без потери качества
#         fig, ax = plt.subplots(figsize=(7, 5))
#
#         # Распаковываем координаты
#         x1, y1 = zip(*coords1)
#         x2, y2 = zip(*coords2)
#         x3, y3 = zip(*coords3)
#         x4, y4 = zip(*coords4)
#         x5, y5 = zip(*coords5)
#
#         # Собираем все уникальные значения координат для засечек
#         all_x = sorted(list(set(x1 + x2 + x3 + x4 + x5)))
#         all_y = sorted(list(set(y1 + y2 + y3 + y4 + y5)))
#
#         x_min_limit = min(all_x) * 0.8
#         y_min_limit = min(all_y) * 0.8
#
#         datasets = [
#             (x1, y1, labels[0]),
#             (x2, y2, labels[1]),
#             (x3, y3, labels[2]),
#             (x4, y4, labels[3]),
#             (x5, y5, labels[4]),
#         ]
#
#         # Маркеры оставляем разными для черно-белой печати
#         markers = ['o', 's', '^', 'D', 'v']
#
#         # Отрисовываем основные линии и их проекции
#         for i, (x, y, label) in enumerate(datasets):
#             # Цвета подбираются автоматически из палитры high-vis
#             line = ax.plot(x, y, marker=markers[i], markersize=6, linestyle='-',
#                            linewidth=1.5, label=label, zorder=5)
#
#             # Извлекаем цвет текущей линии для окраски пунктиров
#             color = line[0].get_color()
#
#             # Рисуем пунктирные линии (проекции)
#             for xi, yi in zip(x, y):
#                 ax.plot([xi, xi], [y_min_limit, yi], color=color, linestyle='--', linewidth=1, alpha=0.6)
#                 ax.plot([x_min_limit, xi], [yi, yi], color=color, linestyle='--', linewidth=1, alpha=0.6)
#
#         # Включаем логарифмический масштаб
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#
#         # Отключаем научный формат (10^x), возвращаем обычные числа
#         ax.xaxis.set_major_formatter(ScalarFormatter())
#         ax.yaxis.set_major_formatter(ScalarFormatter())
#
#         # Принудительно устанавливаем уникальные значения как засечки
#         # SciencePlots сам подберет гармоничный размер шрифта
#         ax.set_xticks(all_x)
#         ax.set_yticks(all_y)
#
#         # Устанавливаем границы осей
#         ax.set_xlim(left=x_min_limit, right=max(all_x) * 1.2)
#         ax.set_ylim(bottom=y_min_limit, top=max(all_y) * 1.2)
#
#         # Подписи осей
#         ax.set_xlabel('Плотность сетки')
#         ax.set_ylabel('Время работы solver, сек.')
#
#         # Легенда: включаем рамку, чтобы она не сливалась с сеткой
#         ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
#
#         # Убираем лишние отступы для аккуратного экспорта
#         plt.tight_layout()
#         plt.show()


def plot_trajectories_v5(coords1, coords2, coords3, coords4, coords5,
                         labels=['JAX (CPU)', 'JAX (GPU)', 'centpy', 'JAX-FLUIDS (GPU)', 'JAX-FLUIDS (CPU)']):
    """
    Отрисовывает графики с приглушенной цветовой палитрой (muted)
    и увеличенными размерами шрифтов для дипломной работы.
    """
    # Задаем стиль. 'muted' - приглушенные, не раздражающие цвета.
    # Если 'russian-font' выдает ошибку, оставьте ['science', 'muted', 'grid']
    with plt.style.context(['science', 'muted', 'grid', 'russian-font']):

        # --- НАСТРОЙКА РАЗМЕРА ШРИФТОВ ---
        # Переопределяем параметры поверх стиля science
        plt.rcParams.update({
            'axes.labelsize': 20,  # Размер подписей осей (Плотность сетки, Время работы)
            'xtick.labelsize': 20,  # Размер цифр на оси X
            'ytick.labelsize': 18,  # Размер цифр на оси Y
            'legend.fontsize': 18,  # Размер текста в легенде
        })

        # Размер графика можно сделать чуть больше (например, 8x6)
        fig, ax = plt.subplots(figsize=(8, 6))

        x1, y1 = zip(*coords1)
        x2, y2 = zip(*coords2)
        x3, y3 = zip(*coords3)
        x4, y4 = zip(*coords4)
        x5, y5 = zip(*coords5)

        all_x = sorted(list(set(x1 + x2 + x3 + x4 + x5)))
        all_y = sorted(list(set(y1 + y2 + y3 + y4 + y5)))

        x_min_limit = min(all_x) * 0.8
        y_min_limit = min(all_y) * 0.8

        datasets = [
            (x1, y1, labels[0]),
            (x2, y2, labels[1]),
            (x3, y3, labels[2]),
            (x4, y4, labels[3]),
            (x5, y5, labels[4]),
        ]

        markers = ['o', 's', '^', 'D', 'v']

        # Отрисовка
        for i, (x, y, label) in enumerate(datasets):
            line = ax.plot(x, y, marker=markers[i], markersize=7, linestyle='-',
                           linewidth=2, label=label, zorder=5)

            # Автоматически берется цвет из палитры 'muted'
            color = line[0].get_color()

            # Пунктирные проекции (делаем их чуть тоньше и прозрачнее)
            for xi, yi in zip(x, y):
                ax.plot([xi, xi], [y_min_limit, yi], color=color, linestyle='--', linewidth=1, alpha=0.5)
                ax.plot([x_min_limit, xi], [yi, yi], color=color, linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())

        # Цифры на осях (теперь они будут 14-го размера, как мы задали в rcParams)
        ax.set_xticks(all_x)
        ax.set_yticks(all_y)

        ax.set_xlim(left=x_min_limit, right=max(all_x) * 1.2)
        ax.set_ylim(bottom=y_min_limit, top=max(all_y) * 1.2)

        # Подписи осей (теперь они 16-го размера)
        ax.set_xlabel('Плотность сетки')
        ax.set_ylabel('Время работы solver, сек.')

        # Легенда
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    #compare_all_slices('centpy_data.npz', 'gpu_data.npz')

#========================================================
# 1D
# ========================================================

    # JAX_FLUIDS для 1d

    jax_fluids_gpu_1d = [
        [256, 3.8937],
        [512, 7.5451],
        [1024, 16.4025],
        [2048, 33.4514],
        [4096, 70.4434],
        [8192, 153.6304],
        [16384, 404.5478]
    ]

    jax_fluids_cpu_1d = [
        [256, 0.954424],
        [512, 2.856985],
        [1024, 4.892743],
        [2048, 13.040001],
        [4096, 44.687992],
        [8192, 161.555818],
        [16384, 633.203686]
    ]


    # centpy SD2 для 1d

    centpy_sd2_1d = [
        [256, 0.4194],
        [512, 0.9339],
        [1024, 2.4237],
        [2048, 7.8631],
        [4096, 22.9333],
        [8192, 85.1810],
        [16384, 430.9722]
    ]

    # centpy FD2 для 1d

    centpy_fd2_1d = [
        [256, 0.4194],
        [512, 0.9339],
        [1024, 2.4237],
        [2048, 7.8631],
        [4096, 22.9333],
        [8192, 85.1810],
        [16384, 430.9722]
    ]

    # JAX SD2 для 1d

    jax_cpu_sd2_1d = [
        [256, 0.0958],
        [512, 0.3935],
        [1024, 2.4990],
        [2048, 7.2545],
        [4096, 29.9283],
        [8192, 86.8622],
        [16384, 349.8919]
    ]

    jax_gpu_sd2_1d = [
        [256, 0.143029],
        [512, 0.178997],
        [1024, 0.370171],
        [2048, 0.720430],
        [4096, 1.500359],
        [8192, 3.930583],
        [16384, 12.804595]
    ]

    # JAX FD2 для 1d

    jax_cpu_fd2_1d = [
        [256, 0.023216],
        [512, 0.056743],
        [1024, 0.206558],
        [2048, 0.784531],
        [4096, 4.067346],
        [8192, 14.376602],
        [16384, 60.462246]
    ]

    jax_gpu_fd2_1d = [
        [256, 0.131342],
        [512, 0.188642],
        [1024, 0.280244],
        [2048, 0.588404],
        [4096, 1.216925],
        [8192, 3.081974],
        [16384, 8.819256]
    ]

# ========================================================
# 2D
# ========================================================

    # JAX_FLUIDS для 2d

    jax_fluids_gpu_2d = [
        [64, 1.753060],
        [128, 5.190888],
        [256, 13.529942],
        [512, 78.416833],
        [1024, 577.726272]
    ]

    jax_fluids_cpu_2d = [
        [64, 1.3223],
        [128, 11.7088],
        [256, 131.8031],
        [512, 1294.9792]
    ]

    # centpy SD2 для 2d

    centpy_sd2_2d = [
        [256, 0.4194],
        [512, 0.9339],
        [1024, 2.4237],
        [2048, 7.8631],
        [4096, 22.9333],
        [8192, 85.1810],
        [16384, 430.9722]
    ]

    # centpy FD2 для 2d

    centpy_fd2_2d = [
        [256, 0.4194],
        [512, 0.9339],
        [1024, 2.4237],
        [2048, 7.8631],
        [4096, 22.9333],
        [8192, 85.1810],
        [16384, 430.9722]
    ]

    # JAX SD2 для 2d

    jax_cpu_sd2_2d = [
        [64, 1.3477],
        [128, 12.9880],
        [256, 119.9545],
        [512, 1047.8839]
    ]

    jax_gpu_sd2_2d = [
        [64, 0.1082],
        [128, 0.3441],
        [256, 3.0355],
        [512, 31.5721],
        [1024, 271.4567]
    ]

    # JAX FD2 для 2d

    jax_cpu_fd2_2d = [
        [64, 0.884974],
        [128, 9.233143],
        [256, 85.072420],
        [512, 793.6203]
    ]

    jax_gpu_fd2_2d = [
        [64, 0.132604],
        [128, 0.231988],
        [256, 1.458120],
        [512, 13.315762],
        [1024, 95.325242]
    ]

    # SD2 1d
    plot_trajectories_v5(jax_cpu_sd2_1d, jax_gpu_sd2_1d, centpy_sd2_1d, jax_fluids_gpu_1d, jax_fluids_cpu_1d)

    # FD2 1d
    plot_trajectories_v5(jax_cpu_fd2_1d, jax_gpu_fd2_1d, centpy_fd2_1d, jax_fluids_gpu_1d, jax_fluids_cpu_1d)

    # SD2 2d
    plot_trajectories_v5(jax_cpu_sd2_2d, jax_gpu_sd2_2d, centpy_sd2_2d, jax_fluids_gpu_2d, jax_fluids_cpu_2d)

    # FD2 2d
    plot_trajectories_v5(jax_cpu_fd2_2d, jax_gpu_fd2_2d, centpy_fd2_2d, jax_fluids_gpu_2d, jax_fluids_cpu_2d)
