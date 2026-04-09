import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#def compare_all_slices(centpy_file, gpu_file):
    # # 1. Загрузка данных
    # data_c = np.load(centpy_file)
    # data_g = np.load(gpu_file)
    #
    # # 2. Извлекаем плотность (компонента 0)
    # # Предполагаем, что размерность (X, Y, 4)
    # rho_c_full = data_c['u'][..., 0]
    # rho_g_full = data_g['u'][..., 0]
    #
    # # Отрезаем ghost-ячейки (по 2 с каждой стороны для метода KT2), если они сохранены
    # # Если вы их уже отрезали при сохранении, закомментируйте следующие две строки
    # rho_c = rho_c_full[2:-2, 2:-2]
    # rho_g = rho_g_full[2:-2, 2:-2]
    #
    # # Координаты (отрезаем ghost-ячейки, если нужно)
    # x = np.linspace(0, 1, rho_c.shape[0])
    # y = np.linspace(0, 1, rho_c.shape[1])
    #
    # # Индексы середин
    # mid_x = rho_c.shape[0] // 2
    # mid_y = rho_c.shape[1] // 2
    #
    # # 3. Подготовка рисунка с 4 графиками
    # fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle('Сравнение решений 2D Euler: CentPy (CPU) vs Custom (GPU)', fontsize=16)
    #
    # # --- График 1: Горизонтальный срез (y = 0.5) ---
    # axs[0, 0].plot(x, rho_c[:, mid_y], 'r--', label='CentPy', linewidth=2)
    # axs[0, 0].plot(x, rho_g[:, mid_y], 'b-', label='GPU', alpha=0.7)
    # axs[0, 0].set_title('Горизонтальный срез (по X)')
    # axs[0, 0].grid(True)
    # axs[0, 0].legend()
    #
    # # --- График 2: Вертикальный срез (x = 0.5) ---
    # axs[0, 1].plot(y, rho_c[mid_x, :], 'r--', label='CentPy', linewidth=2)
    # axs[0, 1].plot(y, rho_g[mid_x, :], 'b-', label='GPU', alpha=0.7)
    # axs[0, 1].set_title('Вертикальный срез (по Y)')
    # axs[0, 1].grid(True)
    # axs[0, 1].legend()
    #
    # # --- График 3: Диагональный срез (x = y) ---
    # # Извлекаем главную диагональ матрицы
    # diag_c = np.diag(rho_c)
    # diag_g = np.diag(rho_g)
    # diag_coords = np.linspace(0, np.sqrt(2), len(diag_c))  # Координата вдоль диагонали
    #
    # axs[1, 0].plot(diag_coords, diag_c, 'r--', label='CentPy', linewidth=2)
    # axs[1, 0].plot(diag_coords, diag_g, 'b-', label='GPU', alpha=0.7)
    # axs[1, 0].set_title('Диагональный срез (X = Y)')
    # axs[1, 0].grid(True)
    # axs[1, 0].legend()
    #
    # # --- График 4: 2D тепловая карта абсолютной ошибки ---
    # error_matrix = np.abs(rho_c - rho_g)
    # im = axs[1, 1].imshow(error_matrix.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    # axs[1, 1].set_title('Абсолютная разница (Карта ошибок)')
    # fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    #
    # plt.tight_layout()
    # plt.savefig('comparison_full.png', dpi=300)
    # plt.show()
    #
    # # 4. Вывод численной метрики в консоль
    # max_error = np.max(error_matrix)
    # print(f"Максимальная абсолютная ошибка между CPU и GPU: {max_error:.2e}")

    # 1. Загружаем оба файла
    # data_cpu = np.load('centpy_data.npz')
    # data_gpu = np.load('gpu_data.npz')
    #
    # # 2. Достаем полные массивы (теперь ключи совпадают!)
    # u_cpu = data_cpu['u']
    # u_gpu = data_gpu['u']
    #
    # # 3. Извлекаем плотность (rho) для финального шага по времени
    # # Индекс [-1] берет последний шаг по времени (t_final)
    # # Индекс [..., 0] берет первую компоненту вектора решения (плотность)
    # rho_cpu_final = u_cpu[-1, ..., 0]
    # rho_gpu_final = u_gpu[-1, ..., 0]
    #
    # # 4. Извлекаем сетку (она должна быть одинаковой, берем из любого файла)
    # X = data_gpu['X']
    # Y = data_gpu['Y']
    #
    # # --- Пример проверки или построения графика ---
    #
    # # Считаем разницу между CPU и GPU
    # error = np.abs(rho_cpu_final - rho_gpu_final)
    # print(f"Максимальная разница в плотности между CPU и GPU: {np.max(error)}")
    #
    # # Если нужно нарисовать
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # axes[0].contourf(X, Y, rho_cpu_final, levels=20, cmap='viridis')
    # axes[0].set_title('CPU Плотность')
    #
    # axes[1].contourf(X, Y, rho_gpu_final, levels=20, cmap='viridis')
    # axes[1].set_title('GPU Плотность')
    #
    # img = axes[2].contourf(X, Y, error, levels=20, cmap='magma')
    # axes[2].set_title('Абсолютная ошибка')
    # plt.colorbar(img, ax=axes[2])
    #
    # plt.tight_layout()
    # plt.show()


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

if __name__ == "__main__":
    #compare_all_slices('centpy_data.npz', 'gpu_data.npz')
    fixed_jax_cpu = [
        [256, 1.9267],
        [512, 2.4545],
        [1024, 6.1065],
        [2048, 19.5295],
        [4096, 75.5111],
        [8192, 326.1863],
        [16384, 867.1382]
    ]

    fixed_jax_gpu = [
        [256, 0.9232],
        [512, 1.3046],
        [1024, 2.2491],
        [2048, 3.7191],
        [4096, 7.5645],
        [8192, 13.6326],
        [16384, 26.7994]
    ]

    fixed_centpy = [
        [256, 0.4194],
        [512, 0.9339],
        [1024, 2.4237],
        [2048, 7.8631],
        [4096, 22.9333],
        [8192, 85.1810],
        [16384, 430.9722]
    ]

    fixed_jax_fluids_gpu = [
        [256, 4.0228],
        [512, 7.2085],
        [1024, 14.1745],
        [2048, 28.4376],
        [4096, 60.4977],
        [8192, 127.7438],
        [16384, 258.8769]
    ]

    fixed_jax_fluids_cpu = [
        [256, 1.1211],
        [512, 2.1045],
        [1024, 4.4921],
        [2048, 12.2068],
        [4096, 43.0459],
        [8192, 160.1402],
        [16384, 583.8857]
    ]

    # plot_trajectories(fixed_jax_cpu, fixed_centpy)
    # plot_trajectories_v2(fixed_jax_cpu, fixed_centpy)
    plot_trajectories_v3(fixed_jax_cpu, fixed_jax_gpu, fixed_centpy, fixed_jax_fluids_gpu, fixed_jax_fluids_cpu)
