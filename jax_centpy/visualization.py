import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def create_animation(results, filename="simulation.mp4", fps=30):
    x = results['x']
    t_arr = results['t']
    u_all = results['u_n']

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(x.min(), x.max())

    u_min, u_max = np.min(u_all), np.max(u_all)
    margin = 0.1 * (u_max - u_min) if u_max != u_min else 1.0
    ax.set_ylim(u_min - margin, u_max + margin)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('JAX CentPy Simulation')

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_data(x, u_all[i])
        time_text.set_text(f'time = {t_arr[i]:.2f}')
        return line, time_text

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(u_all), interval=1000 / fps, blit=True)

    print(f"Saving animation to {filename}...")
    try:
        anim.save(filename, writer='ffmpeg', fps=fps)
    except Exception as e:
        print(f"FFmpeg not found, saving as GIF instead. Error: {e}")
        anim.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps)

    print("Done!")
    plt.close(fig)


def create_animation_2d(results, filename="simulation_2d.mp4", fps=24, var_idx=0):
    t_arr = results['t']
    u_all = results['u']

    # Извлекаем нужную физическую величину (по умолчанию - плотность)
    data = u_all[..., var_idx] if u_all.ndim == 4 else u_all

    fig, ax = plt.subplots(figsize=(8, 6))

    # Инициализация первого кадра
    im = ax.imshow(data[0].T, origin='lower', cmap='jet', animated=True,
                   extent=[results['X'].min(), results['X'].max(),
                           results['Y'].min(), results['Y'].max()])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('JAX 2D Simulation')
    fig.colorbar(im, ax=ax, label='Variable')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', weight='bold')

    def init():
        # ИСПОЛЬЗУЕМ set_data вместо set_array и не делаем ravel()
        im.set_data(data[0].T)
        time_text.set_text('')
        return im, time_text

    def animate(i):
        # ИСПОЛЬЗУЕМ set_data для обновления 2D массива
        im.set_data(data[i].T)

        # Динамическая подгонка цвета под мин/макс текущего кадра
        current_min = data[i].min()
        current_max = data[i].max()
        if current_max > current_min:
            im.set_clim(current_min, current_max)

        time_text.set_text(f'time = {t_arr[i]:.3f}')
        return im, time_text

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(data), interval=1000 / fps, blit=True)

    print(f"Saving 2D animation to {filename}...")
    try:
        anim.save(filename, writer='ffmpeg', fps=fps)
    except Exception as e:
        print(f"FFmpeg not found, saving as GIF instead. Error: {e}")
        anim.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps)
    print("Done!")
    plt.close(fig)
