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
