from jax_wave_solver_parallelism import benchmark_vmap
import jax

device = jax.devices("cpu")[0]
results_loop, results_vmap = benchmark_vmap(nx=1024, ny=1024, batch_size=10, steps=500, device=device)

import matplotlib.pyplot as plt

def plot_vmap_batch(results_loop, results_vmap):
    fig, axes = plt.subplots(2, 8, figsize=(16, 8))

    for i in range(8):
        # vmap результаты (верхний ряд)
        axes[0, i].imshow(results_vmap[i], cmap="viridis")
        axes[0, i].set_title(f"vmap wave {i}")

        # loop результаты (нижний ряд)
        axes[1, i].imshow(results_loop[i], cmap="viridis")
        axes[1, i].set_title(f"loop wave {i}")

    plt.tight_layout()
    plt.savefig('vmap_comparison.png')
    plt.close()
    print("Картинки сохранены: vmap_comparison.png")


plot_vmap_batch(results_loop, results_vmap)
