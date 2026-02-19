import time
import os

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import pandas as pd
import matplotlib.pyplot as plt
from typing import Type

from core import Pars1d, Equation1d
from equations import make_burgers_1d
from solver import Solver1d, FastSolver1d
from parallel_solver import ParallelSolver1d


def run_solver_benchmark(
        solver_class: Type,
        eqn: Equation1d,
        scheme_name: str,
        run_label: str,
        grid_sizes=[1000, 2000, 4000, 8000]
):
    results = []
    print(f"\n--- Benchmarking: {run_label} ---")

    for J in grid_sizes:
        pars = Pars1d(
            x_init=0.0, x_final=6.28, t_final=1.0, dt_out=1.0,
            J=J, cfl=0.45, scheme=scheme_name
        )

        try:
            solver = solver_class(pars, eqn, scheme_name=scheme_name)

            print(f"  Grid {J}: Warming up...", end="\r")
            warmup_start = time.time()
            res_warm = solver.solve()
            if hasattr(res_warm.get('u_n'), 'block_until_ready'):
                res_warm['u_n'].block_until_ready()
            warmup_time = time.time() - warmup_start

            print(f"  Grid {J}: Running...      ", end="\r")
            start = time.time()
            res = solver.solve()
            if hasattr(res.get('u_n'), 'block_until_ready'):
                res['u_n'].block_until_ready()
            duration = time.time() - start

            print(f"  Grid {J}: {duration:.4f} s (Warmup: {warmup_time:.2f} s)")

            results.append({
                "Grid": J,
                "Time": duration,
                "Label": run_label,
                "Cells_Sec": J / duration
            })

        except Exception as e:
            print(f"  Grid {J}: FAILED with error: {e}")

    return pd.DataFrame(results)


def plot_benchmark_results(df_all: pd.DataFrame, filename="benchmark_comparison.png"):

    if df_all.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, group in df_all.groupby("Label"):
        plt.plot(group["Grid"], group["Time"], 'o-', label=label, linewidth=2)

    plt.xlabel("Grid Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time (Lower is Better)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    for label, group in df_all.groupby("Label"):
        plt.plot(group["Grid"], group["Cells_Sec"], 'o-', label=label, linewidth=2)

    plt.xlabel("Grid Size (N)")
    plt.ylabel("Throughput (Cells / sec)")
    plt.title("Throughput (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Benchmark plot saved to {filename}")


if __name__ == "__main__":

    eqn = make_burgers_1d()
    grids = [10000]

    df1 = run_solver_benchmark(FastSolver1d, eqn, "sd2", "JAX Single Core", grids)

    #df2 = run_solver_benchmark(ParallelSolver1d, eqn, "sd2", "JAX Parallel (4dev)", grids)

    #final_df = pd.concat([df1, df2])
    final_df = df1

    print("\nCombined Results:")
    print(final_df)
    plot_benchmark_results(final_df)
