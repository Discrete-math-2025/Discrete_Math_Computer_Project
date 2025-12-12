import argparse
import timeit
import matplotlib.pyplot as plt
import os
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
from data_generator import generate_graph

def benchmark_vary_size(sizes, fixed_density, trials):
    """
    Graph Size (N) with Fixed Density.
    """
    print(f"\n[Experiment: Varying Size] Density={fixed_density}, Trials={trials}")
    print(f"{'N':<10} | {'Dijkstra (ms)':<15} | {'Duan (ms)':<15}")
    print("-" * 45)

    times_dijkstra = []
    times_duan = []

    for n in sizes:
        graph = generate_graph(n, avg_degree=fixed_density, max_weight=100)
        source, target = 0, n - 1

        t_duan = timeit.timeit(lambda: run_duan(graph, source, target), number=trials)
        avg_duan = (t_duan / trials) * 1000
        times_duan.append(avg_duan)

        t_dijk = timeit.timeit(lambda: dijkstra(graph, source, target), number=trials)
        avg_dijk = (t_dijk / trials) * 1000
        times_dijkstra.append(avg_dijk)

        print(f"{n:<10} | {avg_dijk:<15.4f} | {avg_duan:<15.4f}")

    plot_results(sizes, times_dijkstra, times_duan,
                 title=f'Time vs Graph Size (Density={fixed_density})',
                 xlabel='Number of Vertices (N)',
                 filename=f'benchmark_size_d{fixed_density}.png')

def benchmark_vary_density(fixed_n, densities, trials):
    """
    Experiment 2: Vary Density (Avg Degree) with Fixed Size (N).
    """
    print(f"\n[Experiment: Varying Density] N={fixed_n}, Trials={trials}")
    print(f"{'Density':<10} | {'Dijkstra (ms)':<15} | {'Duan (ms)':<15}")
    print("-" * 45)

    times_dijkstra = []
    times_duan = []

    for d in densities:
        graph = generate_graph(fixed_n, avg_degree=d, max_weight=100)
        source, target = 0, fixed_n - 1

        t_duan = timeit.timeit(lambda: run_duan(graph, source, target), number=trials)
        avg_duan = (t_duan / trials) * 1000
        times_duan.append(avg_duan)

        t_dijk = timeit.timeit(lambda: dijkstra(graph, source, target), number=trials)
        avg_dijk = (t_dijk / trials) * 1000
        times_dijkstra.append(avg_dijk)

        print(f"{d:<10} | {avg_dijk:<15.4f} | {avg_duan:<15.4f}")

    plot_results(densities, times_dijkstra, times_duan,
                 title=f'Time vs Density (N={fixed_n})',
                 xlabel='Average Degree (Edges/Node)',
                 filename=f'benchmark_density_n{fixed_n}.png')

def plot_results(x_values, y_dijk, y_duan, title, xlabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_dijk, marker='o', label='Dijkstra (Standard)', color='blue')
    plt.plot(x_values, y_duan, marker='s', label='Duan et al. (2025)', color='red', linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', filename)
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Dijkstra vs Duan et al. (2025)")

    parser.add_argument('--mode', choices=['size', 'density', 'both'], default='both',
                        help="Benchmark mode: vary 'size', vary 'density', or 'both'")

    parser.add_argument('--sizes', type=int, nargs='+',
                        default=[100, 500, 1000, 2000, 5000, 10000],
                        help="List of N values to test (e.g. 100 1000 5000)")
    parser.add_argument('--fixed-density', type=float, default=2.5,
                        help="Fixed density (avg degree) when varying size")

    parser.add_argument('--fixed-n', type=int, default=5000,
                        help="Fixed N when varying density")
    parser.add_argument('--densities', type=float, nargs='+',
                        default=[1.1, 2.0, 4.0, 8.0, 16.0],
                        help="List of densities (avg degree) to test")

    parser.add_argument('--trials', type=int, default=5,
                        help="Number of trials per run for average")

    args = parser.parse_args()

    if args.mode in ['size', 'both']:
        benchmark_vary_size(args.sizes, args.fixed_density, args.trials)

    if args.mode in ['density', 'both']:
        benchmark_vary_density(args.fixed_n, args.densities, args.trials)

if __name__ == "__main__":
    main()
