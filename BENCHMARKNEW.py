import timeit
import matplotlib.pyplot as plt


# Import your algorithms
# Ensure algorithms/duan_algo.py and algorithms/dijkstra.py exist
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
from data_generator import generate_numpy_graph

def run_benchmark_suite():
    # Configuration
    sizes = [50, 100, 200, 300, 500, 1000, 2000, 3000]  # Sizes of N to test
    fixed_density = 1.8           # Sparsity m/n

    times_dijkstra = []
    times_duan = []

    print(f"{'N':<10} | {'Dijkstra (ms)':<15} | {'Duan et al. (ms)':<15}")
    print("-" * 45)

    for n in sizes:
        # Generate graph
        # density = 2.0 means m approx 2*n
        matrix = generate_numpy_graph(n, fixed_density, 20)

        # Define setup for timeit (random source/target)
        source = 0
        target = n - 1

        # 1. Benchmark Dijkstra
        # We use a lambda/partial to pass arguments to timeit
        number = 5
        t_dijk = timeit.timeit(lambda: dijkstra(matrix, source, target), number=number)
        avg_dijk = (t_dijk / number) * 1000 # Convert to ms
        times_dijkstra.append(avg_dijk)

        # 2. Benchmark Duan et al.
        t_duan = timeit.timeit(lambda: run_duan(matrix, source, target), number=number)
        avg_duan = (t_duan / number) * 1000 # Convert to ms
        times_duan.append(avg_duan)

        print(f"{n:<10} | {avg_dijk:<15.4f} | {avg_duan:<15.4f}")

    # Plotting Results
    plot_results(sizes, times_dijkstra, times_duan, fixed_density)

def plot_results(sizes, y1, y2, density):
    plt.figure(figsize=(10, 6))

    plt.plot(sizes, y1, marker='o', label='Dijkstra (Standard)', color='blue')
    plt.plot(sizes, y2, marker='s', label='Duan et al. (2025)', color='red', linestyle='--')

    plt.title(f'SSSP Algorithm Comparison (Density $\\approx$ {density})')
    plt.xlabel('Graph Size (Number of Vertices)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_file = 'results/benchmark_plot.png'
    # Ensure directory exists
    import os
    os.makedirs('results', exist_ok=True)

    plt.savefig(output_file)
    print(f"\n✅ Plot saved to {output_file}")

if __name__ == "__main__":
    run_benchmark_suite()
