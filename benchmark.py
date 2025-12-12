import timeit
import matplotlib.pyplot as plt
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
from data_generator import generate_optimized_graph
import os


def run_benchmark_suite():
    sizes = [2, 10, 50, 250, 750, 1000, 2000, 5000, 10000, 100000]
    fixed_density = 2

    times_dijkstra = []
    times_duan = []

    print(f"{'N':<10} | {'Dijkstra (ms)':<15} | {'Duan et al. (ms)':<15}")
    print("-" * 45)

    for n in sizes:
        # Generate graph
        print('Generating graph')
        matrix = generate_optimized_graph(n, avg_degree=fixed_density, max_weight=100)

        source = 0
        target = n - 1

        number = 5

        t_duan = timeit.timeit(lambda: run_duan(matrix, source, target), number=number)
        avg_duan = (t_duan / number) * 1000
        times_duan.append(avg_duan)

        t_dijk = timeit.timeit(lambda: dijkstra(matrix, source, target), number=number)
        avg_dijk = (t_dijk / number) * 1000
        times_dijkstra.append(avg_dijk)



        print(f"{n:<10} | {avg_dijk:<15.4f} | {avg_duan:<15.4f}")

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
    os.makedirs('results', exist_ok=True)

    plt.savefig(output_file)
    plt.show()
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_benchmark_suite()
