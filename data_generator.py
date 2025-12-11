"""
Simplified graph generator for SSSP benchmarking.
Outputs adjacency matrix as list of lists.
"""

import networkx as nx
import random
import math

INF = float('inf')


def generate_graph(n, m, directed=True, min_w=1, max_w=100):
    """
    Generate a random directed graph with n vertices and m edges.

    Returns:
        adjacency matrix as list of lists
        matrix[u][v] = weight if edge exists, INF otherwise
    """
    G = nx.gnm_random_graph(n, m, directed=directed)

    matrix = [[INF] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 0

    for (u, v) in G.edges():
        matrix[u][v] = random.randint(min_w, max_w)

    return matrix


def generate_benchmark_graphs(num_runs=5):
    """
    Generate multiple random graphs per configuration.
    Returns list of graph dictionaries.
    """
    graphs = []

    sizes = [100, 500, 1000, 5000, 10000]
    sparsity_ratios = [2, 5, 10]

    for n in sizes:

        all_ratios = sparsity_ratios + [math.log2(n)]

        for ratio in all_ratios:
            m = int(n * ratio)

            for run in range(num_runs):
                random. seed(run)
                matrix = generate_graph(n, m, directed=True)

                graphs.append({
                    'graph': matrix,
                    'n': n,
                    'm': m,
                    'sparsity': round(ratio, 2),
                })

    return graphs


if __name__ == "__main__":
    graphs = generate_benchmark_graphs(num_runs=1)

    g = graphs[0]
    print(f"\nn={g['n']}, m={g['m']}")
