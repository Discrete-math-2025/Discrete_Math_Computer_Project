"""
Benchmarking framework for SSSP algorithms using timeit.
Works with adjacency matrix (list of lists).
"""

import timeit
import csv
import random
import heapq
from functools import partial
from data_generator import generate_benchmark_graphs, generate_graph_matrix_guaranteed_path
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


INF = float("inf")

# ============================================================
# GRAPH UTILITIES
# ============================================================

def load_graph_from_csv(filename):
    """
    Load graph from CSV file as adjacency matrix.
    Expected format:  source, dest, weight
    """
    max_node = 0
    edges = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row['source'])
            v = int(row['dest'])
            w = int(row['weight'])
            edges.append((u, v, w))
            max_node = max(max_node, u, v)

    n = max_node + 1

    # Build matrix
    matrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0

    for u, v, w in edges:
        matrix[u][v] = w

    return matrix


def get_graph_stats(matrix):
    """Get basic statistics about a graph."""
    n = len(matrix)
    m = sum(1 for i in range(n) for j in range(n) if matrix[i][j] != 0 and i != j)
    return {'n': n, 'm': m, 'sparsity': round(m / n, 2) if n > 0 else 0}

def benchmark_csv_file(filename, algorithms):
    """Benchmark algorithms on a graph from CSV file."""
    matrix = load_graph_from_csv(filename)
    stats = get_graph_stats(matrix)
    times = compare_algorithms(matrix, algorithms)

    return {'file': filename, **stats, 'times': times}

#________________________________________________________________________________________

def compare_algorithms(matrix, algorithms):
    """
    Compare multiple algorithms on the same graph.
    """
    results = {}
    for name, func in algorithms.items():
        timer = timeit.Timer(partial(func, matrix, 0, random.randint(0, len(matrix))))
        times = timer.repeat(3, 5)
        results[name] = min(times) / 5
    return results



def benchmark_generated_graphs(algorithms):
    """
    Benchmark algorithms on generated graphs.
    """

    graphs = generate_benchmark_graphs(1)
    results = []

    for graph in graphs:
        matrix = graph['graph']

        times = compare_algorithms(matrix, algorithms)

        results.append(tuple([times, times['Dijkstra'] / times['Duan']]))
    return results






# matr = generate_graph_matrix_guaranteed_path(20, density=0.01, max_weight=100)
# print(run_duan(matr, 0, len(matr)-1))
# G = nx.from_numpy_array(np.asarray(matr), create_using=nx.DiGraph)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15)
# plt.title("Graph from Adjacency Matrix")
# plt.show()

if __name__ == "__main__":


    algorithms = {
        'Dijkstra':  dijkstra,
        'Duan':  run_duan
    }


    results_ = benchmark_generated_graphs(algorithms)

    print(results_)

    print(f"\n✅ Benchmarked {len(results_)} graphs")
