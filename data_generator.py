"""
Simplified graph generator for SSSP benchmarking.
Outputs adjacency matrix as list of lists.
"""

import networkx as nx
import random
import math
import numpy as np
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


def generate_graph_matrix_guaranteed_path(n, density=0.3, max_weight=10):
    """
    Генерує граф, де ГАРАНТОВАНО існує шлях від вершини 0 до n-1.
    """
    matrix = np.zeros((n, n))

    nodes = list(range(1, n - 1))
    random.shuffle(nodes) # Перемішуємо їх

    path_nodes = [0] + nodes + [n - 1]

    for k in range(len(path_nodes) - 1):
        u = path_nodes[k]
        v = path_nodes[k + 1]
        matrix[u][v] = random.randint(1, max_weight)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if matrix[i][j] > 0:
                continue
            if random.random() < density:
                matrix[i][j] = random.randint(1, max_weight)

    return matrix


def check_path_weight_manually(matrix, path_tuples):
    """
    Проходиться по матриці згідно зі знайденим шляхом і сумує ваги.
    """
    calculated_weight = 0
    for u, v in path_tuples:
        w = matrix[u][v]
        calculated_weight += w

    return calculated_weight


def generate_sparse_graph_numpy(n, density=2.0, seed=None):
    """
    Generates a directed, weighted sparse graph represented by an adjacency matrix
    using only NumPy.

    The adjacency matrix A is an n x n NumPy array where A[i, j] stores the
    weight of the edge from node i to node j. A[i, j] = 0 means no edge.

    Args:
        n: The number of nodes (size of the matrix).
        density: The target multiplier for the number of edges (m = n * density).
        seed: Optional seed for the random number generator for reproducibility.

    Returns:
        A NumPy ndarray of shape (n, n) representing the adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if n <= 0:
        return np.array([[]])

    # 1. Initialize the n x n adjacency matrix with zeros.
    adj_matrix = np.zeros((n, n), dtype=float)

    max_possible_edges = n * (n - 1)
    m = int(n * density)
    m = min(m, max_possible_edges)

    # 2. Create the "chain" structure (i -> i+1)
    edges_added = 0
    for i in range(n - 1):
        # Generate a random weight between 1 and 10
        weight = random.uniform(1.0, 10.0)
        # Add the edge (i, i+1)
        adj_matrix[i, i + 1] = weight
        edges_added += 1

    # 3. Add remaining random edges until 'm' edges are reached
    attempts = 0
    max_attempts = m * 10

    while edges_added < m and attempts < max_attempts:
        # Select source and destination nodes randomly
        src = random.randint(0, n - 1)
        dst = random.randint(0, n - 1)
        attempts += 1

        # Check conditions: src != dst AND no existing edge (adj_matrix[src, dst] == 0)
        if src != dst and adj_matrix[src, dst] == 0.0:
            weight = random.uniform(1.0, 10.0)
            adj_matrix[src, dst] = weight
            edges_added += 1

    # print(f"Graph generated with {n} nodes and {edges_added} edges.") # Commented out print for cleaner output
    return adj_matrix




def generate_benchmark_graphs(num_runs=5):
    """
    Generate multiple random graphs per configuration.
    Returns list of graph dictionaries.
    """
    graphs = []

    sizes = [10, 20, 30, 100]
    sparsity_ratios = [1, 2, 3]

    for n in sizes:

        all_ratios = sparsity_ratios + [math.log2(n)]

        for ratio in all_ratios:
            m = int(n * ratio)

            for run in range(num_runs):
                random. seed(run)
                matrix = generate_graph_matrix_guaranteed_path(n)

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
