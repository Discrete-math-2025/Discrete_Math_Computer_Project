"""
Simplified graph generator for SSSP benchmarking.
Outputs adjacency matrix as list of lists.
"""

import networkx as nx
import math
import numpy as np
import random


import networkx as nx
import random

def generate_optimized_graph(n, avg_degree=2.5, max_weight=10):
    """
    Generates a sparse graph using NetworkX and returns an Adjacency List.
    Complexity: O(M) instead of O(N^2).
    """
    # 1. Create Graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # 2. Add Hamiltonian Path (Guaranteed Path 0 -> ... -> n-1)
    # Shuffle internal nodes to ensure randomness
    nodes = list(range(1, n - 1))
    random.shuffle(nodes)
    path_nodes = [0] + nodes + [n - 1]

    nx.add_path(G, path_nodes)

    # 3. Add Random Edges efficiently
    # Calculate needed edges
    target_m = int(n * avg_degree)
    current_m = n - 1
    needed = target_m - current_m

    if needed > 0:
        # gnm_random_graph is highly optimized
        R = nx.gnm_random_graph(n, needed, directed=True)
        G.add_edges_from(R.edges())

    # 4. Assign Weights & Convert to Adjacency List
    # adj_list[u] = [(v, weight), ...]
    adj_list = [[] for _ in range(n)]

    # G.adjacency() is fast
    for u, neighbors in G.adjacency():
        for v, attr in neighbors.items():
            # Use existing weight or assign random if new
            w = attr.get('weight', random.randint(1, max_weight))
            adj_list[u].append((v, w))

    return adj_list


def generate_numpy_graph(n, avg_degree=3, max_weight=10):
    """
    Generates a sparse graph (List of Lists) with a guaranteed
    Hamiltonian path from 0 to n-1.
    NO NumPy required.

    Args:
        n (int): Number of vertices.
        avg_degree (float): Average edges per node.
        max_weight (int): Max edge weight.

    Returns:
        list: NxN adjacency matrix (list of lists).
    """
    # 1. Initialize matrix with zeros using standard lists
    # This creates an N x N grid of 0s
    matrix = [[0] * n for _ in range(n)]

    # 2. Create the Guaranteed Path (Hamiltonian)
    # This ensures graph is connected and path 0 -> n-1 exists.
    nodes = list(range(1, n - 1))
    random.shuffle(nodes)
    path_nodes = [0] + nodes + [n - 1]

    # Set path edges
    for k in range(len(path_nodes) - 1):
        u = path_nodes[k]
        v = path_nodes[k + 1]
        matrix[u][v] = random.randint(1, max_weight)

    # 3. Add Random Edges (Sparse Method)
    # We want total edges m approx n * avg_degree
    target_m = int(n * avg_degree)
    current_m = n - 1  # We already added n-1 edges for the path

    if current_m < target_m:
        needed = target_m - current_m

        # Pure Python random generation
        while needed > 0:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)

            if u == v: continue            # No self-loops
            if matrix[u][v] > 0: continue  # Edge already exists

            matrix[u][v] = random.randint(1, max_weight)
            needed -= 1

    return matrix
# def check_path_weight_manually(matrix, path_tuples):
#     """
#     Проходиться по матриці згідно зі знайденим шляхом і сумує ваги.
#     """
#     calculated_weight = 0
#     for u, v in path_tuples:
#         w = matrix[u][v]
#         calculated_weight += w

#     return calculated_weight


# def generate_sparse_graph_numpy(n, density=2.0, seed=None):
#     """
#     Generates a directed, weighted sparse graph represented by an adjacency matrix
#     using only NumPy.

#     The adjacency matrix A is an n x n NumPy array where A[i, j] stores the
#     weight of the edge from node i to node j. A[i, j] = 0 means no edge.

#     Args:
#         n: The number of nodes (size of the matrix).
#         density: The target multiplier for the number of edges (m = n * density).
#         seed: Optional seed for the random number generator for reproducibility.

#     Returns:
#         A NumPy ndarray of shape (n, n) representing the adjacency matrix.
#     """
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)

#     if n <= 0:
#         return np.array([[]])

#     # 1. Initialize the n x n adjacency matrix with zeros.
#     adj_matrix = np.zeros((n, n), dtype=float)

#     max_possible_edges = n * (n - 1)
#     m = int(n * density)
#     m = min(m, max_possible_edges)

#     # 2. Create the "chain" structure (i -> i+1)
#     edges_added = 0
#     for i in range(n - 1):
#         # Generate a random weight between 1 and 10
#         weight = random.uniform(1.0, 10.0)
#         # Add the edge (i, i+1)
#         adj_matrix[i, i + 1] = weight
#         edges_added += 1

#     # 3. Add remaining random edges until 'm' edges are reached
#     attempts = 0
#     max_attempts = m * 10

#     while edges_added < m and attempts < max_attempts:
#         # Select source and destination nodes randomly
#         src = random.randint(0, n - 1)
#         dst = random.randint(0, n - 1)
#         attempts += 1

#         # Check conditions: src != dst AND no existing edge (adj_matrix[src, dst] == 0)
#         if src != dst and adj_matrix[src, dst] == 0.0:
#             weight = random.uniform(1.0, 10.0)
#             adj_matrix[src, dst] = weight
#             edges_added += 1

#     # print(f"Graph generated with {n} nodes and {edges_added} edges.") # Commented out print for cleaner output
#     return adj_matrix




# def generate_benchmark_graphs(num_runs=5):
#     """
#     Generate multiple random graphs per configuration.
#     Returns list of graph dictionaries.
#     """
#     graphs = []

#     sizes = [2000]
#     sparsity_ratios = [3]

#     for n in sizes:
#         for run in range(num_runs):
#             matrix = generate_graph_matrix_guaranteed_path(n, density=0.1)

#             graphs.append({
#                 'graph': matrix,
#                 'sparsity': 0.3,
#             })
#     return graphs


# if __name__ == "__main__":
#     graphs = generate_benchmark_graphs(num_runs=1)

#     g = graphs[0]
#     print(f"\nn={g['n']}, m={g['m']}")
