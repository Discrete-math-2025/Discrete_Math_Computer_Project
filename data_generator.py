"""
Simplified graph generator for SSSP benchmarking.
Outputs adjacency matrix as list of lists.
"""

import networkx as nx
import math
import numpy as np
import random
# INF = float('inf')


# def generate_graph(n, m, directed=True, min_w=1, max_w=100):
#     """
#     Generate a random directed graph with n vertices and m edges.

#     Returns:
#         adjacency matrix as list of lists
#         matrix[u][v] = weight if edge exists, INF otherwise
#     """
#     G = nx.gnm_random_graph(n, m, directed=directed)

#     matrix = [[INF] * n for _ in range(n)]

#     for i in range(n):
#         matrix[i][i] = 0

#     for (u, v) in G.edges():
#         matrix[u][v] = random.randint(min_w, max_w)

#     return matrix
# print(generate_graph(10, 10))


def generate_numpy_graph(n, avg_degree=3, max_weight=10):
    """
    Generates a sparse graph with a guaranteed Hamiltonian path from 0 to n-1.

    Args:
        n (int): Number of vertices.
        avg_degree (float): Average number of edges per node (sparsity control).
                            For sparse graphs, keep this low (e.g., 2, 3, 5).
                            Total edges m approx n * avg_degree.
        max_weight (int): Maximum edge weight.

    Returns:
        np.ndarray: Adjacency matrix (NxN).
    """
    # 1. Initialize empty matrix
    matrix = np.zeros((n, n))

    # 2. Create the Guaranteed Path (Hamiltonian: visits all nodes)
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
    # We want total edges m = n * avg_degree
    target_m = int(n * avg_degree)
    current_m = n - 1  # We already added n-1 edges for the path

    # Generate remaining edges efficiently using NumPy
    if current_m < target_m:
        needed = target_m - current_m

        # Generate 'needed' random pairs.
        # Note: This might generate duplicates or self-loops, but for
        # sparse graphs (needed << N^2), collisions are rare enough to ignore
        # or handle simply.
        while needed > 0:
            # Generate a batch of random coordinates
            us = np.random.randint(0, n, size=needed)
            vs = np.random.randint(0, n, size=needed)

            for u, v in zip(us, vs):
                if u == v: continue            # No self-loops
                if matrix[u][v] > 0: continue  # Edge already exists

                matrix[u][v] = random.randint(1, max_weight)
                needed -= 1
                if needed == 0: break

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
