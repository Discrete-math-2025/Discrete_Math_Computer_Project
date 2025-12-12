"""
Simplified graph generator.
Outputs adjacency matrix as list of lists.
"""

import networkx as nx
import random

def generate_graph(n, avg_degree=2.5, max_weight=10):
    """
    Generate a sparse graph using NetworkX and return an Adjacency List.
    for O(M) instead of O(N^2).
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    nodes = list(range(1, n - 1))
    random.shuffle(nodes)
    path_nodes = [0] + nodes + [n - 1]

    nx.add_path(G, path_nodes)

    target_m = int(n * avg_degree)
    current_m = n - 1
    needed = target_m - current_m

    if needed > 0:
        R = nx.gnm_random_graph(n, needed, directed=True)
        G.add_edges_from(R.edges())

    adj_list = [[] for _ in range(n)]

    for u, neighbors in G.adjacency():
        for v, attr in neighbors.items():
            w = attr.get('weight', random.randint(1, max_weight))
            adj_list[u].append((v, w))

    return adj_list
