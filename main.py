import argparse
import re
import os
import sys
import time
import networkx as nx

# Ensure we can import modules from the current directory
sys.path.append(os.getcwd())

# Import your algorithms
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan

def load_graph_from_file(filepath):
    """
    Reads a file containing a Python-style adjacency list string WITHOUT using ast/eval.
    Format: [[(35, 3)], [(37, 49), (13, 42)], ...]
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit()

    with open(filepath, 'r') as f:
        content = f.read().strip()

    try:
        if content.startswith('[') and content.endswith(']'):
            content = content[1:-1]

        row_strings = re.findall(r'\[([^\]]*)\]', content)

        adj_list = []

        for row_str in row_strings:
            neighbors = []
            tuple_matches = re.findall(r'\((\d+),\s*(\d+)\)', row_str)

            for target_str, weight_str in tuple_matches:
                neighbors.append((int(target_str), int(weight_str)))

            adj_list.append(neighbors)

        return adj_list

    except Exception as e:
        print(f"Error when looking for graph file manually: {e}")
        sys.exit()

def run_networkx_ground_truth(adj_list, source, target):
    """
    Runs NetworkX's highly optimized Dijkstra for verification.
    """
    G = nx.DiGraph()
    n = len(adj_list)
    G.add_nodes_from(range(n))

    for u, neighbors in enumerate(adj_list):
        for v, w in neighbors:
            G.add_edge(u, v, weight=w)

    try:
        start_t = time.perf_counter()
        path = nx.dijkstra_path(G, source, target)
        dist = nx.dijkstra_path_length(G, source, target)
        dur = (time.perf_counter() - start_t) * 1000

        path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return path_tuples, dist, dur
    except nx.NetworkXNoPath:
        return [], float('inf'), 0.0

def main():
    parser = argparse.ArgumentParser(description="Run SSSP algorithms on a specific graph file.")
    parser.add_argument('--file', type=str, default='data/graph.csv',
                        help="Path to the graph file (default: data/graph.csv)")
    parser.add_argument('--source', type=int, default=0, help="Start node index")
    parser.add_argument('--target', type=int, default=-1,
                        help="Target node index (default: -1 for last node)")

    args = parser.parse_args()

    print(f"Loading graph from {args.file}")
    graph = load_graph_from_file(args.file)
    n = len(graph)
    print(f"Graph loaded. Nodes: {n}")

    target = args.target if args.target != -1 else n - 1

    if args.source >= n:
        print(f"Error: Source ({args.source}) out of bounds for N={n}")
        sys.exit()
    if target >= n:
        print(f"Target ({target}) out of bounds. Using last node {n-1}.")
        target = n - 1

    print(f"Finding path: {args.source} -> {target}")
    print("-" * 75)
    print(f"{'Algorithm':<20} | {'Time (ms)':<10} | {'Distance':<10} | {'Status'}")
    print("-" * 75)

    nx_path, nx_dist, nx_time = run_networkx_ground_truth(graph, args.source, target)
    print(f"{'NetworkX (Ref)':<20} | {nx_time:<10.4f} | {nx_dist:<10} | Match")

    start_t = time.perf_counter()
    d_path, d_dist = dijkstra(graph, args.source, target)
    d_time = (time.perf_counter() - start_t) * 1000

    d_status = "Match" if d_dist == nx_dist else f" Fail ({d_dist})"
    print(f"{'Dijkstra (Custom)':<20} | {d_time:<10.4f} | {d_dist:<10} | {d_status}")

    start_t = time.perf_counter()
    duan_path, duan_dist = run_duan(graph, args.source, target)
    duan_time = (time.perf_counter() - start_t) * 1000

    duan_status = "Match" if duan_dist == nx_dist else f"Fail ({duan_dist})"
    print(f"{'Duan et al. (2025)':<20} | {duan_time:<10.4f} | {duan_dist:<10} | {duan_status}")
    print("-" * 75)

    if d_dist != duan_dist:
        print("MISMATCH DETAILS")
        print(f"NetworkX Dist: {nx_dist}")
        print(f"Duan Dist:     {duan_dist}")
    elif nx_dist != float('inf'):
        print(f"\nPath: {nx_path}")

if __name__ == "__main__":
    main()
