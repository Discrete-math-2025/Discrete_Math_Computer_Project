import argparse
import time
import networkx as nx

from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
from data_generator import generate_graph

def run_networkx(adj_list, source, target):
    """
    Ground Truth: Runs NetworkX built-in Dijkstra.
    Converts your Adjacency List -> NetworkX Graph.
    """
    G = nx.DiGraph()
    n = len(adj_list)
    G.add_nodes_from(range(n))

    for u in range(n):
        for v, w in adj_list[u]:
            G.add_edge(u, v, weight=w)

    try:
        start = time.time()
        path = nx.dijkstra_path(G, source, target, weight='weight')
        dist = nx.dijkstra_path_length(G, source, target, weight='weight')
        dur = (time.time() - start) * 1000

        # Convert path [0, 5, 10] -> [(0, 5), (5, 10)]
        path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return path_tuples, dist, dur
    except nx.NetworkXNoPath:
        return [], float('inf'), 0.0

def main():
    parser = argparse.ArgumentParser(description="UCU Discrete Math Project: SSSP Comparison")
    parser.add_argument("--n", type=int, default=1000, help="Number of vertices")
    parser.add_argument("--density", type=float, default=2.5, help="Average degree (sparsity)")
    args = parser.parse_args()

    print(f"Generating Graph (N={args.n}, Avg Degree={args.density})...")
    graph = generate_graph(args.n, args.density)
    source, target = 0, args.n - 1

    print("-" * 60)
    print(f"{'Algorithm':<20} | {'Time (ms)':<10} | {'Dist':<10} | {'Status'}")
    print("-" * 60)

    nx_path, nx_dist, nx_dur = run_networkx(graph, source, target)
    print(f"{'NetworkX (Ref)':<20} | {nx_dur:<10.4f} | {nx_dist:<10} | ✅ Verified")

    start = time.time()
    d_path, d_dist = dijkstra(graph, source, target)
    d_dur = (time.time() - start) * 1000

    d_status = "✅ Match" if d_dist == nx_dist else f"❌ Fail ({d_dist})"
    print(f"{'Dijkstra (Custom)':<20} | {d_dur:<10.4f} | {d_dist:<10} | {d_status}")

    start = time.time()
    duan_path, duan_dist = run_duan(graph, source, target)
    duan_dur = (time.time() - start) * 1000

    duan_status = "✅ Match" if duan_dist == nx_dist else f"❌ Fail ({duan_dist})"
    print(f"{'Duan et al. (2025)':<20} | {duan_dur:<10.4f} | {duan_dist:<10} | {duan_status}")
    print("-" * 60)

    if d_dist != duan_dist:
        print("Algorithms found different distances!")
        print(f"Dijkstra Path Len: {len(d_path)}")
        print(f"Duan Path Len:     {len(duan_path)}")

if __name__ == "__main__":
    main()
