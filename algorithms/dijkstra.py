import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time


def load_matrix(file_path):
    """
    Зчитує CSV файл з матрицею суміжності у NumPy масив.
    """
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Помилка зчитування файлу: {e}")
        return None


def dijkstra(matrix, start_node, end_node, visualize=False, delay=0.8):
    """
    Your original Dijkstra algorithm with optional visualization.

    Args:
        matrix: 2D numpy array (NxN) - Матриця суміжності
        start_node: int
        end_node:  int
        visualize: bool - Whether to show real-time visualization
        delay: float - Seconds between visualization steps

    Returns:
        path_tuples: list of edge tuples [(0,1), (1,3), ...]
        distance: total shortest distance
    """
    n = matrix.shape[0]

    if start_node < 0 or start_node >= n or end_node < 0 or end_node >= n:
        return [], float('inf')

    if visualize:
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    G.add_edge(i, j, weight=matrix[i][j])

        pos = nx.spring_layout(G, seed=42, k=2)
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        visited_set = set()
        def draw_state(current_node=None):
            ax.clear()

            node_colors = []
            for node in range(n):
                if node == start_node:
                    node_colors.append('#2ecc71')  # Green - start
                elif node == end_node:
                    node_colors. append('#e74c3c')  # Red - end
                elif node == current_node:
                    node_colors. append('#f1c40f')  # Yellow - current
                elif node in visited_set:
                    node_colors.append('#9b59b6')  # Purple - visited
                elif any(node == v for _, v in pq):
                    node_colors.append('#3498db')  # Blue - in queue
                else:
                    node_colors.append('#ecf0f1')  # Gray - unvisited

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc',
                                arrows=True, arrowsize=20, width=1.5,
                                connectionstyle="arc3,rad=0.1")

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                node_size=800, edgecolors='black', linewidths=2)

            edge_labels = {(u, v): f"{d['weight']:.0f}"
                        for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=10)

            labels = {}
            for node in range(n):
                d = dist[node]
                d_str = f"{d:.0f}" if d != np.inf else "∞"
                labels[node] = f"{node}\n[{d_str}]"
            nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                    font_size=10, font_weight='bold')

            ax.axis('off')

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(delay)


    dist = np.full(n, np.inf)
    dist[start_node] = 0

    parent = np.full(n, -1, dtype=int)

    pq = [(0, start_node)]



    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if visualize:
            if u in visited_set:
                continue
            visited_set.add(u)
            draw_state(current_node=u)

        if u == end_node:
            if visualize:
                draw_state(current_node=u)
            break

        for v in range(n):
            weight = matrix[u][v]

            if weight > 0:
                new_dist = current_dist + weight

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))


    if dist[end_node] == np.inf:
        if visualize:
            plt.ioff()
            plt.show()
        return [], float('inf')

    path = []
    curr = end_node
    while curr != -1:
        path.append(curr)
        curr = parent[curr]

    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]


    if visualize:
        ax.clear()

        path_set = set(path)
        node_colors = []
        for node in range(n):
            if node == start_node:
                node_colors. append('#2ecc71')
            elif node == end_node:
                node_colors.append("#a02226")
            elif node in path_set:
                node_colors.append('#2ecc71')  # Bright green
            else:
                node_colors.append('#9b59b6')

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc',
                               arrows=True, arrowsize=20, width=1.5,
                               connectionstyle="arc3,rad=0.1")


        nx.draw_networkx_edges(G, pos, edgelist=path_tuples, ax=ax,
                               edge_color='#27ae60', arrows=True,
                               arrowsize=25, width=4,
                               connectionstyle="arc3,rad=0.1")


        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=800, edgecolors='black', linewidths=2)


        edge_labels = {(u, v): f"{d['weight']:.0f}"
                       for u, v, d in G. edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=10)

        labels = {}
        for node in range(n):
            d = dist[node]
            d_str = f"{d:.0f}" if d != np.inf else "∞"
            labels[node] = f"{node}\n[{d_str}]"
        nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                font_size=10, font_weight='bold')


        path_str = " → ".join(map(str, path))
        ax.set_title(f"✅ Найкоротший шлях: {path_str}\n"
                    f"Загальна відстань: {dist[end_node]:.0f}",
                    fontsize=14, fontweight='bold', color='green')
        ax.axis('off')

        fig.canvas.draw()
        plt.ioff()
        plt.show()

    return list(map(lambda x: (int(x[0]), int(x[1])), path_tuples)), dist[end_node]


if __name__ == "__main__":
    # Test matrix
    matrix = np.array([
        [0, 4, 2, 0, 0, 0],
        [0, 0, 1, 5, 0, 0],
        [0, 0, 0, 8, 10, 0],
        [0, 0, 0, 0, 2, 6],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0]
    ], dtype=float)

    print("Матриця суміжності:")
    print(matrix)
    print()

    print("=== Без візуалізації ===")
    path1, dist1 = dijkstra(matrix, 0, 5, visualize=False)
    print(f"Шлях: {path1}")
    print(f"Відстань: {dist1}")
    print()

    print("=== З візуалізацією ===")
    path2, dist2 = dijkstra(matrix, 0, 5, visualize=True, delay=1.0)
    print(f"Шлях: {path2}")
    print(f"Відстань: {dist2}")
