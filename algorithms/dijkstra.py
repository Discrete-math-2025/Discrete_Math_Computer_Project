import heapq

INF = float('inf')

def dijkstra(graph, start_node, end_node):
    """
    Standard Dijkstra algorithm.
    Adapted for Sparse Adjacency Lists.

    Args:
        graph: Adjacency List [[(v, w), ...], ...]
        start_node: int
        end_node:  int
        visualize: bool - Whether to show real-time visualization
        delay: float - Seconds between visualization steps

    Returns:
        path_tuples: list of edge tuples [(0,1), (1,3), ...]
        distance: total shortest distance
    """
    n = len(graph)

    if start_node < 0 or start_node >= n or end_node < 0 or end_node >= n:
        return [], INF

    dist = [INF] * n
    dist[start_node] = 0
    parent = [-1] * n
    pq = [(0, start_node)]

    # --- MAIN LOOP ---
    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u == end_node:
            break

        # Iterate over neighbors in Adjacency List directly
        for v, weight in graph[u]:
            new_dist = current_dist + weight

            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))
                # if visualize: draw_state(current_node=u, pq_items=pq)

    if dist[end_node] == INF:
        return [], INF

    # Reconstruct Path
    path = []
    curr = end_node
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]

    return path_tuples, dist[end_node]
