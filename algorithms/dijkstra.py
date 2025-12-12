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

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u == end_node:
            break

        for v, weight in graph[u]:
            new_dist = current_dist + weight

            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    if dist[end_node] == INF:
        return [], INF

    path = []
    curr = end_node
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]

    return path_tuples, dist[end_node]


def dijkstra_unoptimized(graph, start_node, end_node):
    """
    Unoptimized Dijkstra algorithm using Linear Search.
    Complexity: O(N^2)

    Args:
        graph: Adjacency List [[(v, w), ...], ...]
        start_node: int
        end_node:  int

    Returns:
        path_tuples: list of edge tuples
        distance: total shortest distance
    """
    n = len(graph)

    if start_node < 0 or start_node >= n or end_node < 0 or end_node >= n:
        return [], INF

    dist = [INF] * n
    dist[start_node] = 0
    parent = [-1] * n
    visited = [False] * n

    for _ in range(n):

        u = -1
        min_val = INF
        for i in range(n):
            if not visited[i] and dist[i] < min_val:
                min_val = dist[i]
                u = i

        if u == -1 or dist[u] == INF:
            break

        visited[u] = True

        if u == end_node:
            break

        for v, weight in graph[u]:
            if not visited[v]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u

    if dist[end_node] == INF:
        return [], INF

    path = []
    curr = end_node
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]

    return path_tuples, dist[end_node]
