"""спрощена штука"""
import math
def bellman_ford(adj, source):
    n = len(adj)
    INF = 10**18
    dist = [INF] * n
    dist[source] = 0
    for _ in range(n - 1):
        updated = False
        for u in range(n):
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
        if not updated:
            break

    return dist



def relax(u, v, w, dist, pred):
    if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        pred[v] = u
        return True
    return False


def base_case(adj, source_set, B, dist, pred, k):
    x = next(iter(source_set))
    visited = set([x])
    frontier = [x]
    U = set([x])
    while frontier and len(U) < k + 1:
        new_front = []
        for u in frontier:
            for v, w in adj[u]:
                if dist[u] + w < B and relax(u, v, w, dist, pred):
                    if v not in visited:
                        visited.add(v)
                        new_front.append(v)
                        U.add(v)
        frontier = new_front
    if len(U) <= k:
        return B, U
    else:
        mx = max(dist[v] for v in U)
        U2 = {v for v in U if dist[v] < mx}
        return mx, U2


def find_pivots(adj, S, B, dist, pred, k):
    W = set(S)
    frontier = set(S)
    for _ in range(k):
        new_frontier = set()
        for u in frontier:
            for v, w in adj[u]:
                if dist[u] + w < B and relax(u, v, w, dist, pred):
                    if v not in W:
                        W.add(v)
                        new_frontier.add(v)
        frontier = new_frontier
    if len(W) > k * len(S):
        return W, set(S)
    pivots = set()
    for s in S:
        cnt = 0
        cur = s
        while cur != -1 and cnt <= k:
            cnt += 1
            cur = pred[cur]
        if cnt >= k:
            pivots.add(s)

    return W, pivots


def bmssp(adj, S, B, level, dist, pred, k, t):
    if level == 0:
        return base_case(adj, S, B, dist, pred, k)
    W, P = find_pivots(adj, S, B, dist, pred, k)
    if not P:
        return B, W
    B_current = min(dist[x] for x in P)
    U_total = set()
    while True:
        Si = {x for x in P if dist[x] < B_current}
        if not Si:
            break
        B_i = B_current
        B_new, U_new = bmssp(adj, Si, B_i, level - 1, dist, pred, k, t)
        U_total |= U_new
        B_current = B_new
        if B_current >= B:
            break
    return B_current, U_total | {x for x in W if dist[x] < B_current}


def sssp_complex(adj, source):
    n = len(adj)
    dist = [10**18] * n
    pred = [-1] * n
    dist[source] = 0

    logn = math.log(n)
    k = max(2, int(logn ** (1/3)))
    t = max(2, int(logn ** (2/3)))
    depth = max(1, math.ceil(math.log2(n) / t))

    S = {source}
    B = float('inf')

    _, U = bmssp(adj, S, B, depth, dist, pred, k, t)
    return dist


def main():
    # 0 -> 1 (5), 0 -> 2 (1), 2 -> 1 (2), 1 -> 3 (3)
    adj = [
        [(1, 5), (2, 1)],  # з 0 в 1 і 2
        [(3, 3)],          # з 1 в 3
        [(1, 2)],          # з 2 в 1
        []                 # з 3 нікуди
    ]

    source = 0

    print("=== Bellman–Ford ===")
    bf_dist = bellman_ford(adj, source)
    for i, d in enumerate(bf_dist):
        print(f"dist({source} -> {i}) = {d}")

    print("\n=== Complex SSSP (Duan–Mao–Shu–Yin, simplified) ===")
    cmpl_dist = sssp_complex(adj, source)
    for i, d in enumerate(cmpl_dist):
        print(f"dist({source} -> {i}) = {d}")

if __name__ == "__main__":
    main()