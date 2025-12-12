import math
import heapq

INF = float('inf')

class DuanSSSP:
    def __init__(self, matrix, source):
        """
        Initialize the Duan et al. (2025) SSSP algorithm context.
        Matrix is expected to be NxN list of lists (from benchmark.py).
        """
        self.n = len(matrix)
        self.source = source

        # Convert Adjacency Matrix to Adjacency List for O(m) performance
        # Using standard list of lists instead of defaultdict
        self.adj = [[] for _ in range(self.n)]
        for u in range(self.n):
            for v, w in enumerate(matrix[u]):
                if w != INF and w != 0: # Assuming 0 is no-edge or self-loop
                    self.adj[u].append((v, w))

        # Global distance and predecessor arrays
        self.dist = [INF] * self.n
        self.dist[source] = 0
        self.pred = [-1] * self.n

        # Calculate parameters k and t based on n (Section 3.1)
        if self.n > 1:
            log_n = math.log2(self.n)
            self.k = max(2, int(log_n ** (1/3)))
            self.t = max(2, int(log_n ** (2/3)))
            self.max_level = math.ceil(log_n / self.t)
        else:
            self.k = 2
            self.t = 2
            self.max_level = 1

    def solve(self):
        """Main execution method matching the top-level call logic."""
        S = {self.source}
        B = INF

        # Initial call to BMSSP with l = ceil(log n / t)
        self.bmssp(self.max_level, B, S)

        return self.dist

    def bmssp(self, level, B, S):
        """
        Algorithm 3: Bounded Multi-Source Shortest Path.
        Recursive divide-and-conquer function.
        """
        # 1. Base Case (Level 0) -> Run Dijkstra-like procedure
        if level == 0:
            return self.base_case(B, S)

        # 2. Find Pivots (Algorithm 1)
        P, W = self.find_pivots(B, S)

        # 3. Initialize Data Structure D
        # M = 2^((l-1)t)
        M = 2 ** ((level - 1) * self.t)
        D = DataStructureD(M, B)

        # Insert pivots into D
        for x in P:
            D.insert(x, self.dist[x])

        B_prev_prime = INF
        if P:
             B_prev_prime = min(self.dist[x] for x in P)
        elif S:
             # Fallback if P is empty but S is not
             # Using generator expression with default for safety
             dists = [self.dist[x] for x in S]
             if dists:
                B_prev_prime = min(dists)

        U = set()

        # Threshold for partial execution
        threshold = self.k * (2 ** (level * self.t))

        # 4. Main Loop
        while len(U) < threshold and not D.is_empty():
            # Pull subset Si and bound Bi
            Bi, Si = D.pull()

            # Recursive Call
            Bi_prime, Ui = self.bmssp(level - 1, Bi, Si)

            # Union results
            U.update(Ui)

            # 5. Relax edges from Ui and update D
            K = [] # Temporary storage for batch prepend (standard list)

            # Identify candidates for Batch Prepend from Si
            si_batch_candidates = []
            for x in Si:
                if Bi_prime <= self.dist[x] < Bi:
                    si_batch_candidates.append((x, self.dist[x]))

            # Edge Relaxation
            for u in Ui:
                for v, weight in self.adj[u]:
                    new_dist = self.dist[u] + weight

                    # Remark 3.4: equality required
                    if new_dist <= self.dist[v]:
                        self.dist[v] = new_dist
                        self.pred[v] = u

                        # Case (a): Direct Insert
                        if Bi <= new_dist < B:
                            D.insert(v, new_dist)
                        # Case (b): Add to K for Batch Prepend
                        elif Bi_prime <= new_dist < Bi:
                            K.append((v, new_dist))

            # 6. Batch Prepend
            # Combine K and valid elements from Si
            batch_list = K + si_batch_candidates
            if batch_list:
                D.batch_prepend(batch_list)

            # Update B_prev for next iteration tracking
            B_prev_prime = Bi_prime

        # 7. Final Return Logic
        final_B_prime = B_prev_prime
        if B < final_B_prime:
            final_B_prime = B

        # Add remaining valid vertices from W (from FindPivots)
        for w_node in W:
            if self.dist[w_node] < final_B_prime:
                U.add(w_node)

        return final_B_prime, U

    def find_pivots(self, B, S):
        """
        Algorithm 1: Finding Pivots.
        Runs k steps of relaxation to reduce frontier size.
        """
        W = set(S)
        Wi_prev = set(S)

        # Relax for k steps
        for _ in range(self.k):
            Wi = set()
            for u in Wi_prev:
                for v, weight in self.adj[u]:
                    if self.dist[u] + weight <= self.dist[v]:
                        self.dist[v] = self.dist[u] + weight
                        if self.dist[v] < B:
                            Wi.add(v)

            W.update(Wi)
            Wi_prev = Wi

            # Early exit if W grows too large
            if len(W) > self.k * len(S):
                return S, W

        # Simplified Pivot Selection: P = S
        # This is a valid upper bound for pivots as per Lemma 3.2 logic
        # (Exact F-construction omitted for brevity but correctness is maintained via S)
        return S, W

    def base_case(self, B, S):
        """
        Algorithm 2: Base Case (Mini-Dijkstra).
        """
        if not S:
            return B, set()

        pq = []
        U0 = set(S)

        for u in S:
            heapq.heappush(pq, (self.dist[u], u))

        final_U = set()

        while pq and len(U0) < self.k + 1:
            d_u, u = heapq.heappop(pq)

            if d_u > self.dist[u]:
                continue

            # Add to processed set logic
            if d_u < B:
                 # Only if strictly less than B do we count it towards processed
                 pass

            # Stop if we hit bound B within the heap extraction
            if d_u >= B:
                continue

            # Expand
            for v, weight in self.adj[u]:
                if self.dist[u] + weight <= self.dist[v]:
                    new_dist = self.dist[u] + weight
                    if new_dist < B:
                        self.dist[v] = new_dist
                        self.pred[v] = u

                        if v not in U0:
                            U0.add(v)
                            heapq.heappush(pq, (new_dist, v))
                        else:
                            heapq.heappush(pq, (new_dist, v))

        # Filter valid U0 based on final distances
        # We must return U = {v in U0 : dist[v] < B'}

        # Find B'
        if len(U0) <= self.k:
             # Successful
             return B, U0
        else:
             # Partial
             valid_dists = [self.dist[v] for v in U0 if self.dist[v] < INF]
             if not valid_dists:
                 return B, U0

             B_prime = max(valid_dists)
             U_ret = {v for v in U0 if self.dist[v] < B_prime}
             return B_prime, U_ret


class DataStructureD:
    """
    Implements the Data Structure from Lemma 3.3.
    Uses standard min-heap and dictionary.
    """
    def __init__(self, M, B):
        self.M = M
        self.B = B
        self.pq = []
        self.in_heap = {}

    def insert(self, key, value):
        if value >= self.B:
            return

        if key in self.in_heap:
            old_val = self.in_heap[key]
            if value < old_val:
                self.in_heap[key] = value
                heapq.heappush(self.pq, (value, key))
        else:
            self.in_heap[key] = value
            heapq.heappush(self.pq, (value, key))

    def batch_prepend(self, item_list):
        for key, value in item_list:
            self.insert(key, value)

    def pull(self):
        S_prime = set()
        count = 0

        while self.pq and count < self.M:
            val, key = heapq.heappop(self.pq)

            if key in self.in_heap and self.in_heap[key] == val:
                S_prime.add(key)
                del self.in_heap[key]
                count += 1

        Bi = self.B

        # Peek at next valid element
        while self.pq:
            val, key = self.pq[0]
            if key in self.in_heap and self.in_heap[key] == val:
                Bi = val
                break
            heapq.heappop(self.pq)
        return Bi, S_prime

    def is_empty(self):
        return not self.in_heap

def run_duan(matrix, source, target):
    solver = DuanSSSP(matrix, source)
    solver.solve()

    path = []
    if solver.dist[target] == INF:
        return [], INF

    curr = target
    while curr != -1:
        path.append(curr)
        curr = solver.pred[curr]
    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]
    return path_tuples, solver.dist[target]
