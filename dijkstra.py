import numpy as np
import networkx as nx
import random
import heapq

# print("\033[H\033[J", end="")

def load_matrix(file_path):
    """
    Зчитує CSV файл з матрицею суміжності у NumPy масив.
    """
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Помилка зчитування файлу: {e}")
        return None
def dijkstra(matrix, start_node, end_node):
    """
    Args:
        matrix: 2D numpy array (NxN) - Матриця суміжності
        start_node: int
        end_node: int
    """
    n = matrix.shape[0]

    if start_node < 0 or start_node >= n or end_node < 0 or end_node >= n:
        return [], float('inf')

    dist = np.full(n, np.inf)
    dist[start_node] = 0

    parent = np.full(n, -1, dtype=int)

    pq = [(0, start_node)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u == end_node:
            break

        for v in range(n):
            weight = matrix[u][v]

            if weight > 0:
                new_dist = current_dist + weight

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    # Додаємо в чергу (O(log N))
                    heapq.heappush(pq, (new_dist, v))

    if dist[end_node] == np.inf:
        return [], float('inf')

    path = []
    curr = end_node
    while curr != -1:
        path.append(curr)
        curr = parent[curr]

    path.reverse()

    path_tuples = [(path[i], path[i+1]) for i in range(len(path)-1)]

    return list(map(lambda x: (int(x[0]), int(x[1])), path_tuples)), dist[end_node]

# RANDOM GRAPH GENERATION FOR TESTING ------------------------------------

# def generate_graph_matrix_guaranteed_path(n, density=0.3, max_weight=10):
#     """
#     Генерує граф, де ГАРАНТОВАНО існує шлях від вершини 0 до n-1.
#     """
#     matrix = np.zeros((n, n))

#     nodes = list(range(1, n - 1))
#     random.shuffle(nodes) # Перемішуємо їх

#     path_nodes = [0] + nodes + [n - 1]

#     for k in range(len(path_nodes) - 1):
#         u = path_nodes[k]
#         v = path_nodes[k + 1]
#         matrix[u][v] = random.randint(1, max_weight)
#     for i in range(n):
#         for j in range(n):
#             if i == j: continue
#             if matrix[i][j] > 0:
#                 continue
#             if random.random() < density:
#                 matrix[i][j] = random.randint(1, max_weight)

#     return matrix


# def check_path_weight_manually(matrix, path_tuples):
#     """
#     Проходиться по матриці згідно зі знайденим шляхом і сумує ваги.
#     """
#     calculated_weight = 0
#     for u, v in path_tuples:
#         w = matrix[u][v]
#         calculated_weight += w

#     return calculated_weight


# def run_experiments():
#     for _ in range(1):
#         n = random.randint(1,10)
#         density = random.random()
#         matrix = generate_graph_matrix_guaranteed_path(n, density)
#         print(matrix)

#         path, length = dijkstra(matrix, 0, n-1)
#         mx_path = nx.dijkstra_path(nx.from_numpy_array(matrix, create_using=nx.DiGraph), 0, n-1, weight='weight')
#         nx_path_tuples = [(mx_path[i], mx_path[i+1]) for i in range(len(mx_path)-1)]

#         print(f"Шлях: {path}  -- ")
#         print(f"Шлях: {nx_path_tuples}  -- ")
#         print(f"Довжина: {length}")

#         if path == nx_path_tuples:
#             print("✅ " + str(_) + " N:" +str(n-1) + " Rand: " + str(density))
#         else:
#             true_weight = check_path_weight_manually(matrix, nx_path_tuples)
#             my_weight = check_path_weight_manually(matrix, path)
#             print(f"🟥 DID NOT MATCH: {true_weight} -- {my_weight}")
# if __name__ == "__main__":
#     run_experiments()
