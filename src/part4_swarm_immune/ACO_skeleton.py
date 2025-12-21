import numpy as np
import matplotlib.pyplot as plt

# Número de cidades
num_cities = 200
cities = np.random.rand(num_cities, 2)

# Matriz de distâncias
distance_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(distance_matrix, np.inf)  # Evitar divisão por zero

# Visualização
plt.scatter(cities[:, 0], cities[:, 1], color="blue")
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i))
plt.title("Mapa das Cidades")
plt.show()

# Classe AntColony
class AntColony:
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations,
                 decay, alpha=1, beta=2):
        self.distance_matrix = distance_matrix
        self.pheromone = np.ones(self.distance_matrix.shape) / len(distance_matrix)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for _ in range(self.n_iterations):
            all_paths = self.construct_paths()
            self.spread_pheromones(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def construct_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.generate_path(0)
            all_paths.append((path, self.path_distance(path)))
        return all_paths

    def generate_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distance_matrix) - 1):
            move = self.pick_move(self.pheromone[prev], self.distance_matrix[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        heuristic = 1.0 / dist
        heuristic[np.isinf(heuristic)] = 0

        row = (pheromone ** self.alpha) * (heuristic ** self.beta)

        if row.sum() == 0:
            unvisited = list(set(range(len(self.distance_matrix))) - visited)
            move = np.random.choice(unvisited)
        else:
            move = np.random.choice(range(len(self.distance_matrix)), p=row / row.sum())

        return move

    def path_distance(self, path):
        return sum(self.distance_matrix[i][j] for i, j in path)

    def spread_pheromones(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / dist

# Exemplo de uso
ant_colony = AntColony(distance_matrix, n_ants=70, n_best=5, n_iterations=100, decay=0.3)
best_path = ant_colony.run()
print("Melhor caminho:", best_path)

# Plotar novamente o mapa com o melhor caminho
best_edges, best_distance = best_path

plt.figure(figsize=(6, 6))
plt.scatter(cities[:, 0], cities[:, 1], color="blue")

# Plotar rótulos
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i))

# Desenhar o caminho
for i, j in best_edges:
    x_coords = [cities[i][0], cities[j][0]]
    y_coords = [cities[i][1], cities[j][1]]
    plt.plot(x_coords, y_coords, color="red")

plt.title(f"Melhor Caminho Encontrado (distância = {best_distance:.4f})")
plt.show()
