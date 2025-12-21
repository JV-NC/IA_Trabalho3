import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import Item, Bin, item_heuristic, evaluate_individual, generate_random_items, save_plot, build_bin_from_individual, plot_bin_3d, assert_no_collisions, save_dataframe_csv, plot_history, plot_sensitivity

plot_path = 'output/plots/aco'
metrics_path = 'output/metrics/aco'

SEED = 42
random.seed(SEED)
BIN_W, BIN_H, BIN_D = (25,50,25)
items = generate_random_items(n=20, min_size=5, max_size=25)

#TODO: check collisions?
#TODO: check SEED and items generation

class AntColony:
    def __init__(
        self,
        items: list[Item],
        bin_dims: tuple[float, float, float],
        n_ants: int=30,
        n_best: int=5,
        n_iters: int=100,
        decay: float=0.2,
        alpha: float=1.0,
        beta: float=2.0
    ):
        self.items = items
        self.bin_dims = bin_dims
        self.n_items = len(items)

        # pheromone[item_id][rotation]
        self.pheromone = np.ones((self.n_items, 6))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iters = n_iters
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        #history for plot
        self.history_best: list[float] = []
        self.history_avg: list[float] = []
    
    def construct_solution(self)->list[tuple[int, int]]:
        available = set(range(self.n_items))
        solution = []
        bin = Bin(*self.bin_dims)

        while available:
            choices = []

            for item_id in available:
                for r in range(6):
                    tau = self.pheromone[item_id][r] ** self.alpha
                    eta = item_heuristic(
                        self.items[item_id],
                        bin,
                        type='volume_rate'
                    ) ** self.beta

                    score = tau * eta
                    if score > 0:
                        choices.append((item_id, r, score))

            if not choices:
                break

            total = sum(c[2] for c in choices)
            probs = [c[2] / total for c in choices]

            item_id, rotation = random.choices(
                [(c[0], c[1]) for c in choices],
                weights=probs,
                k=1
            )[0]

            item = self.items[item_id].copy()
            if bin.try_place_item_with_rotation(item, rotation):
                solution.append((item_id, rotation))
                available.remove(item_id)
            else:
                # if doesn't fit, remove form candidate list
                available.remove(item_id)

        return solution

    def evaluate(self, individual: list[tuple[int, int]])->float:
        bin = Bin(*self.bin_dims)
        return evaluate_individual(
            individual,
            self.items,
            bin,
            fitness_type='item_rejected'
        )

    def update_pheromones(self, solutions: list[tuple[int, int]]):
        self.pheromone *= (1 - self.decay)

        solutions.sort(key=lambda x: x[1], reverse=True)

        for sol, fit in solutions[:self.n_best]:
            for item_id, r in sol:
                self.pheromone[item_id][r] += fit

    def run(self):
        best_sol = None
        best_fit = -float('inf')

        for _ in range(self.n_iters):
            solutions = []
            fitness_values = []

            for _ in range(self.n_ants):
                sol = self.construct_solution()
                fit = self.evaluate(sol)

                solutions.append((sol, fit))
                fitness_values.append(fit)

                if fit > best_fit:
                    best_fit = fit
                    best_sol = sol

            self.update_pheromones(solutions)
            
            self.history_best.append(max(fitness_values))
            self.history_avg.append(sum(fitness_values) / len(fitness_values))

        return best_sol, best_fit

def main():
    aco = AntColony(items,(BIN_W, BIN_H, BIN_D),30,5,100,0.2,1,2)

    best_sol, best_fit = aco.run()

    print(f'Best fitness = {best_fit:.4f}')

    plot_history(
        aco.history_best,
        aco.history_avg,
        plot_path,
        filename='aco_fitness_evo.png',
        title='ACO – Fitness Evolution'
    )

    final_bin = build_bin_from_individual(
        best_sol,
        items,
        (BIN_W, BIN_H, BIN_D)
    )

    assert_no_collisions(final_bin)
    print(f'Fill ratio = {100 * final_bin.fill_ratio():.2f}%')

    plot_bin_3d(final_bin, plot_path, 'aco_bin_final_3d.png')

if __name__ == '__main__':
    main()