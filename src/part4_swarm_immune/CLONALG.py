import numpy as np
import random
from typing import Callable, List, Tuple
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import Item, Bin, item_heuristic, evaluate_individual, generate_random_items, save_plot, build_bin_from_individual, plot_bin_3d, assert_no_collisions, save_dataframe_csv, plot_history, plot_sensitivity

plot_path = 'output/plots/clonalg'
metrics_path = 'output/metrics/clonalg'

SEED = 42
random.seed(SEED)
BIN_W, BIN_H, BIN_D = (25,50,25)
items = generate_random_items(n=20, min_size=5, max_size=25)

#TODO: make commons for GA, ACO and CLONALG (fitness, create, mutate, etc)
#TODO: clean create lambda

def create_individual(num_items: int) -> list[tuple[int, int]]:
    """Create individual with a certain number of items and rotations"""
    ids = list(range(num_items))
    random.shuffle(ids)

    individual = []
    for i in ids:
        rotation = random.randint(0, 5)
        individual.append((i, rotation))

    return individual

create = lambda: create_individual(len(items))

def fitness(individual: list[tuple[int, int]])->float:
    """Instantiate a bin and evaluate individual using it"""
    bin = Bin(BIN_W,BIN_H,BIN_D)
    return evaluate_individual(
        individual,
        items,
        bin,
        'fill_ratio'
    )

def mutate_swap(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Mutate swaping individual's genes"""
    ind = individual[:]
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutate_rotation(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Mutate individual changing the rotation"""
    ind = individual[:]
    i = random.randrange(len(ind))
    item_id, _ = ind[i]
    ind[i] = (item_id, random.randint(0, 5))
    return ind

def mutate(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Choose between swap or rotation"""
    if random.random() < 0.5:
        return mutate_swap(individual)
    else:
        return mutate_rotation(individual)

class CLONALG:
    def __init__(
        self,
        fitness_fn: Callable,
        create_ind: Callable,
        mutate: Callable,
        pop_size: int = 50,
        n_select: int = 10,
        clone_factor: int = 5,
        max_iters: int = 200,
        seed: int = 42
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.fitness_fn = fitness_fn
        self.create_ind = create_ind
        self.mutate = mutate

        self.pop_size = pop_size
        self.n_select = n_select
        self.clone_factor = clone_factor
        self.max_iters = max_iters

        self.population = [self.create_ind() for _ in range(self.pop_size)]

        #history
        self.history_best: List[float] = []
        self.history_avg: List[float] = []

    def _clone_and_hypermutate(self, individual, rank):
        """
        Rank-based cloning + hypermutation
        rank = 1 => best indivudal
        """
        n_clones = max(1, self.clone_factor // rank)
        clones = []

        for _ in range(n_clones):
            clone = individual[:]
            clone = self.mutate(clone)
            clones.append(clone)

        return clones
    
    def step(self):
        #Avaliation
        fitness = np.array([self.fitness_fn(ind) for ind in self.population])

        #Statistics
        self.history_best.append(fitness.max())
        self.history_avg.append(fitness.mean())

        #Pick bests
        idx = np.argsort(fitness)[::-1][:self.n_select]
        selected = [self.population[i] for i in idx]

        #Cloning and hypermutation
        new_population = []
        for rank, ind in enumerate(selected, start=1):
            new_population.extend(self._clone_and_hypermutate(ind, rank))

        #Diversity, complete indivudual with random ones
        while len(new_population) < self.pop_size:
            new_population.append(self.create_ind())

        self.population = new_population[:self.pop_size]
    
    def run(self):
        best_ind = None
        best_fit = -float("inf")

        for _ in range(self.max_iters):
            self.step()
            current_best = max(self.population, key=self.fitness_fn)
            current_fit = self.fitness_fn(current_best)

            if current_fit > best_fit:
                best_fit = current_fit
                best_ind = current_best

        return best_ind, best_fit
    
def main():
    clonalg = CLONALG(
        fitness_fn=fitness,
        create_ind=create,
        mutate=mutate,
        pop_size=50,
        n_select=10,
        clone_factor=8,
        max_iters=150
        )
    
    best_ind, best_fit = clonalg.run()
    print(f"Best fitness = {best_fit:.4f}")
    
    plot_history(clonalg.history_best,clonalg.history_avg,plot_path=plot_path,filename='fitness_evo_iter.png',title='CLONALG – Fitness Evolution')

    final_bin = build_bin_from_individual(best_ind,items,(BIN_W, BIN_H, BIN_D))

    assert_no_collisions(final_bin)
    print(f"Fill ratio = {100 * final_bin.fill_ratio():.2f}%")

    plot_bin_3d(final_bin, plot_path, 'bin_final_3d.png')

if __name__ == "__main__":
    main()