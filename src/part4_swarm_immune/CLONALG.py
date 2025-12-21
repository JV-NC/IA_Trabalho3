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
from utils import (Item, Bin, item_heuristic, evaluate_individual, generate_random_items, save_plot, build_bin_from_individual, plot_bin_3d, assert_no_collisions,
                   save_dataframe_csv, plot_history, plot_sensitivity, create_individual, fitness, mutate)

plot_path = 'output/plots/clonalg'
metrics_path = 'output/metrics/clonalg'

SEED = 42
random.seed(SEED)
BIN_W, BIN_H, BIN_D = (25,50,25)
items = generate_random_items(n=20, min_size=5, max_size=25)

POP_SIZES = [30, 50, 80]
N_SELECTS = [5, 10, 15]
CLONE_FACTORS = [4, 8, 12]
MAX_ITERS = [100, 150, 250]

PARAM_GRID = list(product(
    POP_SIZES,
    N_SELECTS,
    CLONE_FACTORS,
    MAX_ITERS
))

create = lambda: create_individual(len(items))

fitness_fn = lambda ind: fitness(ind, items, (BIN_W, BIN_H, BIN_D))

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
        best_fit = -float('inf')

        for _ in range(self.max_iters):
            self.step()
            current_best = max(self.population, key=self.fitness_fn)
            current_fit = self.fitness_fn(current_best)

            if current_fit > best_fit:
                best_fit = current_fit
                best_ind = current_best

        return best_ind, best_fit

def run_clonalg_config(
        config_id: int,
        pop_size: int,
        n_select: int,
        clone_factor: int,
        max_iters: int,
        seed: int
):
    random.seed(seed)
    np.random.seed(seed)

    start = time.perf_counter()

    clonalg = CLONALG(
        fitness_fn=fitness_fn,
        create_ind=create,
        mutate=mutate,
        pop_size=pop_size,
        n_select=n_select,
        clone_factor=clone_factor,
        max_iters=max_iters,
        seed=seed
    )

    best_ind, best_fit = clonalg.run()
    elapsed = time.perf_counter() - start

    return {
        'config_id': config_id,
        'pop_size': pop_size,
        'n_select': n_select,
        'clone_factor': clone_factor,
        'max_iters': max_iters,
        'best_fit': best_fit,
        'best_ind': best_ind,
        'time_sec': elapsed,
        'history_best': clonalg.history_best,
        'history_avg': clonalg.history_avg
    }

def main():
    start = time.perf_counter()

    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_clonalg_config)(i, pop, sel, cf, iters, SEED)
        for i, (pop, sel, cf, iters) in enumerate(PARAM_GRID))
    
    elapsed = time.perf_counter() - start
    print(f'Total CLONALG grid search time: {elapsed:.2f}s')

    total_item_volume = sum([item.volume() for item in items])
    bin_volume = (BIN_W * BIN_H * BIN_D)
    print(f'total item volume = {total_item_volume}\nbin volume = {bin_volume}\nitem/bin rate = {total_item_volume/bin_volume:.4f}')

    best_result = max(raw_results, key=lambda r: r['best_fit'])

    print('\n===== BEST CLONALG CONFIGURATION =====')
    for k in ['pop_size','n_select','clone_factor','max_iters']:
        print(f'{k}: {best_result[k]}')

    print(f'fitness = {best_result['best_fit']:.4f}')
    print(f'time (sec) = {best_result['time_sec']:.2f}')

    plot_history(best_result['history_best'], best_result['history_avg'], plot_path, filename='fitness_evo_iter.png', title='CLONALG – Fitness Evolution')

    final_bin = build_bin_from_individual(best_result['best_ind'],items,(BIN_W, BIN_H, BIN_D))
    assert_no_collisions(final_bin)

    print(f'Fill ratio = {100 * final_bin.fill_ratio():.2f}%')

    plot_bin_3d(final_bin, plot_path, 'bin_final_3d.png')

    df = pd.DataFrame(raw_results)

    fill_ratios = []
    for _, row in df.iterrows():
        final_bin = build_bin_from_individual(row['best_ind'],items,(BIN_W, BIN_H, BIN_D))
        fill_ratios.append(final_bin.fill_ratio())

    df['fill_ratio'] = fill_ratios

    save_dataframe_csv(df.drop(columns=['best_ind', 'history_best', 'history_avg']),metrics_path,'sensitivity_results.csv')

    # best_row = df.loc[df['fill_ratio'].idxmax()]
    # print(f'\nBest fill_ratio = {100 * best_row['fill_ratio']:.2f}%')

    mid = lambda lst: lst[len(lst)//2]
    fixed = {
        'pop_size': mid(POP_SIZES),
        'n_select': mid(N_SELECTS),
        'clone_factor': mid(CLONE_FACTORS),
        'max_iters': mid(MAX_ITERS)
    }

    plot_sensitivity(df, 'pop_size', fixed, plot_path, 'sens_pop_size.png')
    plot_sensitivity(df, 'n_select', fixed, plot_path, 'sens_n_select.png')
    plot_sensitivity(df, 'clone_factor', fixed, plot_path, 'sens_clone_factor.png')
    plot_sensitivity(df, 'max_iters', fixed, plot_path, 'sens_max_iters.png')


if __name__ == '__main__':
    main()