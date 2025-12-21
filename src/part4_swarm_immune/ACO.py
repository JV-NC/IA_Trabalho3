import random
import numpy as np
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

N_ANTS  = [20, 30, 40]
N_BEST  = [3, 5, 7]
N_ITERS = [80, 100, 150]
DECAYS  = [0.1, 0.2, 0.3]
ALPHAS  = [1.0, 2.0]
BETAS   = [1.0, 2.0]

PARAM_GRID = list(product(
    N_ANTS,
    N_BEST,
    N_ITERS,
    DECAYS,
    ALPHAS,
    BETAS
))

#TODO: check SEED and items generation, maybe create json or csv for items and run reading it

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

def run_aco_config(
        config_id: int,
        n_ants: int,
        n_best: int,
        n_iters: int,
        decay: float,
        alpha: float,
        beta: float,
        seed: int
):
    random.seed(seed)
    np.random.seed(seed)

    start = time.perf_counter()

    aco = AntColony(
        items=items,
        bin_dims=(BIN_W, BIN_H, BIN_D),
        n_ants=n_ants,
        n_best=n_best,
        n_iters=n_iters,
        decay=decay,
        alpha=alpha,
        beta=beta
    )

    best_sol, best_fit = aco.run()

    elapsed = time.perf_counter() - start

    return {
        'config_id': config_id,
        'n_ants': n_ants,
        'n_best': n_best,
        'n_iters': n_iters,
        'decay': decay,
        'alpha': alpha,
        'beta': beta,
        'best_fit': best_fit,
        'best_ind': best_sol,
        'time_sec': elapsed,
        'history_best': aco.history_best,
        'history_avg': aco.history_avg,
    }

def main():
    start = time.perf_counter()

    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_aco_config)(
            i, n_ants, n_best, n_iters, decay, alpha, beta, SEED)
            for i, (n_ants, n_best, n_iters, decay, alpha, beta) in enumerate(PARAM_GRID))
            
    elapsed = time.perf_counter() - start
    print(f'Total ACO grid search time: {elapsed:.2f}s')

    total_item_volume = sum([item.volume() for item in items])
    bin_volume = (BIN_W * BIN_H * BIN_D)
    print(f'total item volume = {total_item_volume}\nbin volume = {bin_volume}\nitem/bin rate = {total_item_volume/bin_volume:.4f}')

    best_result = max(raw_results, key=lambda r: r['best_fit'])

    print('\n===== BEST ACO CONFIGURATION =====')
    for k in ['n_ants','n_best','n_iters','decay','alpha','beta']:
        print(f'{k}: {best_result[k]}')

    print(f'fitness = {best_result["best_fit"]:.4f}')
    print(f'time (sec) = {best_result["time_sec"]:.2f}')

    plot_history(best_result['history_best'], best_result['history_avg'], plot_path, filename='fitness_evo_iter.png', title='ACO – Fitness Evolution')

    final_bin = build_bin_from_individual(best_result['best_ind'],items,(BIN_W, BIN_H, BIN_D))
    assert_no_collisions(final_bin)

    print(f'Fill ratio = {100 * final_bin.fill_ratio():.2f}%')

    plot_bin_3d(final_bin, plot_path, 'bin_final_3d.png')

    df = pd.DataFrame(raw_results)

    fill_ratios = []
    for _, row in df.iterrows():
        bin_final = build_bin_from_individual(row['best_ind'],items,(BIN_W, BIN_H, BIN_D))
        fill_ratios.append(bin_final.fill_ratio())

    df['fill_ratio'] = fill_ratios

    save_dataframe_csv(df.drop(columns=['best_ind', 'history_best', 'history_avg']),metrics_path,'sensitivity_results.csv')

    # best_row = df.loc[df['fill_ratio'].idxmax()]
    # print(f'\nBest fill_ratio = {100 * best_row["fill_ratio"]:.2f}%')

    mid = lambda lst: lst[len(lst)//2]
    fixed = {
        'n_ants': mid(N_ANTS),
        'n_best': mid(N_BEST),
        'n_iters': mid(N_ITERS),
        'decay': mid(DECAYS),
        'alpha': mid(ALPHAS),
        'beta': mid(BETAS),
    }

    plot_sensitivity(df, 'n_ants', fixed, plot_path, 'sens_n_ants.png')
    plot_sensitivity(df, 'n_iters', fixed, plot_path, 'sens_n_iters.png')
    plot_sensitivity(df, 'decay', fixed, plot_path, 'sens_decay.png')
    plot_sensitivity(df, 'alpha', fixed, plot_path, 'sens_alpha.png')
    plot_sensitivity(df, 'beta', fixed, plot_path, 'sens_beta.png')

if __name__ == '__main__':
    main()