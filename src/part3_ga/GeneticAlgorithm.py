import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import Item, Bin, evaluate_individual, generate_random_items, save_plot, build_bin_from_individual, plot_bin_3d, assert_no_collisions, save_dataframe_csv, plot_history, plot_sensitivity

plot_path = 'output/plots/ga'
metrics_path = 'output/metrics/ga'

SEED = 42
random.seed(SEED)
BIN_W, BIN_H, BIN_D = (25,50,25)
items = generate_random_items(n=20, min_size=5, max_size=25)

POP_SIZES = [20, 50, 100]
CX_RATES  = [0.6, 0.8, 0.95]
MUT_RATES = [0.05, 0.2, 0.4]
MAX_ITERS = [150, 300, 500]

PARAM_GRID = list(product(
    POP_SIZES,
    CX_RATES,
    MUT_RATES,
    MAX_ITERS
))

def create_individual(num_items: int) -> list[tuple[int, int]]:
    """Create individual with a certain number of items and rotations"""
    ids = list(range(num_items))
    random.shuffle(ids)

    individual = []
    for i in ids:
        rotation = random.randint(0, 5)
        individual.append((i, rotation))

    return individual

def fitness(individual: list[tuple[int, int]])->float:
    """Instantiate a bin and evaluate individual using it"""
    bin = Bin(BIN_W,BIN_H,BIN_D)
    return evaluate_individual(
        individual,
        items,
        bin,
        'item_rejected'
    )

def ox_crossover(p1, p2):
    """Crossover parents genes, creating two oposite children"""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    def make_child(p1, p2):
        child = [None] * size
        child[a:b] = p1[a:b]

        used = {gene[0] for gene in child if gene is not None}
        pos = b

        for gene in p2:
            if gene[0] not in used:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1

        return child

    return make_child(p1, p2), make_child(p2, p1)

def mutate_swap(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Mutate swaping individual's genes"""
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def mutate_rotation(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Mutate individual changing the rotation"""
    i = random.randrange(len(individual))
    item_id, _ = individual[i]
    individual[i] = (item_id, random.randint(0, 5))
    return individual

def mutate(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    """Choose between swap or rotation"""
    if random.random() < 0.5:
        return mutate_swap(individual)
    else:
        return mutate_rotation(individual)


class GA:
    def __init__(self, pop_size, cx_rate, mut_rate, fitness_fn, create_ind, mutate, crossover,
    max_iters=1000):
        self.pop = [create_ind() for _ in range(pop_size)]
        self.cx_rate, self.mut_rate = cx_rate, mut_rate
        self.fitness_fn = fitness_fn
        self.mutate, self.crossover = mutate, crossover
        self.max_iters = max_iters

        #history for evo plot
        self.history_best: list[float] = []
        self.history_avg: list[float] = []

    def select(self, k=2):
        cand = random.sample(self.pop, k)
        cand.sort(key=self.fitness_fn, reverse=True)
        return cand[0]

    def step(self):
        new_pop = []
        while len(new_pop) < len(self.pop):
            p1, p2 = self.select(), self.select()
            c1, c2 = p1[:], p2[:]
            if random.random() < self.cx_rate:
                c1, c2 = self.crossover(c1, c2)
            if random.random() < self.mut_rate:
                c1 = self.mutate(c1)
            if random.random() < self.mut_rate:
                c2 = self.mutate(c2)
            new_pop += [c1, c2]
        self.pop = new_pop[:len(self.pop)]

    def run(self):
        best = max(self.pop, key=self.fitness_fn)
        for _ in range(self.max_iters):
            fitness_values = [self.fitness_fn(ind) for ind in self.pop]

            #save best and avg 's gen in history
            gen_best = max(fitness_values)
            gen_avg = sum(fitness_values) / len(fitness_values)
            self.history_best.append(gen_best)
            self.history_avg.append(gen_avg)

            self.step()
            cand = max(self.pop, key=self.fitness_fn)
            if self.fitness_fn(cand) > self.fitness_fn(best):
                best = cand
        return best, self.fitness_fn(best)

def run_ga_config(
    config_id: int,
    pop_size: int,
    cx_rate: float,
    mut_rate: float,
    max_iters: int,
    seed: int,
):
    """Run ga with specific configuration and returns its execution time, best_ind, best_fit, history and the parameters in a dict"""
    random.seed(seed)

    start = time.perf_counter()

    ga = GA(
        pop_size=pop_size,
        cx_rate=cx_rate,
        mut_rate=mut_rate,
        fitness_fn=fitness,
        create_ind=lambda: create_individual(len(items)),
        mutate=mutate,
        crossover=ox_crossover,
        max_iters=max_iters,
    )

    best_ind, best_fit = ga.run()

    elapsed = time.perf_counter() - start

    return {
        'config_id': config_id,
        'pop_size': pop_size,
        'cx_rate': cx_rate,
        'mut_rate': mut_rate,
        'max_iters': max_iters,
        'best_fit': best_fit,
        'best_ind': best_ind,
        'time_sec': elapsed,
        'history_best': ga.history_best,
        'history_avg': ga.history_avg,
    }

def main():
    start = time.perf_counter()

    #Appling parallelism with joblib
    raw_results = Parallel(n_jobs=-1,backend='loky')(
        delayed(run_ga_config)(
            i,pop,cx,mut,iters,seed=SEED)
            for i, (pop, cx, mut, iters) in enumerate(PARAM_GRID))
    
    elapsed = time.perf_counter()-start
    print(f'Total GA grid search time: {elapsed:.2f}s')

    # for i, item in enumerate(items):
    #     print(f'item {i}: w = {item.w}, h = {item.h}, d = {item.d}')
    total_item_volume = sum([item.volume() for item in items])
    bin_volume = (BIN_W * BIN_H * BIN_D)
    print(f'total item volume = {total_item_volume}\nbin volume = {bin_volume}\nitem/bin rate = {total_item_volume/bin_volume:.4f}')

    best_result =  max(raw_results,key=lambda r: r['best_fit'])

    print('\n===== BEST CONFIGURATION =====')
    for k in ['pop_size','cx_rate','mut_rate','max_iters']:
        print(f'{k}: {best_result[k]}')

    print(f'fitness = {best_result['best_fit']:.4f}')

    plot_history(best_result['history_best'],best_result['history_avg'],plot_path,'ga_fitness_evo.png',title='GA – Fitness Evolution')

    final_bin = build_bin_from_individual(best_result['best_ind'], items, (BIN_W, BIN_H, BIN_D))
    assert_no_collisions(final_bin)
    print(f'fill ratio = {100*final_bin.fill_ratio():.2f}%')
    print(f'time (sec) = {best_result['time_sec']:.2f}')
    plot_bin_3d(final_bin,plot_path,'bin_final_3d.png')

    df = pd.DataFrame(raw_results)

    fill_ratios = []
    for _, row in df.iterrows():
        final_bin = build_bin_from_individual(
            row['best_ind'],
            items,
            (BIN_W, BIN_H, BIN_D)
        )
        fill_ratios.append(final_bin.fill_ratio())
    
    df['fill_ratio'] = fill_ratios
    
    save_dataframe_csv(df.drop(columns=['best_ind', 'history_best', 'history_avg']), metrics_path, 'ga_sensitivity_results.csv')
    best_row = df.loc[df["fill_ratio"].idxmax()]
    # print(f'\nbest fill_ratio = {100*best_row['fill_ratio']:.2f}%')
    mid_value = lambda lst: lst[len(lst)//2]
    fixed = {
        'pop_size': mid_value(POP_SIZES),
        'cx_rate': mid_value(CX_RATES),
        'mut_rate': mid_value(MUT_RATES),
        'max_iters': mid_value(MAX_ITERS)
    }

    plot_sensitivity(df, 'pop_size', fixed, plot_path, 'sens_pop_size.png', title_prefix='GA Sensitivity')
    plot_sensitivity(df, 'cx_rate', fixed, plot_path, 'sens_cx_rate.png', title_prefix='GA Sensitivity')
    plot_sensitivity(df, 'mut_rate', fixed, plot_path, 'sens_mut_rate.png', title_prefix='GA Sensitivity')
    plot_sensitivity(df, 'max_iters', fixed, plot_path, 'sens_max_iters.png', title_prefix='GA Sensitivity')

if __name__ == '__main__':
    main()