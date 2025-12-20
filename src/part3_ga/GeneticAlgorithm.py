import random
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import Item, Bin, evaluate_individual, generate_random_items, save_plot, build_bin_from_individual, plot_bin_3d, assert_no_collisions

#TODO: implement others fitness functions (height, items not used, etc)
#TODO: implement sensibility plot, with parameters changes.
#TODO: check items colision for plot and fitness

plot_path = 'output/plots/ga'

SEED = 42
random.seed(SEED)
BIN_W, BIN_H, BIN_D = (25,50,25)
items = generate_random_items(n=20, min_size=5, max_size=25)

def create_individual(num_items: int) -> list[tuple[int, int]]:
    ids = list(range(num_items))
    random.shuffle(ids)

    individual = []
    for i in ids:
        rotation = random.randint(0, 5)
        individual.append((i, rotation))

    return individual

def fitness(individual: list[tuple[int, int]])->float:
    bin = Bin(BIN_W,BIN_H,BIN_D)
    return evaluate_individual(
        individual,
        items,
        bin
    )

def ox_crossover(p1, p2):
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
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def mutate_rotation(individual: list[tuple[int, int]])->list[tuple[int, int]]:
    i = random.randrange(len(individual))
    item_id, _ = individual[i]
    individual[i] = (item_id, random.randint(0, 5))
    return individual

def mutate(individual: list[tuple[int, int]])->list[tuple[int, int]]:
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

def plot_history(history_best: list[float], history_avg: list[float], filename: str='fitness_evo_gen.png')->None:
    plt.figure()
    plt.plot(history_best, label='Best fitness')
    plt.plot(history_avg, label='Average fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (fill ratio)')
    plt.title('GA – Fitness evolution per generation')
    plt.legend()
    plt.grid(True)
    save_plot(plot_path, filename)

def main():
    ga = GA(
        pop_size=50,
        cx_rate=0.8,
        mut_rate=0.2,
        fitness_fn=fitness,
        create_ind=lambda: create_individual(len(items)),
        mutate=mutate,
        crossover=ox_crossover,
        max_iters=300,
    )
    # for i, item in enumerate(items):
    #     print(f'item {i}: w = {item.w}, h = {item.h}, d = {item.d}')
    total_item_volume = sum([item.volume() for item in items])
    bin_volume = (BIN_W * BIN_H * BIN_D)
    print(f'total item volume = {total_item_volume}\nbin volume = {bin_volume}\nitem/bin rate = {total_item_volume/bin_volume:.4f}')
    best_ind, best_fit = ga.run()
    print(f'best fitness = {best_fit:.4f}')

    plot_history(ga.history_best,ga.history_avg)

    final_bin = build_bin_from_individual(best_ind, items, (BIN_W, BIN_H, BIN_D))
    assert_no_collisions(final_bin)
    plot_bin_3d(final_bin,plot_path,'bin_final_3d.png')

if __name__ == '__main__':
    main()