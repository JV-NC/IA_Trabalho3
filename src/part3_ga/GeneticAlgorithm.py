import random

class GA:
    def __init__(self, pop_size, cx_rate, mut_rate, fitness_fn, create_ind, mutate, crossover,
    max_iters=1000, seed=42):
        random.seed(seed)
        self.pop = [create_ind() for _ in range(pop_size)]
        self.cx_rate, self.mut_rate = cx_rate, mut_rate
        self.fitness_fn = fitness_fn
        self.mutate, self.crossover = mutate, crossover
        self.max_iters = max_iters

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
            self.step()
            cand = max(self.pop, key=self.fitness_fn)
            if self.fitness_fn(cand) > self.fitness_fn(best):
                best = cand
        return best, self.fitness_fn(best)