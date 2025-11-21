import numpy as np
rng = np.random.default_rng(42)

class CLONALG:
    def __init__(self, f, create, mutate, pop=50, clones=5, sel=10, iters=200):
        self.f, self.create, self.mutate = f, create, mutate
        self.pop = [create() for _ in range(pop)]
        self.clones, self.sel, self.iters = clones, sel, iters

def run(self):
    best, bestv = None, np.inf
    for _ in range(self.iters):
        vals = np.array([self.f(x) for x in self.pop])
        idx = vals.argsort()[:self.sel] # menor é melhor
        selected = [self.pop[i] for i in idx]
        new = []
        for rank, s in enumerate(selected, start=1):
            ncl = max(1, self.clones//rank)
            for _ in range(ncl):
                new.append(self.mutate(s))
        while len(new) < len(self.pop):
            new.append(self.create())
        self.pop = new
        vmin = vals.min()
        if vmin < bestv:
            bestv, best = vmin, selected[0]
    return best, bestv