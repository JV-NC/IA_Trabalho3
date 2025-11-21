import numpy as np
rng = np.random.default_rng(42)

class PSO:
    def __init__(self, f, bounds, n=30, w=0.7, c1=1.4, c2=1.4, iters=500):
        self.f, self.bounds = f, np.array(bounds) # [(min,max), ...]
        dim = len(bounds)
        low, high = self.bounds[:,0], self.bounds[:,1]
        self.X = rng.uniform(low, high, size=(n, dim))
        self.V = rng.uniform(-abs(high-low), abs(high-low), size=(n, dim))*0.1
        self.pbest = self.X.copy()
        self.pbest_val = np.apply_along_axis(self.f, 1, self.X)
        gidx = np.argmin(self.pbest_val)
        self.gbest = self.pbest[gidx].copy()
        self.gbest_val = self.pbest_val[gidx]
        self.w, self.c1, self.c2, self.iters = w, c1, c2, iters

    def step(self):
        r1, r2 = rng.random(self.X.shape), rng.random(self.X.shape)
        self.V = self.w*self.V + self.c1*r1*(self.pbest - self.X) + self.c2*r2*(self.gbest -
        self.X)
        self.X = self.X + self.V
        self.X = np.minimum(np.maximum(self.X, self.bounds[:,0]), self.bounds[:,1]) # clamp
        vals = np.apply_along_axis(self.f, 1, self.X)
        improve = vals < self.pbest_val
        self.pbest[improve] = self.X[improve]
        self.pbest_val[improve] = vals[improve]
        if self.pbest_val.min() < self.gbest_val:
            idx = self.pbest_val.argmin()
            self.gbest, self.gbest_val = self.pbest[idx].copy(), self.pbest_val[idx]

    def run(self):
        for _ in range(self.iters):
            self.step()
        return self.gbest, self.gbest_val
