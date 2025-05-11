# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import itertools

# %%
n = 11

prng = np.random.RandomState(1234)
pos = prng.uniform(low=0, high=1, size=(n, 2))

#%%

def tours_undirected(n):
    fixed = 0
    for other in itertools.permutations(range(1, n)):
        tour = (fixed, *other)
        if tour <= (fixed, *reversed(other)):
            yield (*tour, fixed)

tours = list(tours_undirected(n))
print(f"n = {n}: {len(tours)} tours")

#%%
def f(pos, tour):
    cities = pos[tour, :]
    deltas = np.diff(cities, axis=0)
    distances = np.sum(deltas**2, axis=1)**0.5
    return np.sum(distances)


#%%
print(f"Computing distances for {len(tours)} tours")
distances = [f(pos, tour) for tour in tours]
print("Done")

#%%
avg_dist = np.mean(pdist(pos))
std_dist = np.std(pdist(pos))
print(f"Average distance between 2 nodes: {avg_dist}")

avg_tour = n * avg_dist
std_tour = (n*(1 - 2/(n-1)))**0.5 * std_dist
print(f"Average tour length: {avg_tour}")
print(f"Standard deviation: {std_tour}")

#%%
plt.hist(distances)
plt.axvline(x=avg_tour, color='red', linestyle='dashed', linewidth=2)
plt.axvline(x=avg_tour - 2*std_tour, color='gray', linestyle='dashed', linewidth=2)
plt.axvline(x=avg_tour + 2*std_tour, color='gray', linestyle='dashed', linewidth=2)

plt.show()
# %%
