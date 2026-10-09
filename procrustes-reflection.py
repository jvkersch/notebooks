# %%
from scipy.spatial import procrustes
import numpy as np

# %%
x = np.array([
    [0, 0],
    [2, 0],
    [0, 1]
], dtype=float)
y = x.copy()
y[:, 1] *= -1.0

# %%
sx, sy, w = procrustes(x, y)
print(sx)
print(sy)
print(w)
# %%
