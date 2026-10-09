# %%

import numpy as np
import matplotlib.pyplot as plt

# %%

# Inputs should be in the range [-1, 1]
def baby_siren(X):
    A, B, C, D = X.T
    t1 = np.arcsin((D - A) / 2) 
    t2 = np.arcsin((B - C) / 2)
    wx = (t1 + t2) / 2
    wy = (t1 - t2) / 2
    return np.column_stack((wx, wy))

def baby_siren_linearized(X):
    A, B, C, D = X.T
    wx = (D - A + B - C) / 4
    wy = (D - A - B + C) / 4
    return np.column_stack((wx, wy))

# %%

def make_circle():
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.column_stack(
        (np.sin(t), 
        np.sin(t),
        np.zeros_like(t),
        np.zeros_like(t)))
    return X

def make_square():
    X = np.zeros((100, 4))
    X[0:25, 0] = np.linspace(0, 1, 25)
    X[25:50, 0] = 1.0
    X[25:50, 1] = np.linspace(0, 1, 25)
    X[50:75, 0] = np.linspace(1, 0, 25)
    X[50:75, 1] = 1.0
    X[75:100, 1] = np.linspace(1, 0, 25)  
    return X

def make_eight():
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.column_stack(
        (np.sin(t), 
        np.sin(2 * t),
        np.zeros_like(t),
        np.zeros_like(t)))
    return X

def make_scribble():
    t = np.linspace(0, 4 * np.pi, 100)
    X = np.column_stack(
        (t / (4 * np.pi) * 2 - 1,
        np.sin(3 * t) * 0.5,
        np.zeros_like(t),
        np.zeros_like(t)))
    return X

# %%

# X = make_circle()
# X = make_square()
# X = make_eight()
X = make_scribble()

W = baby_siren(X)
W_linearized = baby_siren_linearized(X)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(X[:, 0], X[:, 1])
ax.plot(W[:, 0], W[:, 1])
ax.plot(W_linearized[:, 0], W_linearized[:, 1])

# %%
