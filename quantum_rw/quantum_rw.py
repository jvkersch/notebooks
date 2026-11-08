import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 50
SIZE = 2 * N_STEPS + 10
state = np.zeros((2, SIZE), dtype=np.complex64)
state[:, SIZE // 2] = np.array((1, 1j)) / 2**0.5

hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / 2**0.5
for n in range(N_STEPS):
    state = hadamard @ state
    state = np.vstack((np.roll(state[0], 1), np.roll(state[1], -1)))

norm_squared = np.sum(abs(state * state.conj()), axis=0)

fig, ax = plt.subplots()
ax.plot(norm_squared)
ax.set_xlabel("Position")
ax.set_ylabel("Probability")
fig.tight_layout()
plt.show()
plt.savefig("quantum_rw.pdf")
