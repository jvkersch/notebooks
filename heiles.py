from math import log
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def hamiltonian(q1, q2, p1, p2):
    T = (p1**2 + p2**2) / 2
    V1 = (q1**2 + q2**2) / 2
    V2 = q1**2 * q2 - q2**3 / 3
    return T + V1 + V2


def gradient_hamiltonian(q1, q2, p1, p2):
    return np.array([q1 + 2 * q1 * q2, q2 + q1**2 - q2**2, p1, p2])


def feedback_rhs(q1, q2, p1, p2, K, H0):
    dh_dq1, dh_dq2, dh_dp1, dh_dp2 = gradient_hamiltonian(q1, q2, p1, p2)
    H = hamiltonian(q1, q2, p1, p2)
    KH_diff = K * (H - H0)

    return np.array(
        [
            -dh_dp1 - KH_diff * dh_dq1,
            -dh_dp2 - KH_diff * dh_dq2,
            dh_dq1 - KH_diff * dh_dp1,
            dh_dq2 - KH_diff * dh_dp2,
        ]
    )


def euler(f, tau, x0, tmax, args=None):
    if args is None:
        args = {}

    xs = []
    x = np.array(x0)
    t = 0

    while t < tmax:
        x = x + tau * f(*x, **args)
        xs.append((t, x))
        t += tau

    return xs


def chaotic_energy(K):
    x = (0.25, 0.5, 0, 0)
    tau = 0.1
    tmax = 200
    H0 = hamiltonian(*x)

    xs = euler(feedback_rhs, tau, x, tmax, args={"K": K, "H0": H0})
    err = [(t, log(abs(hamiltonian(*x) - H0))) for (t, x) in xs[5:]]
    return err


def chaotic_energy_rk45(K):
    x = (0.25, 0.5, 0, 0)
    tmax = 200
    H0 = hamiltonian(*x)

    def rhs(t, x):
        return feedback_rhs(*x, K, H0)

    res = solve_ivp(rhs, (0, tmax), x)
    xs = res.y.T
    ts = res.t
    err = [(t, log(abs(hamiltonian(*x) - H0))) for (t, x) in zip(ts[5:], xs[5:])]
    return err


def plot_energy_errors(errors, K):
    for method, err in errors.items():
        err = np.asarray(err)
        plt.plot(err[:, 0], err[:, 1], label=method)
    plt.xlabel("$t$")
    plt.ylabel(r"$\log|H(t) - H_0|$")
    plt.suptitle(f"Feedback method energy error ($K = {K}$)")
    plt.title("Henon-Heiles chaotic orbit $(q_1, q_2, p_1, p_2) = (0.25, 0.5, 0, 0)$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Check that the gradient of the Hamiltonian agrees with the expression
    # obtained from automatic differentiation.

    from jax import grad

    x = (1.1, 2.3, -0.6, 0.5)
    gradient = gradient_hamiltonian(*x)

    for i, dh in enumerate(gradient):
        jax_dh = grad(hamiltonian, i)(*x)
        assert abs(jax_dh - dh) < 1e-12, f"{jax_dh} != {dh} (component {i})"

    # Check that the Euler integrator works as expected.
    tau = 0.1
    f = lambda q, p: np.array([p, -q])
    xs = euler(f, tau, (1, 0), 0.5)
    # print(xs)

    # Chaotic orbit (figure 1 in the paper)
    K = 20
    err_euler = chaotic_energy(K)
    err_rk45 = chaotic_energy_rk45(K)
    errors = {"Euler": err_euler, "RK45": err_rk45}

    plot_energy_errors(errors, K)
