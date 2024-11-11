import numpy as np


def get_states():
    states = []
    for i in range(1, 5):
        for j in range(i + 1, 5):
            states.append((i, j))

    return states


def create_matrix(g: float):
    def v(bra: tuple[int, int], ket: tuple[int, int]) -> float:
        alpha, beta = bra
        gamma, delta = ket

        if alpha == gamma and beta == delta:
            return -g

        if alpha in ket or beta in ket:
            return -g / 2

        return 0

    def h0(bra: tuple[int, int], ket: tuple[int, int]) -> float:
        alpha, beta = bra
        gamma, delta = ket

        if alpha == gamma and beta == delta:
            return 2 * (alpha + beta - 2)

        return 0

    states = get_states()
    num_states = len(states)
    matrix = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(i, num_states):
            matrix[i, j] = h0(states[i], states[j]) + v(states[i], states[j])
            matrix[j, i] = matrix[i, j]

    return matrix


def get_energies(g: float) -> np.ndarray:
    matrix = create_matrix(g)
    return np.linalg.eigvalsh(matrix)


if __name__ == "__main__":
    g_values = np.linspace(0, 1, 100)
    energies = np.zeros((len(g_values), 6))

    for i, g in enumerate(g_values):
        energies[i] = get_energies(g)

    import matplotlib.pyplot as plt

    for i in range(6):
        plt.plot(g_values, energies[:, i], label=f"Energy level {i}")

    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()
