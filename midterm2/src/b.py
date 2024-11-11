import numpy as np


def get_states(max_level: int = 4):
    states = []
    for i in range(1, max_level + 1):
        for j in range(i + 1, max_level + 1):
            states.append((i, j))

    return states


def create_matrix(g: float, states: list[tuple[int, int]]) -> np.ndarray:
    def v(bra: tuple[int, int], ket: tuple[int, int]) -> float:
        alpha, beta = bra

        if bra == ket:
            return -g

        if alpha in ket or beta in ket:
            return -g / 2

        return 0

    def h0(bra: tuple[int, int], ket: tuple[int, int]) -> float:
        alpha, beta = bra

        if bra == ket:
            return 2 * (alpha + beta - 2)

        return 0

    num_states = len(states)
    matrix = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(i, num_states):
            matrix[i, j] = h0(states[i], states[j]) + v(states[i], states[j])
            matrix[j, i] = matrix[i, j]

    return matrix


def get_energy_from_states(g: float, states: list[tuple[int, int]]) -> float:
    matrix = create_matrix(g, states)
    return np.linalg.eigvalsh(matrix)


def get_all_energies_below(g: float, max_level: int = 4) -> np.ndarray:
    states = get_states(max_level)
    return get_energy_from_states(g, states)


if __name__ == "__main__":
    g_values = np.linspace(-1, 1, 100)

    max_level = 4
    num_states = max_level * (max_level - 1) // 2
    energies = np.zeros((len(g_values), num_states))

    for i, g in enumerate(g_values):
        energies[i] = get_all_energies_below(g, max_level)

    import matplotlib.pyplot as plt

    for i in range(num_states):
        plt.plot(g_values, energies[:, i], label=f"Energy level {i}")

    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()
