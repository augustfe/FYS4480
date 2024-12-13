from __future__ import annotations  # noqa: CPY001, D100, INP001

import numpy as np

default_g = np.linspace(-1, 1, 101)


def get_states(max_level: int = 4) -> list[tuple[int, int]]:
    """Generate a list of states represented as tuples of integers.

    Args:
        max_level (int): The maximum level to consider for generating states.

    Returns:
        list[tuple[int, int]]: A list of states represented as tuples of integers.

    """
    states = []
    for i in range(1, max_level + 1):
        for j in range(i + 1, max_level + 1):
            states.extend([(i, j)])

    return states


def create_matrix(g: float, states: list[tuple[int, int]]) -> np.ndarray:
    """Create the Hamiltonian matrix for the given interaction strength and states.

    Args:
        g (float): Interaction strength.
        states (list[tuple[int, int]]): List of states.

    Returns:
        np.ndarray: The Hamiltonian matrix.

    """

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
    """Calculate the energy from the given interaction strength and states.

    Args:
        g (float): Interaction strength.
        states (list[tuple[int, int]]): List of states.

    Returns:
        float: The calculated energy.

    """
    matrix = create_matrix(g, states)
    return np.linalg.eigvalsh(matrix)


def get_coeffs_from_states(g: float, states: list[tuple[int, int]]) -> np.ndarray:
    """Compute the coefficients from the given interaction strength and states.

    Args:
        g (float): Interaction strength.
        states (list[tuple[int, int]]): List of states.

    Returns:
        np.ndarray: Coefficients for the given states.

    """
    matrix = create_matrix(g, states)
    return np.linalg.eigh(matrix)[1][:, 0]


def get_all_energies_below(g: float, max_level: int = 4) -> np.ndarray:
    """Calculate all energies below a certain level for a given interaction strength.

    Args:
        g (float): Interaction strength.
        max_level (int): The maximum level to consider for generating states.

    Returns:
        np.ndarray: Array of calculated energies.

    """
    states = get_states(max_level)
    return get_energy_from_states(g, states)


def FCI_energies(g_values: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
    """Calculate the Full Configuration Interaction (FCI) energies.

    Returns:
        np.ndarray: Array of FCI energies for different interaction strengths.

    """
    if g_values is None:
        g_values = default_g
    energies = np.zeros((len(g_values), 6))
    for i, g in enumerate(g_values):
        energies[i] = get_all_energies_below(g, 4)

    return energies


def CI_energies(g_values: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
    """Calculate the Configuration Interaction (CI) energies.

    Returns:
        np.ndarray: Array of CI energies for different interaction strengths.

    """
    if g_values is None:
        g_values = default_g
    energies = np.zeros((len(g_values), 5))
    states = get_states(4)[:-1]
    for i, g in enumerate(g_values):
        energies[i] = get_energy_from_states(g, states)

    return energies


def CI_coeffs(g_values: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
    r"""Calculate the Configuration Interaction (CI) coefficients.

    This refers to the coefficients
        C_0^i | \Phi_i >
    for 2p-2h states | \Phi_i >.

    Returns:
        np.ndarray: Array of CI coefficients for different interaction strengths.

    """
    if g_values is None:
        g_values = default_g
    coeffs = np.zeros((len(g_values), 5))
    states = get_states(4)[:-1]
    for i, g in enumerate(g_values):
        coeffs[i] = get_coeffs_from_states(g, states)

    return coeffs[:, 1:]


if __name__ == "__main__":
    """
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
    """

    import matplotlib.pyplot as plt

    states = get_states(4)[:-1]
    g_values = np.linspace(-1, 1, 100)
    coeffs = np.zeros((len(g_values), len(states)))
    for i, g in enumerate(g_values):
        coeffs[i] = get_coeffs_from_states(g, states)

    for i in range(len(states)):
        plt.plot(g_values, coeffs[:, i], label=f"Coefficient {i}")

    plt.legend()
    plt.show()
