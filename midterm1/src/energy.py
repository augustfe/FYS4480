import numpy as np
import sympy as sp

from utils import read_elements, Z


def one_body_energy(F: int, n: int = 1) -> sp.Expr:
    """Compute the one-body energy

    Args:
        F (int): The Fermi-level
        n (int, optional): The energy level. Defaults to 1.

    Returns:
        sp.Expr: The one-body energy
    """
    energy = 0
    # Matching energy level n with the index
    for _ in range(1, F + 1):
        energy += -(Z**2) / (2 * n**2)

    return energy


def two_body_energy(values: np.ndarray, F: int, n: int = 3) -> sp.Expr:
    """Compute the two-body energy

    Args:
        values (np.ndarray): The matrix elements
        F (int): The Fermi-level
        n (int, optional): The size of the matrix. Defaults to 3.

    Returns:
        sp.Expr: The two-body energy
    """
    energy = 0
    # Matching energy level n with the index

    # Change later to an index set
    for i in range(F):
        for j in range(F):
            energy += values[i, j, i, j]

    return energy


def compute_groundstate_energy(F: int = 1, n: int = 1, n_max: int = 3) -> sp.Expr:
    """Compute the groundstate energy

    Args:
        F (int, optional): The Fermi-level. Defaults to 1.
        n (int, optional): The size of the matrix. Defaults to 3.
        n_max (int, optional): The size of the matrix. Defaults to 3.

    Returns:
        sp.Expr: The groundstate energy
    """
    values = read_elements(n_max)

    energy = 2 * one_body_energy(n) + two_body_energy(values, F, n)

    return energy
