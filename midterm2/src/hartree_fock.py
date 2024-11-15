from __future__ import annotations  # noqa: CPY001, D100, INP001

import numpy as np

num_orbitals = 4
spin_degen = 2


def swap_spins(rho: np.ndarray) -> np.ndarray:
    """Swap the spin indices of the given density matrix.

    Args:
        rho (np.ndarray): The density matrix to swap spins for.

    Returns:
        np.ndarray: The density matrix with swapped spin indices.

    """
    new_rho = rho.reshape(
        (num_orbitals, spin_degen, num_orbitals, spin_degen),
    ).swapaxes(1, 2)
    flipped = np.copy(new_rho)
    flipped[..., 0, 0], flipped[..., 1, 1] = new_rho[..., 1, 1], new_rho[..., 0, 0]
    flipped[..., 0, 1], flipped[..., 1, 0] = new_rho[..., 1, 0], new_rho[..., 0, 1]
    new_rho = flipped

    return new_rho.swapaxes(1, 2).reshape(
        (num_orbitals * spin_degen, num_orbitals * spin_degen),
    )


def get_rho(coeffs: np.ndarray) -> np.ndarray:
    """Calculate the density matrix from the coefficient matrix.

    Args:
        coeffs (np.ndarray): The coefficient matrix.

    Returns:
        np.ndarray: The density matrix.

    """
    return coeffs.conj() @ coeffs.T


def run(
    g: float,
    max_iter: int = 100,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the Hartree-Fock calculation.

    Args:
        g (float): Interaction strength.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Convergence tolerance. Defaults to 1e-14.

    Returns:
        tuple[np.ndarray, np.ndarray]: The final energies and coefficient matrix.

    """
    particles = np.arange(1, num_orbitals + 1).repeat(spin_degen)
    h = np.diag(particles - 1)

    coeffs = np.eye(num_orbitals * spin_degen, dtype=complex)

    def iteration(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho = get_rho(coeffs)
        fock_matrix = h - 0.5 * g * swap_spins(rho)
        energies, new_coeffs = np.linalg.eigh(fock_matrix)
        return energies, new_coeffs

    old_energies = np.zeros(num_orbitals * spin_degen)
    for itr in range(max_iter):
        energies, coeffs = iteration(coeffs)

        diff = np.linalg.norm(energies - old_energies, ord=1) / num_orbitals

        if diff < tol:
            print(f"Converged in {itr} iterations")  # noqa: T201
            break

        old_energies = energies

    print(coeffs)  # noqa: T201

    return energies, coeffs


if __name__ == "__main__":
    run(-1)
