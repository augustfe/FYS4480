import numpy as np

num_orbitals = 4
spin_degen = 2


def foo(rho: np.ndarray) -> np.ndarray:
    (
        np.flip(
            rho.reshape((num_orbitals, spin_degen, num_orbitals, spin_degen))
            .swapaxes(1, 2)
            .reshape(num_orbitals, num_orbitals, spin_degen * spin_degen),
            axis=-1,
        )
        .reshape((num_orbitals, num_orbitals, spin_degen, spin_degen))
        .swapaxes(1, 2)
        .reshape((num_orbitals * spin_degen, num_orbitals * spin_degen))
    )
    return (
        rho.reshape((num_orbitals, spin_degen, num_orbitals, spin_degen))
        .swapaxes(1, 2)
        .swapaxes(-2, -1)
        .swapaxes(1, 2)
        .reshape((num_orbitals * spin_degen, num_orbitals * spin_degen))
    )


def swap_spins(rho: np.ndarray) -> np.ndarray:
    new_rho = rho.reshape(
        (num_orbitals, spin_degen, num_orbitals, spin_degen)
    ).swapaxes(1, 2)
    flipped = np.copy(new_rho)
    flipped[..., 0, 0], flipped[..., 1, 1] = new_rho[..., 1, 1], new_rho[..., 0, 0]
    flipped[..., 0, 1], flipped[..., 1, 0] = new_rho[..., 1, 0], new_rho[..., 0, 1]
    new_rho = flipped

    new_rho = new_rho.swapaxes(1, 2).reshape(
        (num_orbitals * spin_degen, num_orbitals * spin_degen)
    )
    return new_rho


def get_rho(coeffs: np.ndarray) -> np.ndarray:
    return coeffs.conj().T @ coeffs


def run(g: float, max_iter: int = 100, tol: float = 1e-14):

    particles = np.arange(1, num_orbitals + 1).repeat(spin_degen)
    h = np.diag(particles - 1)

    coeffs = np.eye(num_orbitals * spin_degen, dtype=complex)

    def iteration(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho = get_rho(coeffs)
        fock_matrix = h - 0.5 * g * swap_spins(rho)
        print(fock_matrix)
        energies, new_coeffs = np.linalg.eigh(fock_matrix)
        return energies, new_coeffs

    old_energies = np.zeros(num_orbitals * spin_degen)
    for iter in range(max_iter):
        energies, coeffs = iteration(coeffs)

        diff = np.linalg.norm(energies - old_energies, ord=1) / num_orbitals

        if diff < tol:
            print(f"Converged in {iter} iterations")
            break

        old_energies = energies

    print(coeffs)

    return energies, coeffs


def get_grounstate_energy(g: float, coeffs: np.ndarray) -> float:
    rho = get_rho(coeffs)
    fock_matrix = h


if __name__ == "__main__":
    run(-1)
