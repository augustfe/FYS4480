import numpy as np
import sympy as sp


from pathlib import Path
from contextlib import redirect_stdout

from utils import read_elements, to_eV, Z


class HartreeFock:
    def __init__(self, F: int, Z_val: int) -> None:
        self.F = F
        self.Z = Z_val
        self.num_orbitals = 6

        self.levels = np.arange(1, 4).repeat(2)
        self.spins = np.array([0, 1] * 3)
        self.electrons = np.vstack([self.levels, self.spins]).T

        self.h = np.diag(-(Z_val**2) / (2 * self.levels**2))

        def insert_Z(x: sp.Expr) -> sp.Float:
            return x.subs(Z, Z_val).evalf()

        values = read_elements()
        self.values = np.vectorize(insert_Z)(values).astype(np.float64)
        self.coulomb_integrals = self.setup_coulomb()

    def setup_coulomb(self) -> np.ndarray:
        coulomb_integrals = np.zeros(
            (self.num_orbitals, self.num_orbitals, self.num_orbitals, self.num_orbitals)
        )

        for i, alpha in enumerate(self.electrons):
            for j, beta in enumerate(self.electrons):
                for k, gamma in enumerate(self.electrons):
                    for l, delta in enumerate(self.electrons):  # noqa
                        coulomb_integrals[i, j, k, l] = self.antisymmetrized(
                            alpha, beta, gamma, delta
                        )

        return coulomb_integrals

    def antisymmetrized(
        self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray
    ) -> float:
        direct = self.V(alpha, beta, gamma, delta)
        exchange = self.V(alpha, beta, delta, gamma)
        return direct - exchange

    def V(
        self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray
    ) -> float:
        spin_ok = (alpha[1] == gamma[1]) and (beta[1] == delta[1])
        electron_ok = (alpha != beta).any() and (gamma != delta).any()
        if (not spin_ok) or (not electron_ok):
            return 0

        indices = np.array([alpha[0], beta[0], gamma[0], delta[0]]) - 1

        return self.values[tuple(indices)]

    def groundstate_energy(self, coeffs: np.ndarray) -> float:
        C = coeffs[: self.F, :]

        h0 = np.einsum("ab, ia, ib ->", self.h, C, C)
        h1 = np.einsum("abcd, ia, jb, ic, jd ->", self.coulomb_integrals, C, C, C, C)

        return h0 + 0.5 * h1

    def setup_density_matrix(self, coefficients: np.ndarray) -> np.ndarray:
        C = coefficients[: self.F, :]
        density_matrix = np.einsum("ia, ib -> ab", C, C)

        return density_matrix

    def iteration(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        density_matrix = self.setup_density_matrix(coefficients)
        fock_matrix = self.h + self.coulomb(density_matrix)

        energy, coefficients = self.diagonalize(fock_matrix)
        return energy, coefficients

    def diagonalize(self, fock_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energy, coefficients = np.linalg.eigh(fock_matrix)
        return energy, coefficients.T

    def coulomb(self, density_matrix: np.ndarray) -> np.ndarray:
        coulomb_matrix = np.einsum(
            "abcd, bd -> ac", self.coulomb_integrals, density_matrix
        )

        return coulomb_matrix

    def run(self, max_iter=100, tol=1e-14):
        coeffs = np.eye(self.num_orbitals)
        old_energies = np.zeros(self.num_orbitals)

        for iteration in range(max_iter):
            density_matrix = self.setup_density_matrix(coeffs)
            fock_matrix = self.h + self.coulomb(density_matrix)
            energies, coeffs = self.diagonalize(fock_matrix)

            difference = (
                np.linalg.norm(energies - old_energies, ord=1) / self.num_orbitals
            )
            if difference < tol:
                print(f"Converged in {iteration} iterations")
                break

            old_energies = energies
        return energies, coeffs

    def groundstate_loop(self, coeffs: np.ndarray) -> np.ndarray:
        h0 = 0
        for i in range(self.F):
            for alpha in range(self.num_orbitals):
                for beta in range(self.num_orbitals):
                    h0 += self.h[alpha, beta] * coeffs[i, alpha] * coeffs[i, beta]

        h1 = 0
        for i in range(self.F):
            for j in range(self.F):
                for alpha in range(self.num_orbitals):
                    for beta in range(self.num_orbitals):
                        for gamma in range(self.num_orbitals):
                            for delta in range(self.num_orbitals):
                                h1 += (
                                    self.coulomb_integrals[alpha, beta, gamma, delta]
                                    * coeffs[i, alpha]
                                    * coeffs[j, beta]
                                    * coeffs[i, gamma]
                                    * coeffs[j, delta]
                                )

        return h0 + 0.5 * h1


def first_iteration() -> None:
    print("Results after one iteration:")
    cases = [(2, "Helium"), (4, "Beryllium")]
    for Z_val, name in cases:
        print(f"{name}:")
        hf = HartreeFock(Z_val, Z_val)
        coefficients = np.eye(6)

        energies, coefficients = hf.iteration(coefficients)
        groundstate_energy = hf.groundstate_energy(coefficients)

        assert np.isclose(groundstate_energy, hf.groundstate_loop(coefficients))

        print(f"New single-particle energies: {energies}")
        print(f"New ground state energy: {groundstate_energy}")
        print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")


def iter_until_convergence(tol: float = 1e-14) -> None:
    print(f"Results after full convergence: (with tolerance {tol})")
    cases = [(2, "Helium"), (4, "Beryllium")]
    for Z_val, name in cases:
        print(f"{name}:")
        hf = HartreeFock(Z_val, Z_val)
        energies, coefficients = hf.run()
        groundstate_energy = hf.groundstate_energy(coefficients)

        print(f"Final single-particle energies: {energies}")
        print(f"Final ground state energy: {groundstate_energy}")
        print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")


if __name__ == "__main__":
    write_path = Path(__file__).parent / "hartree_fock.txt"

    with open(write_path, "w") as outfile:
        with redirect_stdout(outfile):
            first_iteration()
            iter_until_convergence()
