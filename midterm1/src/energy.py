import numpy as np
import sympy as sp

from itertools import permutations

from utils import read_elements, Z


class Energy:
    def __init__(self, groundstate: np.ndarray, F: int = 1) -> None:
        """Initialize the Energy class

        Args:
            groundstate (np.ndarray): Array with the groundstates, and spin values
            F (int, optional): Fermi level. Defaults to 1.
        """
        self.F = F
        self.groundstate = groundstate
        self.values = read_elements()

        self.E0_ref = self.groundstate_onebody()
        self.EI_ref = self.groundstate_twobody()
        self.E_ref = self.E0_ref + self.EI_ref

    @property
    def groundstate(self) -> np.ndarray:
        return self._groundstate

    @groundstate.setter
    def groundstate(self, groundstate: np.ndarray) -> None:
        if not isinstance(groundstate, np.ndarray):
            raise TypeError("Groundstate must be a numpy array")

        if not groundstate.shape == (2, 3):
            raise ValueError("The groundstate must be of shape (2, 3)")

        row_sums = np.sum(groundstate, axis=1)
        M_S = row_sums[1] - row_sums[0]
        if not M_S == 0:
            raise ValueError("The groundstate must have total spin 0")

        if np.any(groundstate[:, self.F:]):  # fmt: skip
            raise ValueError(
                "The groundstate must have all particles below the Fermi level"
            )

        if np.any((groundstate < 0) | (groundstate > 1)):
            raise ValueError("The groundstate must be binary")

        self._groundstate = groundstate.astype(bool)

    def onebody_energy(self, alpha: int) -> sp.Expr:
        """Compute one-body energy for one electron

        Args:
            alpha (int): The energy level of the electron

        Returns:
            sp.Expr: The one-body energy
        """
        return -(Z**2) / (2 * alpha**2)

    def groundstate_onebody(self) -> sp.Expr:
        """Compute the one-body groundstate energy

        Returns:
            sp.Expr: The one-body energy
        """
        _, c_idx = np.nonzero(self.groundstate)
        energy_level = c_idx + 1

        return np.sum(self.onebody_energy(energy_level))

    def twobody_energy(self, alpha: int, beta: int) -> sp.Expr:
        """Compute the two-body energy for two electrons

        Args:
            alpha (int): The energy level of the first electron
            beta (int): The energy level of the second electron

        Returns:
            sp.Expr: The two-body energy
        """
        return self.values[alpha, beta, alpha, beta] / 2

    def groundstate_twobody(self) -> sp.Expr:
        """Compute the two-body groundstate energy

        Returns:
            sp.Expr: The two-body energy
        """
        _, c_idx = np.nonzero(self.groundstate)

        energy = 0
        for alpha, beta in permutations(c_idx, r=2):
            energy += self.twobody_energy(alpha, beta)

        return energy


if __name__ == "__main__":
    groundstate = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    energy = Energy(groundstate)
    print(energy.E0_ref)
    print(energy.EI_ref)
    print(energy.E0_ref + energy.EI_ref)
