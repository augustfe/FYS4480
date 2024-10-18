import numpy as np
import sympy as sp

from utils import read_elements, to_eV, Z
from typing import NamedTuple


class Electron(NamedTuple):
    spin: int
    level: int


class SetupMatrix:
    def __init__(self, F: int) -> None:
        self.F = F
        self.groundstate = np.zeros((2, 3))
        self.groundstate[:, :F] = 1
        self.values = read_elements()

        self.ref_energy = self.reference_energy()

    def reference_energy(self) -> sp.Expr:
        onebody = 0
        two_body = 0
        hole_arrs = np.argwhere(self.groundstate[:, :self.F] == 1)  # fmt: skip
        holes = [Electron(*hole) for hole in hole_arrs]

        for hole in holes:
            onebody += self.h0(hole, hole)

        for hole in holes:
            for other_hole in holes:
                two_body += self.antisymmetrized(hole, other_hole, hole, other_hole) / 2

        return onebody + two_body

    def get_hole_particle(self, state: np.ndarray) -> tuple[Electron, Electron]:
        hole = np.argwhere(state[:, :self.F] == 0)[0]  # fmt: skip
        particle = np.argwhere(state[:, self.F:] == 1)[0]  # fmt: skip
        particle = particle + np.array([0, self.F])

        return Electron(*hole), Electron(*particle)

    def energy_from_state(
        self, bra_state: np.ndarray, ket_state: np.ndarray
    ) -> sp.Expr:

        if np.all(bra_state == self.groundstate) and np.all(
            ket_state == self.groundstate
        ):
            return self.ref_energy

        if np.all(bra_state == self.groundstate) or np.all(
            ket_state == self.groundstate
        ):
            acting = bra_state if np.all(ket_state == self.groundstate) else ket_state
            i, a = self.get_hole_particle(acting)

            return self.f(i, a)

        i, a = self.get_hole_particle(bra_state)
        j, b = self.get_hole_particle(ket_state)

        energy = self.ref_energy * self.delta(i, j) * self.delta(a, b)
        energy += self.f(a, b) * self.delta(i, j)
        energy -= self.f(i, j) * self.delta(a, b)
        energy += self.antisymmetrized(a, j, i, b)

        return energy

    def delta(self, alpha: int, beta: int) -> int:
        if alpha == beta:
            return 1
        return 0

    def h0(self, p: Electron, q: Electron) -> sp.Expr:
        # if p != q:
        #     return 0
        n = p.level
        return -(Z**2) / (2 * (n + 1) ** 2) * self.delta(p, q)

    def v(
        self,
        p: Electron,
        q: Electron,
        r: Electron,
        s: Electron,
    ) -> sp.Expr:
        spins = [p.spin, q.spin, r.spin, s.spin]
        levels = [p.level, q.level, r.level, s.level]

        if not self.spin_ok(*spins):
            return 0

        if p == q or r == s:
            return 0

        return self.values[levels[0], levels[1], levels[2], levels[3]]

    def antisymmetrized(
        self,
        p: Electron,
        q: Electron,
        r: Electron,
        s: Electron,
    ) -> sp.Expr:
        return self.v(p, q, r, s) - self.v(p, q, s, r)

    def f(self, p: Electron, q: Electron) -> sp.Expr:
        energy = self.h0(p, q)

        for spin in range(2):
            for k in range(self.F):
                k_s = Electron(spin, k)
                energy += self.antisymmetrized(p, k_s, q, k_s)

        return energy

    def spin_ok(self, a: int, b: int, c: int, d: int) -> bool:
        if a == c and b == d:
            return True
        return False

    def annihilate_and_create(self, sigma: int, i: int, a: int) -> np.ndarray:
        new_state = np.copy(self.groundstate)
        new_state[sigma, i] -= 1
        new_state[sigma, a] += 1

        assert new_state[sigma, i] == 0
        assert new_state[sigma, a] == 1

        return new_state

    def get_total_states(self) -> list[np.ndarray]:
        total_states = [self.groundstate]

        for i in range(self.F):
            for a in range(self.F, 3):
                for sigma in range(2):
                    new_state = self.annihilate_and_create(sigma, i, a)
                    total_states.append(new_state)

        return total_states

    def get_hamiltonian(self) -> sp.Matrix:
        total_states = self.get_total_states()
        Hamiltonian = sp.zeros(len(total_states), len(total_states))

        for i, bra_state in enumerate(total_states):
            for j, ket_state in enumerate(total_states):
                Hamiltonian[i, j] = self.energy_from_state(bra_state, ket_state)

        return Hamiltonian

    def get_energy(self, z_val: int) -> float:
        Hamiltonian = self.get_hamiltonian()
        eigenvals = Hamiltonian.subs(Z, z_val).eigenvals()
        eigs = [sp.re(key.evalf()) for key in eigenvals.keys()]

        return min(eigs)


def evaluate(F: int, z_val: int) -> None:
    he_setup = SetupMatrix(F=F)
    ground_energy_expr = he_setup.ref_energy
    print(ground_energy_expr)
    print(ground_energy_expr.evalf())
    ground_energy_atomic = ground_energy_expr.subs(Z, z_val)
    print(ground_energy_atomic.evalf())
    ground_energy = to_eV(ground_energy_atomic.evalf())
    print(ground_energy)

    print()

    energy = he_setup.get_energy(z_val)
    print(energy)
    print(to_eV(energy))


if __name__ == "__main__":
    # evaluate(F=1, z_val=2)
    print()
    evaluate(F=2, z_val=4)
