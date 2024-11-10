import numpy as np
from typing import NamedTuple
import sympy as sp
from sympy.printing import pprint


class Result(NamedTuple):
    state: np.ndarray
    value: int


class State:
    def __init__(self) -> None:
        self.F = 2
        self.P_max = 4
        self.groundstate = np.zeros((4, 2))
        self.groundstate[:self.F] = 1  # fmt: skip

        self.all_states = self.generate_initial_states()
        self.g = sp.Symbol("g", real=True)

    def generate_initial_states(self) -> list[np.ndarray]:
        states = [self.groundstate]

        # 2p2h states:
        for i in range(self.F - 1, -1, -1):
            for a in range(self.F, self.P_max):
                state = self.groundstate.copy()
                state = self.pair_annihilate(state, i)
                state = self.pair_create(state, a)

                states.append(state)

        # 4p4h states:
        for i in range(self.F - 1, -1, -1):
            for j in range(i + 1, self.F):
                for a in range(self.F, self.P_max):
                    for b in range(a + 1, self.P_max):
                        state = self.groundstate.copy()
                        state = self.pair_annihilate(state, i, j)
                        state = self.pair_create(state, a, b)

                        states.append(state)

        return states

    def pair_annihilate(self, state: np.ndarray, *ps: int) -> np.ndarray:
        for p in ps:
            state = self.annihilate_with_spin(state, p, 0)
            state = self.annihilate_with_spin(state, p, 1)

        return state

    def annihilate_with_spin(self, state: np.ndarray, p: int, sigma: int) -> np.ndarray:
        state[p, sigma] -= 1

        return state

    def pair_create(self, state: np.ndarray, *ps: int) -> np.ndarray:
        for p in ps:
            state = self.create_with_spin(state, p, 0)
            state = self.create_with_spin(state, p, 1)

        return state

    def create_with_spin(self, state: np.ndarray, p: int, sigma: int) -> np.ndarray:
        state[p, sigma] += 1

        return state

    def h0(self, state: np.ndarray, p: int, sigma: int) -> Result:
        # Adjust for zero indexed
        coeff = p
        state = self.annihilate_with_spin(state, p, sigma)
        if np.any(state < 0):
            coeff = 0

        state = self.create_with_spin(state, p, sigma)
        if np.any(state > 1):
            coeff = 0

        return Result(state, coeff)

    def H0(self, bra_state: np.ndarray, ket_state: np.ndarray) -> float:
        value = 0.0
        for p in range(self.P_max):
            for sigma in range(2):
                result = self.h0(ket_state.copy(), p, sigma)
                if result.value == 0:
                    continue

                if np.array_equal(bra_state, result.state):
                    value += result.value

        return value

    def v(self, state: np.ndarray, p: int, q: int) -> Result:
        coeff = -self.g / 2

        state = self.pair_annihilate(state, q)
        if np.any(state < 0):
            coeff = 0

        state = self.pair_create(state, p)
        if np.any(state > 1):
            coeff = 0

        return Result(state, coeff)

    def V(self, bra_state: np.ndarray, ket_state: np.ndarray) -> float:
        value = 0.0
        for p in range(self.P_max):
            for q in range(self.P_max):
                result = self.v(ket_state.copy(), p, q)
                if result.value == 0:
                    continue

                if np.array_equal(bra_state, result.state):
                    value += result.value

        return value

    def setup_hamiltonian(self) -> sp.Matrix:
        total_states = self.all_states
        Hamiltonian = sp.zeros(len(total_states), len(total_states))

        for i, bra_state in enumerate(total_states):
            for j, ket_state in enumerate(total_states):
                bra, ket = bra_state.copy(), ket_state.copy()
                Hamiltonian[i, j] = self.H0(bra.copy(), ket.copy()) + self.V(
                    bra.copy(), ket.copy()
                )

        return Hamiltonian

    def diagonalize(self) -> sp.Matrix:
        Hamiltonian = self.setup_hamiltonian()
        return Hamiltonian.eigenvals()


if __name__ == "__main__":
    state = State()
    hamiltonian = state.setup_hamiltonian()
    pprint(hamiltonian)

    print()
    # print(state.diagonalize())

    # eigenvals = state.diagonalize()
    # eigenvals = hamiltonian.eigenvals(rational=True, simplify=True)
    # for key, value in eigenvals.items():
    #     pprint(key)
    #     pprint(value)
    #     print()

    g_range = np.linspace(-1, 1, 100)
    eigenvals = np.zeros((len(g_range), len(state.all_states)))
    for i, g in enumerate(g_range):
        ham = hamiltonian.subs(state.g, g)
        ham_np = np.array(ham).astype(np.float64)
        eigs = np.linalg.eigvalsh(ham_np)
        # lowest_energy = np.min(eigs)
        # eigs = hamiltonian.subs(state.g, g).eigenvals(rational=True, simplify=True)
        # eigs = [sp.re(e) for e in eigs.keys()]
        # lowest_energy = min(eigs)
        eigenvals[i] = eigs

    import matplotlib.pyplot as plt

    plt.style.use("latex.mplstyle")

    fig = plt.figure(figsize=(4, 3))
    labels = [rf"$\varepsilon_{i}$" for i in range(len(state.all_states))]
    linestyles = ["-", "--", "-.", ":"]

    multiplicities = {}
    for val in eigenvals[0]:
        for seen in multiplicities:
            if np.isclose(val, seen):
                multiplicities[seen] += 1
                break
        else:
            multiplicities[val] = 1

    shown = set()
    for i, label in enumerate(labels):
        eigenvals_i = eigenvals[:, i]
        if eigenvals_i[0] in shown:
            continue
        oops = False
        for j in range(0, len(eigenvals)):
            if np.isclose(eigenvals[j, 0], eigenvals_i[0]):
                oops = True
                break
        if oops:
            continue

        shown.add(eigenvals_i[0])
        start = 1 if multiplicities[eigenvals_i[0]] > 1 else 0
        for j in range(start, multiplicities[eigenvals_i[0]] + start):
            plt.plot(g_range, eigenvals_i, label=label, linestyle=linestyles[j])

    print(multiplicities)

    # plt.plot(g_range, eigenvals)
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.title("Ground state energy as a function of $g$")
    plt.grid()
    # plt.legend(labels, loc="best")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("../figures/ground_state_energy.pdf")
    plt.show()
