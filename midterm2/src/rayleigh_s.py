from typing import Literal  # noqa: CPY001, D100, INP001

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from exact_energy import get_all_energies_below

g = sp.Symbol("g")

default_g = np.linspace(-1, 1, 101)

holes = (1, 2)
particles = (3, 4)


def zeroth_RS() -> Literal[2]:  # noqa: N802
    """Return the zeroth order Rayleigh-Schrödinger perturbation term.

    Returns:
        Literal[2]: The zeroth order perturbation term.

    """
    return 2


def first_RS() -> sp.Expr:  # noqa: N802
    """Return the first order Rayleigh-Schrödinger perturbation term.

    Returns:
        sp.Expr: The first order perturbation term.

    """
    return -g


def second_RS() -> sp.Expr:  # noqa: N802
    """Return the second order Rayleigh-Schrödinger perturbation term.

    Returns:
        sp.Expr: The second order perturbation term.

    """
    total = 0
    for i in holes:
        for a in particles:
            total += g**2 / (i - a)

    total /= 8
    return total


def third_RS() -> sp.Expr:  # noqa: C901, N802
    """Return the third order Rayleigh-Schrödinger perturbation term.

    Each diagram is implemented as a separate function.

    Returns:
        sp.Expr: The third order perturbation term.

    """

    def four() -> sp.Expr:
        total = 0
        for i in holes:
            for a in particles:
                for c in particles:
                    total += g**3 / ((i - a) * (i - c))

        total /= -32
        return total

    def five() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    total += g**3 / ((i - a) * (k - a))

        total /= -32
        return total

    def eight() -> sp.Expr:
        total = 0
        for i in holes:
            for a in particles:
                total += g**3 / ((i - a) ** 2)

        total /= 16
        return total

    return four() + five() + eight()


def fourth_RS() -> sp.Expr:  # noqa: C901, N802
    """Return the fourth order Rayleigh-Schrödinger perturbation term.

    Each diagram is implemented as a separate function.

    Returns:
        sp.Expr: The fourth order perturbation term.

    """

    def five() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (k - a) * (k - c))

        total /= 128
        return total

    def six() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (i - c) * (k - c))

        total /= 128
        return total

    def fourteen() -> sp.Expr:
        total = 0
        for i in holes:
            for a in particles:
                for c in particles:
                    for e in particles:
                        total += g**4 / ((i - a) * (i - c) * (i - e))

        total /= 128
        return total

    def fifteen() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for m in holes:
                    for a in particles:
                        total += g**4 / ((i - a) * (k - a) * (m - a))

        total /= 128
        return total

    def thirtysix() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (i + k - a - c) * (i - c))

        total /= 128
        return total

    def thirtyseven() -> sp.Expr:
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (i + k - a - c) * (k - a))

        total /= 128
        return total

    diagrams = (
        five,
        six,
        fourteen,
        fifteen,
        thirtysix,
        thirtyseven,
    )
    total = 0
    for diagram in diagrams:
        total += diagram()

    return total


def compute_RS_func(order: int = 2) -> sp.Expr:  # noqa: N802
    """Compute the Rayleigh-Schrödinger perturbation series up to a given order.

    Args:
        order (int): The order of the perturbation series to compute.

    Returns:
        sp.Expr: The symbolic expression of the perturbation series.

    """
    funcs = (
        zeroth_RS,
        first_RS,
        second_RS,
        third_RS,
        fourth_RS,
    )[: order + 1]
    total = 0
    for func in funcs:
        total += func()

    return total


def compute_RS(order: int = 2, g_values: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
    """Compute the Rayleigh-Schrödinger perturbation series values for a range of g.

    Args:
        order (int): The order of the perturbation series to compute.

    Returns:
        np.ndarray: The computed values of the perturbation series for a range of g.

    """
    if g_values is None:
        g_values = default_g
    func = compute_RS_func(order)
    return sp.lambdify(g, func, "numpy")(g_values)


def RS_coeffs(g_values: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
    """Compute the Rayleigh-Schrödinger perturbation series coefficients.

    This refers to the coefficients from second-order.

    Returns:
        np.ndarray: The coefficients of the perturbation series.

    """
    if g_values is None:
        g_values = default_g
    coeffs = np.zeros((len(g_values), 4))
    counter = 0
    for i in holes:
        for a in particles:
            coeffs[:, counter] += g_values / (i - a)
            counter += 1

    coeffs /= -4

    return coeffs


def RS4_vs_FCI() -> None:  # noqa: N802
    """Plot the comparison between RS4 and FCI energies and their difference."""
    g_values = np.linspace(-1, 1, 100)
    RS4 = compute_RS_func(4)  # noqa: N806
    RS4 = sp.lambdify(g, RS4, "numpy")(g_values)  # noqa: N806
    FCI_energies = np.zeros(100)  # noqa: N806
    for i, g_value in enumerate(g_values):
        FCI_energies[i] = get_all_energies_below(g_value, 4)[0]

    plt.plot(g_values, RS4, label="RS4")
    plt.plot(g_values, FCI_energies, label="FCI")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

    plt.plot(g_values, FCI_energies - RS4)
    plt.xlabel("$g$")
    plt.ylabel("FCI - RS4")
    plt.title("Difference between FCI and RS4")
    plt.show()


if __name__ == "__main__":
    print(compute_RS_func(4).simplify())  # noqa: T201
