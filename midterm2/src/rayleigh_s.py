import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from exact_energy import get_all_energies_below, get_states, get_energy_from_states

g = sp.Symbol("g")

holes = (1, 2)
particles = (3, 4)


def zeroth_RS():
    return 2


def first_RS():
    return -g


def second_RS():
    total = 0
    for i in holes:
        for a in particles:
            total += g**2 / (i - a)

    total /= 8
    return total


def third_RS():
    def four():
        total = 0
        for i in holes:
            for a in particles:
                for c in particles:
                    total += g**3 / ((i - a) * (i - c))

        total /= -32
        return total

    def five():
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    total += g**3 / ((i - a) * (k - a))

        total /= -32
        return total

    def eight():
        total = 0
        for i in holes:
            for a in particles:
                total += g**3 / ((i - a) ** 2)

        total /= 16
        return total

    total = four() + five() + eight()
    return total


def fourth_RS():
    def five():
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (k - a) * (k - c))

        total /= 128
        return total

    def six():
        total = 0
        for i in holes:
            for k in holes:
                for a in particles:
                    for c in particles:
                        total += g**4 / ((i - a) * (i - c) * (k - c))

        total /= 128
        return total

    def fourteen():
        total = 0
        for i in holes:
            for a in particles:
                for c in particles:
                    for e in particles:
                        total += g**4 / ((i - a) * (i - c) * (i - e))

        total /= 128
        return total

    def fifteen():
        total = 0
        for i in holes:
            for k in holes:
                for m in holes:
                    for a in particles:
                        total += g**4 / ((i - a) * (k - a) * (m - a))

        total /= 128
        return total

    def thirtysix():
        total = 0
        for i in holes:
            for k in holes:
                # if i == k:
                #     continue
                for a in particles:
                    for c in particles:
                        # if a == c:
                        #     continue
                        total += g**4 / ((i - a) * (i + k - a - c) * (i - c))

        total /= 128
        return total

    def thirtyseven():
        total = 0
        for i in holes:
            for k in holes:
                # if i == k:
                #     continue
                for a in particles:
                    for c in particles:
                        # if a == c:
                        #     continue
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


def compute_RS(order: int = 2):
    funcs = (
        zeroth_RS,
        first_RS,
        second_RS,
        third_RS,
        fourth_RS,
    )[: order + 1]
    total = 0
    for func in funcs:
        print(func.__name__)
        total += func()

    return total


def RS4_vs_FCI():
    g_values = np.linspace(-1, 1, 100)
    RS4 = compute_RS(4)
    RS4 = sp.lambdify(g, RS4, "numpy")(g_values)
    FCI_energies = np.zeros(100)
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
    g_values = np.linspace(-1, 1, 100)
    for order in (2, 3, 4):
        energy_func = sp.lambdify(g, compute_RS(order), "numpy")
        energy_values = energy_func(g_values)
        plt.plot(g_values, energy_values, label=f"Order {order}")

    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    # plt.show()

    RS4_vs_FCI()
