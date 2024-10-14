import numpy as np
import sympy as sp
import re

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Computer Modern"
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


matplotlib.rcParams["font.size"] = SMALL_SIZE  # controls default text sizes
matplotlib.rcParams["axes.titlesize"] = BIGGER_SIZE  # fontsize of the axes title
matplotlib.rcParams["axes.labelsize"] = MEDIUM_SIZE  # fontsize of the x and y labels
matplotlib.rcParams["xtick.labelsize"] = SMALL_SIZE  # fontsize of the tick labels
matplotlib.rcParams["ytick.labelsize"] = SMALL_SIZE  # fontsize of the tick labels
matplotlib.rcParams["legend.fontsize"] = SMALL_SIZE  # legend fontsize
matplotlib.rcParams["figure.titlesize"] = BIGGER_SIZE  # fontsize of the figure title


Z = sp.symbols("Z")


def convert_sqrt(expression: str) -> str:
    """Convert from Mathematica's Sqrt[...] to Sympy's sqrt(...)."""
    pattern = r"Sqrt\[(.*?)\]"
    replacement = r"sqrt(\1)"
    return re.sub(pattern, replacement, expression)


def read_line(s: str) -> tuple[tuple[int, int, int, int], sp.Expr]:
    """Read a line from the matrix_elements

    Args:
        s (str): A line from the matrix_elements.txt file

    Returns:
        tuple[int]: The indicies of the matrix element
        sp.Expr: The value of the matrix element
    """
    left, right = s.split(" = ")

    (a, b), (c, d) = left[1:-1].split("|V|")
    indicies = *map(lambda x: int(x) - 1, (a, b, c, d)),  # fmt: skip

    val = convert_sqrt(right.strip())
    value = sp.parse_expr(val)

    return indicies, value


def read_elements(n_max: int = 3) -> np.ndarray:
    """Read the matrix elements from the file

    Args:
        n_max (int, optional): The size of the matrix. Defaults to 3.

    Returns:
        np.ndarray: The matrix elements
    """
    path = Path(__file__).parent / "matrix_elements.txt"

    values = np.zeros((n_max, n_max, n_max, n_max), dtype=object)

    with open(path) as infile:
        for line in infile:
            indicies, value = read_line(line)
            values[indicies] = value

    return values


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


def plot_energy(save_path: Path) -> None:
    save_path.mkdir(exist_ok=True)

    energy = compute_groundstate_energy()
    print(f"Ground state energy: {energy}")
    energy_func = sp.lambdify(Z, energy, "numpy")

    Z_vals = np.arange(1, 11)
    energies = energy_func(Z_vals)
    plt.figure(figsize=(4, 3))

    # Plot the energy at the discrete points, with red crosses and green line
    plt.plot(Z_vals, energies, "k--", lw=2)
    plt.plot(Z_vals, energies, "rx", ms=7)
    plt.xlabel("$Z$")
    plt.ylabel("Potential energy")
    plt.title("Ground state energy")
    plt.grid()

    # Add the atomic numbers next to the points
    periodictable = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]

    for i, txt in enumerate(periodictable):
        plt.annotate(
            txt,
            (Z_vals[i], energies[i]),
            textcoords="offset points",
            xytext=(5, 8),
            ha="center",
        )

    bottom, top = min(energies), max(energies)
    scaling = 0.15 * (top - bottom)

    plt.ylim(bottom - scaling, top + scaling)
    plt.xticks(Z_vals, [f"${Z}$" for Z in Z_vals])
    plt.xlim(0, 11)
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.tight_layout()

    plt.savefig(save_path / "energy_plot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    save_path = Path(__file__).parent / "figs"
    plot_energy(save_path)
    # plt.show()

    [
        [1, 0, 0],
        [1, 0, 0],
    ]
    [2, 0, 0]
