import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from b import get_all_energies_below

dir_path = Path(__file__).parents[1]
plt.style.use(dir_path / "src" / "latex.mplstyle")


def plot_eigenvalues(save: bool = True) -> None:
    g_values = np.linspace(-1, 1, 100)
    max_level = 4
    energies = np.zeros((len(g_values), 6))
    for i, g in enumerate(g_values):
        energies[i] = get_all_energies_below(g, max_level)

    plt.figure(figsize=(4, 3))
    labels = [rf"$\varepsilon_{i}$" for i in range(6)]
    for i in range(6):
        linestyle = "-" if i != 3 else "--"
        plt.plot(g_values, energies[:, i], label=labels[i], linestyle=linestyle)

    plt.xlabel(r"$g$")
    plt.ylabel(r"Energy")
    plt.title(r"Eigenvalues as a function of $g$")

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(dir_path / "figures" / "b_eigenvalues_energy.pdf")
        plt.clf()
    else:
        plt.show()


def plot_groundstate(save: bool = True) -> None:
    g_values = np.linspace(-1, 1, 100)
    max_level = 4
    energies = np.zeros((len(g_values), 6))
    for i, g in enumerate(g_values):
        energies[i] = get_all_energies_below(g, max_level)

    plt.figure(figsize=(4, 3))

    plt.plot(g_values, energies[:, 0], label="Ground state")

    plt.xlabel(r"$g$")
    plt.ylabel(r"Energy")
    plt.title(r"Groundstate energy as a function of $g$")

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(dir_path / "figures" / "b_ground_state_energy.pdf")
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    plot_eigenvalues()
    plot_groundstate()
