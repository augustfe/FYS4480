import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from b import get_all_energies_below, get_energy_from_states, get_states

dir_path = Path(__file__).parents[1]
plt.style.use(dir_path / "src" / "latex.mplstyle")


def b_plot_eigenvalues(save: bool = True) -> None:
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
    # plt.tight_layout()
    if save:
        plt.savefig(
            dir_path / "figures" / "b_eigenvalues_energy.pdf", bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.show()


def b_plot_groundstate(save: bool = True) -> None:
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
    # plt.tight_layout()
    if save:
        plt.savefig(
            dir_path / "figures" / "b_ground_state_energy.pdf", bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.show()


def c_plot_groundstate(save: bool = True) -> None:
    max_level = 4
    # Remove Phi_5
    states = get_states(max_level)[:-1]
    g_values = np.linspace(-1, 1, 100)
    energies = np.zeros((len(g_values), 5))
    for i, g in enumerate(g_values):
        energies[i] = get_energy_from_states(g, states)

    plt.figure(figsize=(4, 3))

    plt.plot(g_values, energies[:, 0], label="Ground state")

    plt.xlabel(r"$g$")
    plt.ylabel(r"Energy")
    plt.title(r"Groundstate energy as a function of $g$")

    plt.legend()
    # plt.tight_layout()
    if save:
        plt.savefig(
            dir_path / "figures" / "c_ground_state_energy.pdf", bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.show()


def c_plot_diff(save: bool = True) -> None:
    max_level = 4
    b_states = get_states(max_level)
    c_states = b_states[:-1]

    g_values = np.linspace(-1, 1, 100)
    energies = np.zeros(len(g_values))

    for i, g in enumerate(g_values):
        b_energy = get_energy_from_states(g, b_states)
        c_energy = get_energy_from_states(g, c_states)
        energies[i] = b_energy[0] - c_energy[0]

    plt.figure(figsize=(4, 3))

    plt.plot(g_values, energies)

    plt.xlabel(r"$g$")
    plt.ylabel(r"$\Delta$ Energy")
    plt.title(r"Difference in groundstate energy as a function of $g$")

    # plt.tight_layout()
    if save:
        plt.savefig(
            dir_path / "figures" / "c_diff_ground_state_energy.pdf", bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    b_plot_eigenvalues()
    b_plot_groundstate()
    c_plot_groundstate()
    c_plot_diff()
