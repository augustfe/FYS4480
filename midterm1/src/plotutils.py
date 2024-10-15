import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from pathlib import Path

from energy import compute_groundstate_energy
from utils import Z

dir_path = Path(__file__).parents[1]
save_path = dir_path / "figs"

style_path = dir_path / "src" / "latex.mplstyle"
plt.style.use(style_path)


def plot_energy(save_path: Path = save_path) -> None:
    """Plot the ground state energy as a function of Z

    Args:
        save_path (Path): The path to save the plot
    """
    save_path.mkdir(exist_ok=True)

    energy_expr = compute_groundstate_energy()
    print(f"Ground state energy: {energy_expr}")
    energy_func = sp.lambdify(Z, energy_expr, "numpy")

    Z_vals = np.arange(1, 11)
    energies = energy_func(Z_vals)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca()

    # Plot the energy at the discrete points, with red crosses and green line
    ax.plot(Z_vals, energies, "k--", lw=2)
    ax.plot(Z_vals, energies, "rx", ms=7)
    ax.set_xlabel("$Z$")
    ax.set_ylabel("Potential energy", labelpad=-3)
    ax.set_title("Ground state energy")
    ax.grid()

    # Add the atomic numbers next to the points
    periodictable = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]

    for i, txt in enumerate(periodictable):
        ax.annotate(
            txt,
            (Z_vals[i], energies[i]),
            textcoords="offset points",
            xytext=(5, 8),
            ha="center",
        )

    bottom, top = min(energies), max(energies)
    scaling = 0.15 * (top - bottom)

    ax.set_ylim(bottom - scaling, top + scaling)
    ax.set_xticks(Z_vals, [f"${Z}$" for Z in Z_vals])
    # plt.yticks([-i * 10 for i in range(11)])
    ax.set_xlim(0, 11)
    ax = plt.gca()
    ax.set_box_aspect(1)
    fig.tight_layout()

    fig.savefig(save_path / "energy_plot.pdf")  # , bbox_inches="tight")


if __name__ == "__main__":
    plot_energy()
    # plt.show()
