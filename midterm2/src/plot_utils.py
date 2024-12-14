from __future__ import annotations  # noqa: CPY001, D100, INP001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coupled_cluster import CCD_energies
from exact_energy import CI_coeffs, CI_energies, FCI_energies
from rayleigh_s import RS_coeffs, compute_RS

# g_values = np.linspace(-1, 1, 100)


class Plotter:
    """A class used to create various plots for energy calculations."""

    def __init__(self) -> None:
        """Initialize the Plotter class with default values and settings."""
        self.dir_path = Path(__file__).parents[1]
        plt.style.use(self.dir_path / "src" / "latex.mplstyle")
        self.g_values = np.linspace(-1, 1, 101)

        self.FCI = FCI_energies(self.g_values)
        self.CI = CI_energies(self.g_values)
        self.CCD = CCD_energies(self.g_values)
        self.RS2 = compute_RS(2, self.g_values)
        self.RS3 = compute_RS(3, self.g_values)
        self.RS4 = compute_RS(4, self.g_values)

    @staticmethod
    def setup_figure(figsize: tuple[int, int] = (4, 3)) -> tuple[plt.Figure, plt.Axes]:
        """Set up a matplotlib figure and axes with default settings.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes.

        """
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_xlabel(r"$g$")
        return fig, ax

    def add_energy(
        self,
        ax: plt.Axes,
        energy: np.ndarray,
        label: str | None = None,
        linestyle: str = "-",
    ) -> None:
        """Add an energy plot to the given axes.

        Args:
            ax (plt.Axes): The axes to plot on.
            energy (np.ndarray): The energy values to plot.
            label (str | None, optional): The label for the plot. Defaults to None.
            linestyle (str, optional): The line style for the plot. Defaults to "-".

        """
        ax.plot(self.g_values, energy, label=label, linestyle=linestyle)

    def add_all_FCI(self, ax: plt.Axes, energies: np.ndarray) -> None:  # noqa: N802
        """Add all FCI energy plots to the given axes.

        Args:
            ax (plt.Axes): The axes to plot on.
            energies (np.ndarray): The energy values to plot.

        """
        linestyles = ["-"] * 3 + [":"] + ["-"] * 2
        labels = [rf"$\varepsilon_{i}$" for i in range(6)]

        for energy, linestyle, label in zip(energies.T, linestyles, labels):
            self.add_energy(ax, energy, label, linestyle)

    def plot_diff(
        self,
        energy1: np.ndarray,
        energy2: np.ndarray,
        label1: str,
        label2: str,
    ) -> None:
        """Plot the difference between two energy arrays.

        Args:
            energy1 (np.ndarray): The first energy array.
            energy2 (np.ndarray): The second energy array.
            label1 (str): The label for the first energy array.
            label2 (str): The label for the second energy array.

        Returns:
            plt.Figure: The matplotlib figure object.

        """
        fig, ax = self.setup_figure()
        self.add_energy(ax, energy1 - energy2)
        ax.set_ylabel(rf"$E_{{{label1}}} - E_{{{label2}}}$")
        ax.set_title(f"Difference between {label1} and {label2} groundstate energy")

        return fig

    def exercise_2(self) -> None:
        """Generate and save plots for exercise 2."""
        save_name = "b_eigenvalues_energy.pdf"
        fig, ax = self.setup_figure()

        self.add_all_FCI(ax, self.FCI)
        ax.set_ylabel(r"Energy")
        ax.set_title(r"Eigenvalues (FCI)")
        ax.legend()

        self.save(fig, save_name)

        save_name = "b_groundstate_energy.pdf"
        fig, ax = self.setup_figure()
        self.add_energy(ax, self.FCI[:, 0])
        ax.set_ylabel(r"Energy")
        ax.set_title(r"Groundstate energy (FCI)")

        self.save(fig, save_name)

    def exercise_3(self) -> None:
        """Generate and save plots for exercise 3."""
        FCI_gs = self.FCI[:, 0]  # noqa: N806
        CI_gs = self.CI[:, 0]  # noqa: N806
        save_name = "c_groundstate_energy.pdf"
        fig, ax = self.setup_figure()

        self.add_energy(ax, FCI_gs, "FCI")
        self.add_energy(ax, CI_gs, "CI")
        ax.set_ylabel(r"Energy")
        ax.set_title(r"Groundstate energy (FCI vs CI)")
        ax.legend()

        self.save(fig, save_name)

        fig = self.plot_diff(FCI_gs, CI_gs, "FCI", "CI")
        self.save(fig, "c_groundstate_energy_diff.pdf")

    def HF_plots(self) -> None:  # noqa: N802
        """Generate and save plots for Hartree-Fock (HF) calculations."""
        FCI_gs = self.FCI[:, 0]  # noqa: N806
        HF_gs = 2 - self.g_values  # noqa: N806

        fig, ax = self.setup_figure()
        ax.set_ylabel(r"Energy")
        ax.set_title(r"Groundstate energy (FCI vs HF)")

        self.add_energy(ax, FCI_gs, "FCI")
        self.add_energy(ax, HF_gs, "HF")
        ax.legend()

        self.save(fig, "e_groundstate_energy.pdf")

        fig = self.plot_diff(FCI_gs, HF_gs, "FCI", "HF")
        self.save(fig, "e_groundstate_energy_diff.pdf")

    def RS_plots(self) -> None:  # noqa: N802
        """Generate and save plots for RSPT calculations."""
        FCI_gs = self.FCI[:, 0]  # noqa: N806
        CI_gs = self.CI[:, 0]  # noqa: N806

        fig, ax = self.setup_figure()
        ax.set_ylabel(r"Energy")
        ax.set_title(r"Groundstate energy ((F)CI vs RS3)")

        self.add_energy(ax, FCI_gs, "FCI")
        self.add_energy(ax, CI_gs, "CI")
        self.add_energy(ax, self.RS3, "RS3")

        ax.legend()

        self.save(fig, "e_groundstate_energy_RS.pdf")

        fig, ax = self.setup_figure()
        ax.set_ylabel(r"$\Delta$ Energy")
        ax.set_title(r"Difference in groundstate energy ((F)CI vs RS3)")

        self.add_energy(ax, FCI_gs - self.RS3, r"$E_{FCI} - E_{RS}^{(3)}$")
        self.add_energy(ax, CI_gs - self.RS3, r"$E_{CI} - E_{RS}^{(3)}$")
        ax.legend()

        self.save(fig, "e_groundstate_energy_diff_RS.pdf")

    def exercise_5(self) -> None:
        """Generate and save plots for exercise 5."""
        self.HF_plots()
        self.RS_plots()

    def exercise_6(self) -> None:
        """Generate and save plots for exercise 6."""
        fig, ax = self.setup_figure()
        CI_gs = self.CI[:, 0]  # noqa: N806
        RS2 = self.RS2  # noqa: N806
        self.add_energy(ax, CI_gs, "CI")
        self.add_energy(ax, RS2, "RS2")

        ax.legend()
        ax.set_title(r"Groundstate energy (CI vs RS2)")
        ax.set_ylabel(r"Energy")
        self.save(fig, "f_groundstate_energy.pdf")

        fig, ax = self.setup_figure()
        ax.set_title(r"Difference in groundstate energy (CI vs RS2)")
        ax.set_ylabel(r"$E_{CI} - E_{RS}^{(2)}$")
        self.add_energy(ax, CI_gs - self.RS2)

        self.save(fig, "f_groundstate_energy_diff.pdf")

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        CI_coeffs_ = CI_coeffs(self.g_values)  # noqa: N806
        RS2_coeffs = RS_coeffs(self.g_values)  # noqa: N806

        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.plot(self.g_values, CI_coeffs_[:, i], label="CI")
            ax.plot(self.g_values, RS2_coeffs[:, i], label="RS2")
            ax.set_title(rf"$\vert \Phi_{{{i + 1}}} \rangle$")
            ax.set_ylabel(r"Coefficient")
            ax.legend()

        fig.suptitle(r"CI vs RS2 coefficients")
        # ax.set_title(r"CI vs RS2 coefficients")
        ax.set_ylabel(r"Coefficient")
        ax.legend()

        self.save(fig, "f_coefficients.pdf")

    def exercise_7(self) -> None:
        """Generate and save plots for exercise 7."""
        fig, ax = self.setup_figure()
        FCI_gs = self.FCI[:, 0]  # noqa: N806
        RS4 = self.RS4  # noqa: N806

        self.add_energy(ax, FCI_gs, "FCI")
        self.add_energy(ax, RS4, "RS4")

        ax.legend()
        ax.set_title(r"Groundstate energy (FCI vs RS4)")
        ax.set_ylabel(r"Energy")
        self.save(fig, "g_groundstate_energy.pdf")

        fig, ax = self.setup_figure()
        ax.set_title(r"Difference in groundstate energy (FCI vs RS4)")
        ax.set_ylabel(r"$E_{FCI} - E_{RS}^{(4)}$")
        self.add_energy(ax, FCI_gs - RS4)

        self.save(fig, "g_groundstate_energy_diff.pdf")

    def save(self, fig: plt.Figure, filename: str) -> None:
        """Save the figure to a file.

        Args:
            fig (plt.Figure): The figure to save.
            filename (str): The name of the file to save the figure as.

        """
        fig.savefig(self.dir_path / "figures" / filename, bbox_inches="tight")
        plt.clf()

    def CCD_plots(self) -> None:
        """Generate and save plots for CCD calculations."""
        fig, ax = self.setup_figure()
        FCI_gs = self.FCI[:, 0]
        CCD = self.CCD

        self.add_energy(ax, FCI_gs, "FCI")
        self.add_energy(ax, CCD, "CCD")

        ax.legend()
        ax.set_title(r"Groundstate energy (FCI vs CCD)")
        ax.set_ylabel(r"Energy")
        self.save(fig, "ccd_groundstate_energy.pdf")

        fig, ax = self.setup_figure()
        ax.set_title(r"Difference in groundstate energy (FCI vs CCD)")
        ax.set_ylabel(r"$E_{FCI} - E_{CCD}$")
        self.add_energy(ax, FCI_gs - CCD)

        self.save(fig, "ccd_groundstate_energy_diff.pdf")

    def diff_plots(self) -> None:
        """Generate difference plots for the different methods."""

        fig, ax = self.setup_figure()
        FCI_gs = self.FCI[:, 0]
        CI_gs = self.CI[:, 0]
        HF = 2 - self.g_values
        RS2 = self.RS2
        RS3 = self.RS3
        RS4 = self.RS4
        CCD = self.CCD

        def add_diff(energy: np.ndarray, suffix: str, abs: bool = False) -> None:
            diff = FCI_gs - energy
            if abs:
                diff = np.abs(diff)
            label = f"$E_{suffix}$"
            self.add_energy(ax, diff, label)

        energies = [
            (CI_gs, r"{CI}"),
            # (HF, "HF"),
            (RS2, r"{RS}^{(2)}"),
            (RS3, r"{RS}^{(3)}"),
            # (RS4, r"{RS}^{(4)}"),
            (CCD, r"{CCD}"),
        ]

        for energy, suffix in energies:
            add_diff(energy, suffix)

        ax.legend()
        ax.set_title(r"Difference in groundstate energy (FCI vs other methods)")
        ax.set_ylabel(r"$E_{FCI} - E_*$")
        self.save(fig, "differences.pdf")

        fig, ax = self.setup_figure((4, 9 / 4))
        ax.set_title(r"Absolute difference in groundstate energy")
        ax.set_ylabel(r"$|E_{FCI} - E_*|$")

        energies = [
            (CI_gs, r"{CI}"),
            (RS2, r"{RS}^{(2)}"),
            (RS3, r"{RS}^{(3)}"),
            (CCD, r"{CCD}"),
            (HF, r"{HF}"),
            (RS4, r"{RS}^{(4)}"),
        ]

        for energy, suffix in energies:
            add_diff(energy, suffix, abs=True)

        ax.set_yscale("log")
        ax.legend()
        self.save(fig, "absolute_differences.pdf")

    def RS_diff_plots(self) -> None:
        """Generate and save plots for the difference with RS2, RS3 and RS4."""
        fig, ax = self.setup_figure()
        FCI_gs = self.FCI[:, 0]
        energies = [
            (self.RS2, r"{RS}^{(2)}"),
            (self.RS3, r"{RS}^{(3)}"),
            (self.RS4, r"{RS}^{(4)}"),
        ]

        def add_diff(energy: np.ndarray, suffix: str, abs: bool = False) -> None:
            diff = FCI_gs - energy
            if abs:
                diff = np.abs(diff)
            label = f"$E_{suffix}$"
            self.add_energy(ax, diff, label)

        for energy, suffix in energies:
            add_diff(energy, suffix)

        ax.legend()
        ax.set_title(r"Difference in groundstate energy (FCI vs RS)")
        ax.set_ylabel(r"$E_{FCI} - E_{RS}^{(*)}$")

        self.save(fig, "rs_diff.pdf")


def midterm2_plots() -> None:
    """Generate and save plots for the midterm 2."""
    plotter = Plotter()
    plotter.exercise_2()
    plotter.exercise_3()
    plotter.exercise_5()
    plotter.exercise_6()
    plotter.exercise_7()


def exam_plots() -> None:
    plotter = Plotter()
    # plotter.CCD_plots()
    # plotter.diff_plots()
    plotter.RS_diff_plots()


if __name__ == "__main__":
    # midterm2_plots()
    exam_plots()
