import numpy as np
import sympy as sp

import re
from pathlib import Path

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
    path = Path(__file__).parents[1] / "matrix_elements.txt"

    values = np.zeros((n_max, n_max, n_max, n_max), dtype=object)

    with open(path) as infile:
        for line in infile:
            indicies, value = read_line(line)
            values[indicies] = value

    return values


def to_eV(value: float) -> float:
    """Convert from atomic units to eV

    Args:
        value (float): The value in atomic units

    Returns:
        float: The value in eV
    """
    return value * 2 * 13.6
