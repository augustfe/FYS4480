import sympy as sp

# a, c, i = sp.symbols("a c i")
g = sp.symbols("g")

I = (1, 2)
A = (3, 4)

total = 2 - g

one = 0
for i in I:
    for a in A:
        one += g**2 / (i - a)

total += one / 8

four = 0
for i in I:
    for a in A:
        for c in A:
            four += g**3 / ((i - a) * (i - c))

total += -four / 32

five = 0
for i in I:
    for k in I:
        for a in A:
            five += g**3 / ((i - a) * (k - a))

total += -five / 32

eight = 0
for i in I:
    for a in A:
        eight += g**3 / ((i - a) ** 2)

total += eight / 16

nine = 0
for i in I:
    for a in A:
        nine += g**3 / ((i - a) ** 2)

# total += nine / 16

print(total.factor())

g_func = sp.lambdify(g, total)

import numpy as np
import matplotlib.pyplot as plt
from exact_energy import get_states, get_energy_from_states

g_values = np.linspace(-1, 1, 100)
plt.plot(g_values, g_func(g_values), label="RS")

max_level = 4
states = get_states(max_level)[:-1]
g_values = np.linspace(-1, 1, 100)
energies = np.zeros((len(g_values), 5))
for i, g in enumerate(g_values):
    energies[i] = get_energy_from_states(g, states)

# plt.figure(figsize=(4, 3))

plt.plot(g_values, energies[:, 0], label="Ground state")

plt.legend()
plt.show()
