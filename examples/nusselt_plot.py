import numpy as np
import matplotlib.pyplot as plt
from MassDiffusion3Dcpp import CDGraetzIsothermalSolution

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})

# Axial positions (x = x_phys / (Pe R)), logarithmically spaced
x = np.logspace(-4, 0, 200)

# Graetz isothermal solution: uniform inlet value T0 = 1.0 and wall value T_wall = 0.0.
# The Nusselt number is only defined for this problem (the stratified solution has zero-flux walls).
graetz = CDGraetzIsothermalSolution(1.0, 0.0, 0)

fig, ax = plt.subplots(figsize=(5.5, 3.9), layout="constrained")

# Bare solution (axial diffusion neglected, Peclet -> infinity)
graetz.setup_bare_solution()
ax.loglog(x, graetz.get_nusselt_number(x.tolist()), "k--", label=r"$\mathrm{Pe}\to\infty$")

# Finite-Peclet solutions, which include axial diffusion
for peclet in (50.0, 10.0, 2.0):
    graetz.setup_fp_solution(peclet)
    ax.loglog(x, graetz.get_nusselt_number(x.tolist()), label=rf"$\mathrm{{Pe}}={peclet:g}$")

# Fully developed value (last setup, i.e. the smallest Peclet number)
ax.axhline(graetz.get_fully_developed_nusselt_number(), color="gray", lw=0.6, ls=":")

ax.set(xlabel=r"$x$", ylabel=r"$\mathrm{Nu}(x)$", title="Local Nusselt number, Graetz isothermal problem")
ax.grid(True, which="both", lw=0.3, alpha=0.5)
ax.legend()
plt.show()
