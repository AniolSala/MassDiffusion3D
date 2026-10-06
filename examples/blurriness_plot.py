import numpy as np
import matplotlib.pyplot as plt
from MassDiffusion3Dcpp import CDStratifiedSolution, CDGraetzIsothermalSolution

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})

# Axial positions (x = x_phys / (Pe R)), logarithmically spaced
x = np.logspace(-4, 0, 200).tolist()

# Stratified solution (layers from bottom to top) and Graetz isothermal solution
stratified = CDStratifiedSolution([-0.55, 0.45], [1.0, 0.3, 0.0], 0)
graetz = CDGraetzIsothermalSolution(1.0, 0.0, 0)

fig, ax = plt.subplots(figsize=(5.5, 3.9), layout="constrained")
for (name, sol), color in zip({"Stratified": stratified, "Graetz isothermal": graetz}.items(), ("C0", "C1")):
    # Bare solution (axial diffusion neglected) and finite-Peclet solution
    sol.setup_bare_solution()
    ax.semilogx(x, sol.get_blurriness(x), "--", color=color, label=rf"{name}, $\mathrm{{Pe}}\to\infty$")
    sol.setup_fp_solution(10.0)
    ax.semilogx(x, sol.get_blurriness(x), color=color, label=rf"{name}, $\mathrm{{Pe}}=10$")

# B = 0 is the inlet profile, B = 1 the far-field (fully mixed) state
ax.set(xlabel=r"$x$", ylabel=r"$B(x)$", ylim=(0, 1.05), title="Blurriness")
ax.grid(True, which="both", lw=0.3, alpha=0.5)
ax.legend()
plt.show()
