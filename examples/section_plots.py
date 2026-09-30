import numpy as np
import matplotlib.pyplot as plt
from MassDiffusion3Dcpp import CDStratifiedSolution, CDGraetzIsothermalSolution
from make_triangulation import make_triangulation

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})

# Define grid points of the cross-section x = x0, in polar coordinates (y = r sin(phi), z = r cos(phi))
x0, tri = 0.01, make_triangulation()
r, phi = np.minimum(np.hypot(tri.x, tri.y), 1.0), np.arctan2(tri.x, tri.y)

# Stratified solution: the interfaces (z coordinates) split the inlet into layers, and each layer
# has its own concentration, listed from bottom to top (last argument: verbose level)
layers_interfaces = [-0.55, 0.45]
layers_values = [1.0, 0.3, 0.0]
stratified = CDStratifiedSolution(layers_interfaces, layers_values, 0)

# Graetz isothermal solution: uniform inlet value T0 = 1.0 and wall value T_wall = 0.0
graetz = CDGraetzIsothermalSolution(1.0, 0.0, 0)

# Set up the finite-Peclet solutions, which include axial diffusion
peclet = 10.0
stratified.setup_fp_solution(peclet)
graetz.setup_fp_solution(peclet)

# Evaluate the solutions at the grid points (get_solution_at_planes returns one list per plane)
values = {"Stratified": stratified.get_solution_at_planes([x0], r, phi)[0],
          "Graetz isothermal": graetz.get_solution_at_planes([x0], r, phi)[0]}

# Plot both sections with a shared color scale
fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), layout="constrained")
for ax, (title, c) in zip(axes, values.items()):
    im = ax.tripcolor(tri, c, shading="gouraud", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.tricontour(tri, c, levels=np.linspace(0.1, 0.9, 9), colors="k", linewidths=0.4, alpha=0.5)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=0.8))
    ax.set(title=title, xlabel=r"$y$", ylabel=r"$z$", aspect="equal", xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
fig.colorbar(im, ax=axes, label=r"$\bar{c}$", shrink=0.8)
fig.suptitle(rf"Cross-section at $x = {x0}$")
plt.show()
