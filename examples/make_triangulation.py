import numpy as np
from matplotlib.tri import Triangulation


def make_triangulation(h=0.015):
    """Delaunay triangulation of the unit disk, built from concentric rings of spacing ~h."""
    r, phi = np.array([(r, -np.pi + 2 * np.pi * (j + 0.5) / n) for r in np.linspace(0, 1, int(1 / h))
                       for n in [max(int(2 * np.pi * r / h), 1)] for j in range(n)]).T
    return Triangulation(r * np.sin(phi), r * np.cos(phi))
