"""Normalised bare and finite-Peclet radial eigenfunctions for two wall conditions.

The exact radial mode is the finite-Peclet mode of the accompanying article.  At
every non-zero kappa its decay rate is
root-found from the finite-Peclet characteristic equation before the radial
factor is evaluated; this is not a perturbative approximation.

Each curve is divided by its exact radial norm.  In particular, the
finite-Peclet curves use Eq. (norm_fp),
``N^2_fp = integral_0^1 [(1-r^2) + 2*kappa*Lambda] R^2 r dr``;
the bare curves are its exact ``kappa = 0`` limit.  This keeps high-order
modes, such as ``n = 10, m = 51``, on the same meaningful scale.

The output is ``figures/Figure_1a.pdf`` at the root of the repository.  Edit the module-level
configuration below and run ``python3 plot_radial_functions.py``; there are no
command-line arguments.  The plotted curves are stored in ``DATA_FILE`` and reused
on the next run while the stored configuration matches the script's; set
``FORCE_RECOMPUTE = True`` to recompute them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

import MassDiffusion3Dcpp as md


# This script writes a PDF and should also work in a terminal without a graphical
# display (for example, a remote VS Code session).
matplotlib.use("Agg")


DEFS = md.definitions
# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
ROOT_DIRECTORY = REPOSITORY
FIGURES_DIRECTORY = REPOSITORY / "figures"
DATA_FILE = REPOSITORY / "data_files" / "radial_functions.npz"

# ---------------------------------------------------------------------------
# Plot configuration. Each entry is a displayed (n, m) mode label; no
# one-based shift is applied. Dirichlet modes are necessarily axisymmetric.
# ---------------------------------------------------------------------------
NEUMANN_MODES = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 50),
]
DIRICHLET_MODES = [
    (0, 5),
    (0, 10),
    (0, 20),
    (0, 50)
]

KAPPA_VALUES = [0.01, 0.1, 1.0, 10.0]
RADIAL_POINTS = 800
FORCE_RECOMPUTE = False  # False: reuse DATA_FILE when its configuration matches
USE_TEX = True
ROOT_REL_TOL = 1e-14
ROOT_MAX_ITER = 200
NORM_PANELS = 128

EXACT_LINESTYLES = (
    (0, (7, 2)),
    (0, (4, 1.5)),
    (0, (6, 1.5, 1.5, 1.5)),
    (0, (1.5, 1.5)),
)
LINE_WIDTH = 1.0


@dataclass(frozen=True)
class Problem:
    label: str
    wall_condition: "md.WallCondition"
    roots_path: Path

    def has_zero_mode(self, n: int) -> bool:
        return self.wall_condition == md.WallCondition.Neumann and n == 0


PROBLEMS = (
    Problem(
        label="Neumann",
        wall_condition=md.WallCondition.Neumann,
        roots_path=ROOT_DIRECTORY / "src/CDStratifiedSolution/data/roots_100x100.txt",
    ),
    Problem(
        label="Dirichlet",
        wall_condition=md.WallCondition.Dirichlet,
        roots_path=ROOT_DIRECTORY / "src/CDGraetzIsothermalSolution/data/graetz_roots_100.txt",
    ),
)


def solve_base_roots(n: int, wall: "md.WallCondition", count: int) -> np.ndarray:
    """Compute bare roots when the shipped table has no suitable row."""
    roots: list[float] = [0.0] if wall == md.WallCondition.Neumann and n == 0 else []
    beta = 0.25 if roots else 1e-6
    previous = DEFS.fp_char(n, beta * beta, 0.0, wall)
    while len(roots) < count:
        upper = beta + 0.25
        current = DEFS.fp_char(n, upper * upper, 0.0, wall)
        if (previous > 0.0) != (current > 0.0):
            root, _ = DEFS.fp_solve_bracketed(
                n, 0.0, wall, beta * beta, upper * upper, ROOT_REL_TOL, ROOT_MAX_ITER,
            )
            roots.append(math.sqrt(root))
        beta, previous = upper, current
        if beta > 20.0 * n + 8.0 * count + 1.0e4:
            raise RuntimeError(f"could not locate {count} bare roots for n = {n}")
    return np.asarray(roots)


def base_roots(problem: Problem, n: int, count: int) -> np.ndarray:
    """Read the shipped bare-root table, falling back to the scalar solver."""
    data = np.loadtxt(problem.roots_path)
    row = data if data.ndim == 1 and n == 0 else (data[n] if data.ndim == 2 and n < data.shape[0] else None)
    if row is None or row.size < count:
        return solve_base_roots(n, problem.wall_condition, count)
    return np.asarray(row[:count], dtype=float)


def exact_rate(problem: Problem, n: int, index: int, beta_row: np.ndarray, kappa: float) -> float:
    """The selected positive finite-Peclet decay rate, root-found on its branch."""
    rates, _ = DEFS.fp_solve_ladder(
        n, beta_row * beta_row, kappa, problem.wall_condition,
        problem.has_zero_mode(n), ROOT_REL_TOL, ROOT_MAX_ITER,
    )
    return float(rates[index])


def radial_values(n: int, b: float, bt: float, radii: np.ndarray) -> np.ndarray:
    """Evaluate the exact regular radial factor at the requested radii."""
    return np.fromiter((DEFS.psinm_r_fp(n, b, bt, float(r)) for r in radii), dtype=float,
                       count=radii.size)


def normalised_radial_values(n: int, b: float, bt: float, kappa: float,
                             rate: float, radii: np.ndarray) -> np.ndarray:
    """Evaluate R_nm / N_nm, using the exact generalised norm (eq. norm_fp)."""
    norm_squared = DEFS.fp_generalized_norm(n, b, bt, kappa, rate, NORM_PANELS)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise RuntimeError(
            f"invalid exact radial norm for n = {n}, Lambda = {rate}: {norm_squared}"
        )
    return radial_values(n, b, bt, radii) / math.sqrt(norm_squared)


def mode_curves(problem: Problem, n: int, m: int, radii: np.ndarray) -> dict[str, np.ndarray]:
    """One normalised radial mode, bare and at every kappa, keyed 'bare' and 'kappa<value>'."""
    if problem.wall_condition == md.WallCondition.Dirichlet and n != 0:
        raise ValueError("Dirichlet radial modes must have n = 0")

    beta_row = base_roots(problem, n, m + 1)
    beta = float(beta_row[m])
    curves = {"bare": normalised_radial_values(n, beta, beta, 0.0, beta * beta, radii)}
    for kappa in KAPPA_VALUES:
        rate = exact_rate(problem, n, m, beta_row, kappa)
        b = math.sqrt(rate)
        bt = DEFS.btilde_from_b(b, kappa)
        curves[f"kappa{kappa:g}"] = normalised_radial_values(n, b, bt, kappa, rate, radii)
    return curves


def data_configuration() -> dict:
    """Everything the curves depend on; a cache computed otherwise is rejected."""
    return json.loads(json.dumps({
        "format_version": 1,
        "neumann_modes": NEUMANN_MODES,
        "dirichlet_modes": DIRICHLET_MODES,
        "kappa_values": KAPPA_VALUES,
        "radial_points": RADIAL_POINTS,
        "root_rel_tol": ROOT_REL_TOL,
        "root_max_iter": ROOT_MAX_ITER,
        "norm_panels": NORM_PANELS,
    }, sort_keys=True))


def curve_name(problem: Problem, n: int, m: int, curve: str) -> str:
    return f"{problem.label}_n{n}_m{m}_{curve}"


def get_curves(radii: np.ndarray) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Reuse DATA_FILE when allowed and compatible, otherwise compute and store the curves."""
    modes = [(problem, n, m) for problem, problem_modes in zip(PROBLEMS, (NEUMANN_MODES, DIRICHLET_MODES))
             for n, m in problem_modes]
    curve_keys = ["bare", *(f"kappa{kappa:g}" for kappa in KAPPA_VALUES)]
    if not FORCE_RECOMPUTE:
        try:
            with np.load(DATA_FILE, allow_pickle=False) as archive:
                if json.loads(str(archive["configuration"].item())) != data_configuration():
                    raise ValueError("the stored configuration does not match this script")
                stored = {(problem.label, n, m): {curve: archive[curve_name(problem, n, m, curve)]
                                                  for curve in curve_keys}
                          for problem, n, m in modes}
            print(f"reusing the curves stored in {DATA_FILE}")
            return stored
        except (OSError, ValueError, KeyError) as error:
            print(f"could not reuse {DATA_FILE}: {error}; recomputing the curves")

    computed = {(problem.label, n, m): mode_curves(problem, n, m, radii) for problem, n, m in modes}
    arrays = {"configuration": np.array(json.dumps(data_configuration(), sort_keys=True)), "r": radii}
    for problem, n, m in modes:
        for curve, values in computed[(problem.label, n, m)].items():
            arrays[curve_name(problem, n, m, curve)] = values
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_FILE, **arrays)
    print(f"saved data {DATA_FILE}")
    return computed


def plot_mode(axis, n: int, m: int, radii: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    """Plot one normalised radial mode and its finite-Peclet deformations."""
    axis.plot(radii, curves["bare"], color="#202020", linewidth=LINE_WIDTH, linestyle="-")
    for kappa, linestyle in zip(KAPPA_VALUES, EXACT_LINESTYLES):
        axis.plot(radii, curves[f"kappa{kappa:g}"], color="#202020", linestyle=linestyle,
                  linewidth=LINE_WIDTH)

    axis.axhline(0.0, color="#9A9A9A", linewidth=0.45, alpha=0.65, zorder=0)
    axis.set_xlim(0.0, 1.0)
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
    axis.text(0.96, 0.92, rf"$n = {n},\; m = {m}$", transform=axis.transAxes,
              ha="right", va="top", fontsize=17)


def main() -> None:
    if not NEUMANN_MODES or not DIRICHLET_MODES:
        raise ValueError("NEUMANN_MODES and DIRICHLET_MODES must both be non-empty")
    if not KAPPA_VALUES or len(KAPPA_VALUES) > len(EXACT_LINESTYLES):
        raise ValueError(f"use one to {len(EXACT_LINESTYLES)} positive KAPPA_VALUES")
    if any(n < 0 or m < 0 for n, m in (*NEUMANN_MODES, *DIRICHLET_MODES)):
        raise ValueError("mode indices must be non-negative")
    if any(n != 0 for n, _ in DIRICHLET_MODES):
        raise ValueError("DIRICHLET_MODES must contain only n = 0 modes")
    if any(kappa <= 0.0 for kappa in KAPPA_VALUES):
        raise ValueError("KAPPA_VALUES must be positive")

    matplotlib.rcParams.update({
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.4,
    })
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    radii = np.linspace(0.0, 1.0, RADIAL_POINTS)
    curves = get_curves(radii)
    column_count = max(len(NEUMANN_MODES), len(DIRICHLET_MODES))
    figure, axes = plt.subplots(2, column_count, figsize=(3.0 * column_count + 1.9, 5.8),
                                 sharex=True, squeeze=False)
    for row, (problem, modes) in enumerate(zip(PROBLEMS, (NEUMANN_MODES, DIRICHLET_MODES))):
        for column, (n, m) in enumerate(modes):
            plot_mode(axes[row, column], n, m, radii, curves[(problem.label, n, m)])
            axes[row, column].yaxis.set_major_locator(
                MaxNLocator(nbins=4, integer=True, min_n_ticks=4)
            )
            axes[row, column].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        for axis in axes[row, len(modes):]:
            axis.set_visible(False)
        axes[row, 0].set_ylabel(r"$\tilde{\mathcal{R}}_{nm}(r) / \tilde{\mathcal{N}}_{nm}$")
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"$r$")
    figure.text(0.792, 0.72, "Neumann", rotation="vertical", va="center", ha="center", fontsize=17)
    figure.text(0.792, 0.28, "Dirichlet", rotation="vertical", va="center", ha="center", fontsize=17)
    figure.legend(
        [Line2D([], [], color="#202020", linewidth=2.6, linestyle="-")]
        + [Line2D([], [], color="#202020", linewidth=2.3, linestyle=style)
           for style in EXACT_LINESTYLES[:len(KAPPA_VALUES)]],
        [r"$\kappa = 0$"]
        + [rf"$\kappa = {kappa:g}$" for kappa in KAPPA_VALUES],
        loc="center left", bbox_to_anchor=(0.805, 0.50), framealpha=0.92,
        fontsize=14, handlelength=1.8, handletextpad=0.65, labelspacing=0.55,
        borderpad=0.55,
    )
    figure.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.12, wspace=0.18, hspace=0.16)

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIRECTORY / "Figure_1a.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
