"""Relative inlet-projection error, bare and finite-Peclet.

Theory: the accompanying article (the inlet projection, its Parseval-type identity and
the truncation error of the series).

For each cutoff, the Galerkin coefficients satisfy ``G c = b`` in the chosen
projection space. Therefore

    ||f_K||^2 = c.T G c = c.T b,
    ||f - f_K||^2 = ||f||^2 - ||f_K||^2.

``CDBaseSolution.get_inlet_projection_square_norm()`` evaluates the first quantity
directly from the load vector assembled by the solver.  This script only supplies the
closed-form norm of the piecewise-constant inlet and plots the relative difference.
The bare coefficients are projected in L2_omega; a radial plot therefore shows
only the finite-Peclet curves rather than misusing the bare weighted norm.

Rates of convergence in the weighted plot
-----------------------------------------
The stratified inlet is discontinuous in the INTERIOR, at the interfaces, and both bases
resolve an interior jump the same way: the curves overlap at O(beta_max^-1), and sweeping
Pe barely moves them.  The Graetz inlet is smooth inside and mismatched only at the wall,
which is exactly where the two bases differ.  The bare radial equation carries the weight
omega = 1 - r^2, which VANISHES at r = 1, so r = 1 is a turning point: the bare modes stop
oscillating there and resolve a wall layer no finer than beta^(-2/3) (Airy scaling),
giving an error ~ beta_max^(-4/3).  The pencil adds kappa Lam^2 to that weight
(eq. radial_pencil), which does not vanish at the wall, so the modified modes oscillate
right up to it with wall wavenumber sqrt(kappa) Lam ~ beta, resolve a layer of width
1/beta_max, and give an error ~ beta_max^-2.

How far the shipped root tables reach
-------------------------------------
The plotted quantity is the error at truncation beta_max only while the retained set is a
genuine beta-ball. The stratified catalogue (roots_100x100.txt) is 100 x 100, so a block
beyond n = 99 would already be retained at beta_max = 200 and the first row runs out at
beta_max = 397; past that, raising beta_max stops adding modes and the curve flattens onto
whatever the table holds rather than onto the true truncation error. The Graetz catalogue
carries 400 axisymmetric roots reaching beta = 1599, and its missing n > 0 blocks are not
a truncation at all -- a uniform inlet excites only n = 0 -- so it is unconstrained here.
``supported_cutoff`` computes both limits and the sweep is clamped to them, which is why
the stratified curves stop short of the Graetz one. Lifting that needs a larger table
(``write_roots_and_norms`` in src/main.cpp), not a change to this script.

Why every point gets its own solve
----------------------------------
The modified coefficients come from one dense Galerkin system per angular block, whose
size is the truncation. Adding a mode changes every coefficient in its block. Each
displayed curve is consequently evaluated through the same simple path: build a solver
at each cutoff, run the requested setup, and ask the solver for ``||f_K||^2``.

Usage
-----
All run parameters are plain module-level constants below (no command-line interface): the
sweep itself as bare constants, everything the solver is told as the SOLVER_CONFIG dict.
Edit them directly and run:

    python3 parseval_identity_error_finite_peclet.py

The errors are stored in ``DATA_FILE`` and reused on the next run while the stored
configuration matches the script's; set ``FORCE_RECOMPUTE = True`` to recompute them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

import MassDiffusion3Dcpp as md


# Batch script: only PDF/PNG are written, so a headless backend keeps it usable from
# terminals with no display. Must precede every pyplot import.
matplotlib.use("Agg")

DEFS = md.definitions

# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
FIGURES_DIRECTORY = REPOSITORY / "figures"
OUTPUT_STEM = "Figure_2"
DATA_FILE = REPOSITORY / "data_files" / "parseval_identity_error_finite_peclet.npz"

# ---------------------------------------------------------------------------
# Run configuration. Edit directly.
# ---------------------------------------------------------------------------
PECLETS = (10.0, 100.0, 1000.0, 10000.0)  # one panel per value, kappa = Pe^-2
MAX_ROOT = 400.0                      # largest bare-root cutoff
LEVELS = 18                           # truncation levels, log-spaced in beta_max
MIN_LEVEL = 5.0                       # smallest cutoff on the x axis (a log axis cannot reach 0)
ENFORCE_BALL_TRUNCATION = True        # drop cutoffs the shipped root table cannot fill (see below)
FORCE_RECOMPUTE = False               # False: reuse DATA_FILE when its configuration matches
USE_TEX = True                        # render labels with a real LaTeX installation
PANEL_SIZE = (9.0, 6.5)               # inches per subplot; four values give a 2 x 2 grid, others a column

# Solver-side controls, applied by configure_solver to every solver this script builds.
# The projection space is selected explicitly; the other values are solver defaults.
#
# oversampling_factor/oversampling_margin fix the Gram node count N = max(factor*K +
# margin, minimum_factor*K). They are validated defaults: lowering the margin to 1
# leaves high-n, small-K blocks under-resolved and the coefficients wrong by orders of
# magnitude, WITHOUT tripping the solver's own gate. See the node-rule comment in
# src/finite_peclet_gram_radial.h.
SOLVER_CONFIG = {
    # The norm of both the inlet and its projection follows this choice.
    "projection_space": "weighted",
    "n_gauss_points": 100,
    "fp_tolerance": 1e-13,
    "fp_max_iter": 200,
    "gram_method": "gauss_jacobi_qr",   # or "ultraspherical" (weighted projection only)
    "oversampling_factor": 2,
    "oversampling_margin": 20,
    "minimum_factor": 2,
    "order_check": False,
    "cap_quad_margin": 100,             # DirectQuadrature only; unused by the representer
    "rhs_method": "representer",        # or "direct_quadrature"
    "representer_margin": 40,
}

GRAM_METHODS = {
    "gauss_jacobi_qr": md.GramMethod.GaussJacobiQR,
    "ultraspherical": md.GramMethod.Ultraspherical,
}
RHS_METHODS = {
    "direct_quadrature": md.RhsMethod.DirectQuadrature,
    "representer": md.RhsMethod.Representer,
}


# Interfaces z_i and nondimensional inlet values f_i, wall to wall. These are the
# normalized profiles plotted by the original script: f = (max(u) - u)/(max(u) - min(u)).
STRATIFIED_CASES = (
    ([0.2], [0.0, 1.0]),
    ([-0.55, 0.4], [0.0, 0.7, 1.0]),
    ([-0.7, -0.2, 0.4], [0.0, 0.3, 0.7, 1.0]),
)
# Isothermal Graetz in its natural nondimensional scaling: f = 1 at the inlet and
# f = 0 at the wall. The modal coefficients are independent of these two values,
# while this choice makes the user-defined inlet norm equal to the dimensionless one.
GRAETZ_CASE = (1.0, 0.0)

# Okabe & Ito (2008): separable under deuteranopia, protanopia and tritanopia.
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

# Colour encodes the inlet configuration, line style and marker encode the model.
BARE_STYLE = dict(linestyle=(0, (5, 2)), marker="o", markerfacecolor="none",
                  linewidth=3.0, markersize=11, markeredgewidth=1.6, alpha=0.42)
FP_STYLE = dict(linestyle="-", marker="s", linewidth=1.7, markersize=6.5)

# Power-law guides: (slope, label, anchor case, which curve, vertical offset factor).
# Each carries its own dash pattern and grey below so the three stay distinguishable.
REFERENCE_SLOPES = (
    (-1.0, r"$\mathcal{O}(\beta_K^{-1})$", "stratified2", "bare", 1.7),
    (-4.0 / 3.0, r"$\mathcal{O}(\beta_K^{-4/3})$", "graetz", "bare", 1.9),
    (-2.0, r"$\mathcal{O}(\beta_K^{-2})$", "graetz", "finite_peclet", 1.9),
)
GUIDE_STYLES = (
    dict(color="#8a8a8a", linestyle=(0, (1, 2)), linewidth=1.4),
    dict(color="#606060", linestyle=(0, (6, 2, 1, 2)), linewidth=1.4),
    dict(color="#303030", linestyle=(0, (9, 3)), linewidth=1.4),
)


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------
def configure_solver(solver):
    """Apply SOLVER_CONFIG to a freshly built solver, before any setup_* call.

    Unknown selector names raise here rather than silently falling back, mirroring the
    no-fallback policy the C++ setters follow.
    """
    gram = SOLVER_CONFIG["gram_method"]
    rhs = SOLVER_CONFIG["rhs_method"]
    projection = SOLVER_CONFIG["projection_space"]
    if gram not in GRAM_METHODS:
        raise ValueError(f"Unknown gram_method: {gram!r}. Expected one of {sorted(GRAM_METHODS)}.")
    if rhs not in RHS_METHODS:
        raise ValueError(f"Unknown rhs_method: {rhs!r}. Expected one of {sorted(RHS_METHODS)}.")
    if projection not in ("weighted", "radial"):
        raise ValueError("projection_space must be 'weighted' or 'radial'.")

    solver.set_number_of_gauss_points(int(SOLVER_CONFIG["n_gauss_points"]))
    # Older pre-built Graetz bindings do not expose these two setters even though
    # the C++ implementation has them.  In that case the solver's compiled defaults
    # are used; all other configuration remains unchanged.
    if hasattr(solver, "set_fp_tolerance"):
        solver.set_fp_tolerance(float(SOLVER_CONFIG["fp_tolerance"]))
    if hasattr(solver, "set_fp_max_iter"):
        solver.set_fp_max_iter(int(SOLVER_CONFIG["fp_max_iter"]))
    solver.set_gram_method(GRAM_METHODS[gram])
    solver.set_gram_oversampling(int(SOLVER_CONFIG["oversampling_factor"]),
                                 int(SOLVER_CONFIG["oversampling_margin"]))
    solver.set_gram_minimum_factor(int(SOLVER_CONFIG["minimum_factor"]))
    solver.set_gram_order_check(bool(SOLVER_CONFIG["order_check"]))
    solver.set_fp_cap_quad_margin(int(SOLVER_CONFIG["cap_quad_margin"]))
    solver.set_rhs_method(RHS_METHODS[rhs])
    solver.set_rhs_representer_margin(int(SOLVER_CONFIG["representer_margin"]))
    solver.set_projection_space(projection)
    return solver


# ---------------------------------------------------------------------------
# Reference norms
# ---------------------------------------------------------------------------
def radial_cap_area(z: float) -> float:
    """Area of the unit disk above the horizontal chord at height z."""
    return float(np.arccos(z) - z * np.sqrt(max(0.0, 1.0 - z * z)))


def stratified_reference_norm(solver, layer_values) -> float:
    """Squared inlet norm in the selected projection space, layer by layer."""
    edges = [-1.0, *solver.get_zi(), 1.0]
    cap = DEFS.cap_weight if SOLVER_CONFIG["projection_space"] == "weighted" else radial_cap_area
    areas = np.array([cap(lower) - cap(upper)
                      for lower, upper in zip(edges[:-1], edges[1:])])
    inlet = np.asarray(layer_values, dtype=float)
    return float(np.sum(inlet * inlet * areas))


def graetz_reference_norm() -> float:
    """Uniform inlet norm: pi/2 in L2_omega or pi in L2_r, times T0 squared."""
    disk_area = (DEFS.cap_weight(-1.0) if SOLVER_CONFIG["projection_space"] == "weighted"
                 else np.pi)
    return float(GRAETZ_CASE[0] ** 2 * disk_area)


# ---------------------------------------------------------------------------
# Projection errors
# ---------------------------------------------------------------------------
def projection_errors(make_solver, levels: np.ndarray, reference: float,
                      peclet: float | None = None) -> np.ndarray:
    """(||f||^2 - ||f_K||^2) / ||f||^2, with one solve per cutoff."""
    if peclet is None and SOLVER_CONFIG["projection_space"] == "radial":
        raise ValueError("The bare coefficients are projected in L2_omega, not L2_r.")
    errors = np.empty(levels.size)
    for position, level in enumerate(levels):
        solver = make_solver()
        solver.set_max_root(float(level))
        if peclet is None:
            solver.setup_bare_solution()
        else:
            solver.setup_fp_solution(peclet)
        projection = float(solver.get_inlet_projection_square_norm())
        errors[position] = (reference - projection) / reference
    return errors


def supported_cutoff(solver) -> float:
    """Largest beta_max for which {beta_nm <= beta_max} still fits inside the shipped catalog.

    Past this point raising beta_max stops adding the modes a genuine beta-ball would
    contain, so the plotted quantity is no longer "the error at truncation beta_max" and
    the curve flattens onto whatever the table happens to hold. Two independent limits:

    radial   the shortest row runs out, min_n beta_{n,last};
    angular  a block beyond the last tabulated one would have been retained, which first
             happens at beta_{n_last,0}. A single-row catalogue carries no angular limit:
             the Graetz table ships only n = 0 because a uniform inlet excites only the
             axisymmetric family, so nothing is missing there.
    """
    catalog = [np.asarray(row, dtype=float) for row in solver.get_bare_root_catalog()]
    radial = min(float(row[-1]) for row in catalog)
    angular = float(catalog[-1][0]) if len(catalog) > 1 else np.inf
    return min(radial, angular)


def truncation_levels(solver, name: str) -> np.ndarray:
    """Log-spaced cutoffs up to MAX_ROOT, clamped to what the catalog actually supports."""
    levels = np.logspace(np.log10(MIN_LEVEL), np.log10(MAX_ROOT), LEVELS)
    if not ENFORCE_BALL_TRUNCATION:
        return levels
    cutoff = supported_cutoff(solver)
    if cutoff >= MAX_ROOT:
        return levels
    print(f"  {name}: catalog supports a full beta-ball only up to beta_max = {cutoff:.1f}; "
          f"levels above it are dropped (set ENFORCE_BALL_TRUNCATION = False to keep them)")
    return np.logspace(np.log10(MIN_LEVEL), np.log10(cutoff), LEVELS)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class Curve:
    key: str
    label: str
    colour: str
    levels: np.ndarray
    bare: np.ndarray | None                  # only in the weighted projection
    finite_peclet: dict[float, np.ndarray]   # keyed by Pe


def run() -> list[Curve]:
    print("=" * 92)
    print(f"PARSEVAL DEFECT of the truncated series  |  Pe = {', '.join(f'{p:g}' for p in PECLETS)}, "
          f"beta_max <= {MAX_ROOT:g}")
    print(f"finite-Peclet inlet projection: {SOLVER_CONFIG['projection_space']}")
    print(f"Gram {SOLVER_CONFIG['gram_method']}, nodes N = "
          f"max({SOLVER_CONFIG['oversampling_factor']}K + {SOLVER_CONFIG['oversampling_margin']}, "
          f"{SOLVER_CONFIG['minimum_factor']}K)  |  RHS {SOLVER_CONFIG['rhs_method']}")
    print("=" * 92)

    curves: list[Curve] = []

    print("\nStratified (zero flux)")
    seed = configure_solver(md.CDStratifiedSolution(*STRATIFIED_CASES[0], 0))
    levels = truncation_levels(seed, "stratified")

    for index, (interfaces, layer_values) in enumerate(STRATIFIED_CASES):
        solver = configure_solver(md.CDStratifiedSolution(interfaces, layer_values, 0))
        reference = stratified_reference_norm(solver, layer_values)
        print(f"\n  {len(layer_values)} layers  z_i = {interfaces}  f_i = {layer_values}  "
              f"||f||^2 = {reference:.6f}")
        started = time.perf_counter()
        make_solver = lambda: configure_solver(
            md.CDStratifiedSolution(interfaces, layer_values, 0))
        bare = (projection_errors(make_solver, levels, reference)
                if SOLVER_CONFIG["projection_space"] == "weighted" else None)
        modified = {peclet: projection_errors(make_solver, levels, reference, peclet)
                    for peclet in PECLETS}
        report(levels, bare, modified, time.perf_counter() - started)
        key = f"stratified{len(layer_values)}"
        curves.append(Curve(key, *curve_presentation(key), levels, bare, modified))

    print("\nGraetz (isothermal)")
    seed = configure_solver(md.CDGraetzIsothermalSolution(*GRAETZ_CASE, 0))
    levels = truncation_levels(seed, "graetz")

    reference = graetz_reference_norm()
    print(f"\n  T0 = {GRAETZ_CASE[0]:g}, T_wall = {GRAETZ_CASE[1]:g}  "
          f"||f||^2 = {reference:.6f}")
    started = time.perf_counter()
    make_solver = lambda: configure_solver(
        md.CDGraetzIsothermalSolution(*GRAETZ_CASE, 0))
    bare = (projection_errors(make_solver, levels, reference)
            if SOLVER_CONFIG["projection_space"] == "weighted" else None)
    modified = {peclet: projection_errors(make_solver, levels, reference, peclet)
                for peclet in PECLETS}
    report(levels, bare, modified, time.perf_counter() - started)
    curves.append(Curve("graetz", *curve_presentation("graetz"), levels, bare, modified))

    return curves


def curve_presentation(key: str) -> tuple[str, str]:
    """Legend label and colour of a curve, from its key."""
    if key == "graetz":
        return r"Graetz", OKABE_ITO[3]
    layer_counts = [len(layer_values) for _, layer_values in STRATIFIED_CASES]
    index = layer_counts.index(int(key.removeprefix("stratified")))
    return rf"Stratified (${layer_counts[index]}$ layers)", OKABE_ITO[index]


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
def data_configuration() -> dict:
    """Everything the errors depend on; a cache computed otherwise is rejected."""
    return json.loads(json.dumps({
        "format_version": 1,
        "peclets": PECLETS,
        "max_root": MAX_ROOT,
        "levels": LEVELS,
        "min_level": MIN_LEVEL,
        "enforce_ball_truncation": ENFORCE_BALL_TRUNCATION,
        "solver": SOLVER_CONFIG,
        "stratified_cases": STRATIFIED_CASES,
        "graetz_case": GRAETZ_CASE,
    }, sort_keys=True))


def save_curves(curves: list[Curve]) -> None:
    """Store the errors of every curve in one compressed, non-pickled NumPy archive."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"configuration": np.array(json.dumps(data_configuration(), sort_keys=True)),
              "keys": np.array([curve.key for curve in curves])}
    for curve in curves:
        arrays[f"{curve.key}_beta_max"] = curve.levels
        if curve.bare is not None:
            arrays[f"{curve.key}_bare"] = curve.bare
        for peclet, errors in curve.finite_peclet.items():
            arrays[f"{curve.key}_Pe{peclet:g}"] = errors
    np.savez_compressed(DATA_FILE, **arrays)
    print(f"  saved data {DATA_FILE}")


def load_curves() -> list[Curve]:
    """Read the stored errors, refusing a file computed with another configuration."""
    with np.load(DATA_FILE, allow_pickle=False) as archive:
        if json.loads(str(archive["configuration"].item())) != data_configuration():
            raise ValueError("the stored configuration does not match this script")
        curves = []
        for key in archive["keys"].astype(str):
            bare = archive[f"{key}_bare"] if f"{key}_bare" in archive.files else None
            modified = {peclet: archive[f"{key}_Pe{peclet:g}"] for peclet in PECLETS}
            curves.append(Curve(key, *curve_presentation(key), archive[f"{key}_beta_max"],
                                bare, modified))
    return curves


def get_curves() -> list[Curve]:
    """Reuse DATA_FILE when allowed and compatible, otherwise compute and store the errors."""
    if not FORCE_RECOMPUTE:
        try:
            curves = load_curves()
            print(f"Reusing the errors stored in {DATA_FILE}")
            return curves
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not reuse {DATA_FILE}: {error}; recomputing the errors")
    curves = run()
    save_curves(curves)
    return curves


def report(levels: np.ndarray, bare: np.ndarray | None, modified: dict[float, np.ndarray],
           elapsed: float) -> None:
    headers = "".join(f" | {f'Pe = {peclet:g}':>11}" for peclet in modified)
    bare_header = f" | {'bare':>11}" if bare is not None else ""
    print(f"    {'beta_max':>9}{bare_header}{headers}")
    for position, level in enumerate(levels):
        bare_value = f" | {bare[position]:>11.4e}" if bare is not None else ""
        values = "".join(f" | {errors[position]:>11.4e}" for errors in modified.values())
        print(f"    {level:>9.2f}{bare_value}{values}")
    print(f"    ({elapsed:.1f} s)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def reference_guide(axis, levels: np.ndarray, errors: np.ndarray, slope: float,
                    offset: float, label: str, style: dict):
    """A grey power-law guide anchored to the middle of the curve it describes."""
    from matplotlib.lines import Line2D

    # Drawn over the asymptotic half of its own curve only: a guide spanning the whole
    # axis would cut across every other curve for no gain.
    anchor = len(levels) // 2
    constant = offset * errors[anchor] / levels[anchor] ** slope
    span = np.logspace(np.log10(levels[anchor - 1]), np.log10(levels[-1]), 64)
    axis.plot(span, constant * span ** slope, zorder=1, **style)
    return Line2D([0], [0], label=label, **style)


def draw_panel(axis, curves: list[Curve], peclet: float) -> list:
    """One Pe: each available curve and the weighted-only slope guides."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    by_key = {curve.key: curve for curve in curves}
    for curve in curves:
        if curve.bare is not None:
            axis.plot(curve.levels, curve.bare, color=curve.colour, **BARE_STYLE)
        axis.plot(curve.levels, curve.finite_peclet[peclet], color=curve.colour, **FP_STYLE)

    guide_handles = []
    guides = REFERENCE_SLOPES if SOLVER_CONFIG["projection_space"] == "weighted" else ()
    for (slope, label, key, kind, offset), style in zip(guides, GUIDE_STYLES):
        errors = by_key[key].bare if kind == "bare" else by_key[key].finite_peclet[peclet]
        guide_handles.append(reference_guide(axis, by_key[key].levels, errors, slope,
                                             offset, label, style))

    axis.set_title(rf"$\mathrm{{Pe}} = {peclet:g}$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 400]))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.4)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(which="major", direction="in", top=True, right=True, width=1.6, length=8)
    axis.tick_params(which="minor", direction="in", top=True, right=True, width=1.1, length=4.5)
    return guide_handles


def make_figure(curves: list[Curve]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # A 2 x 2 grid for exactly four values, otherwise one panel on top of the other.
    rows, columns = (2, 2) if len(PECLETS) == 4 else (len(PECLETS), 1)
    figure, axes = plt.subplots(rows, columns, sharex=True, sharey=True, squeeze=False,
                                figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows),
                                layout="constrained")
    for axis, peclet in zip(axes.flat, PECLETS):
        guide_handles = draw_panel(axis, curves, peclet)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\beta_K$")
    norm = r"\omega" if SOLVER_CONFIG["projection_space"] == "weighted" else "r"
    for axis in axes[:, 0]:
        axis.set_ylabel(rf"$\|f - f_K\|^2_{{L^2_{norm}}} \, / \, "
                        rf"\|f\|^2_{{L^2_{norm}}}$")

    # Every panel shares colours and guide styles, so one legend below the grid serves all.
    colour_handles = [Line2D([0], [0], color=curve.colour, linewidth=3.0, label=curve.label)
                      for curve in curves]
    if columns == 1:
        # Narrow figure: inlets fill the first legend column, guides the second.
        handles, legend_columns = colour_handles + guide_handles, 2 if guide_handles else 1
    elif not guide_handles:
        handles, legend_columns = colour_handles, len(colour_handles)
    else:
        # Wide figure: inlets on the first legend row, guides on the second. Legend
        # columns fill top-down, so interleave, padding the shorter list with blanks.
        blank = Line2D([], [], linestyle="none", label=" ")
        width = max(len(colour_handles), len(guide_handles))
        padded = [group + [blank] * (width - len(group))
                  for group in (colour_handles, guide_handles)]
        handles, legend_columns = [h for pair in zip(*padded) for h in pair], width
    # Place the shared legend above the grid; bbox_inches="tight" on savefig includes
    # the legend even though it is anchored outside the figure canvas.
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=legend_columns, framealpha=0.92, labelspacing=0.6, handlelength=1.3,
                  columnspacing=1.8)

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(figure)


def main() -> None:
    matplotlib.rcParams.update({
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 32,
        "axes.labelsize": 32,
        "axes.titlesize": 32,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
    })

    curves = get_curves()
    print("\nFigure")
    print("-" * 92)
    make_figure(curves)
    print("-" * 92)


if __name__ == "__main__":
    main()
