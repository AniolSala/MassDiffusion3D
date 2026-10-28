"""Exact finite-Peclet decay rates Lam_nm(kappa), and the flow-equation checks on them.

Theory: the accompanying article, subsection "Flow equation and monotonicity".

The rates are obtained the way the solver obtains them: as the root with zero-based radial
index m of the scalar characteristic function of eqs.
(char_dirichlet_fp)/(char_neumann_fp), via the
compiled ladder solver ``fp_solve_ladder``. That is the robust route and the one the
manuscript advocates -- one bracketed scalar root-find per mode, no mode integrals.

The flow equation (eq. flow_equation)

    dLam/dkappa = - Lam^2 * C / (B + 2 kappa Lam C) = - Lam^2 / (1 - <r^2> + 2 kappa Lam)

is deliberately NOT integrated here. Its role in the paper is to supply the enclosure
(eq. rate_enclosure) that makes that root-find safe, plus the initial slope behind the
closed-form seed. It is checked POINTWISE against central differences of the root-found
rates, which is both the stronger statement and numerically robust.

Why not integrate it: the right-hand side is only meaningful on an eigenpair. Off the
branch the mode no longer satisfies the wall condition, R(1) swells by many orders of
magnitude, and since C = int r R^2 dr weights the wall fully while B = int (1-r^2) r R^2 dr
suppresses it, the ratio B/C is set by a spurious wall tail. Measured at n = 90: a relative
offset of 1e-6 in Lam drives the right-hand side from its true value 2.0 to 15.9, and 1e-5
to 81.1. Any ODE integrator evaluates off-branch by construction, so this is not a defect of
RK4 -- it is a property of that way of evaluating the right-hand side. Root-finding sidesteps
it entirely and works at every n.

Everything numerical comes from the compiled ``src/`` kernels through
``MassDiffusion3Dcpp.definitions``: ``fp_solve_ladder`` (the rates, one call per angular
block and kappa -- all m at once), ``fp_char`` (base roots for blocks no table covers),
``fp_unweighted_overlap`` / ``fp_generalized_norm`` / ``btilde_from_b`` (the flow-equation
right-hand side, for the pointwise check only). Base roots Lam_nm(0) = beta_nm^2 come from
the tables shipped with each subclass:

    src/CDStratifiedSolution/data/roots_100x100.txt          (zero flux / Neumann)
    src/CDGraetzIsothermalSolution/data/graetz_roots_100.txt (isothermal / Dirichlet)

The Graetz tables hold only the n = 0 row, so any n > 0 Dirichlet block is solved from
src/ instead (see ``base_roots``); that fallback reproduces both shipped tables exactly.

Outputs (at the root of the repository; the figures have no in-image captions)
--------------------------------------------------------------------------------
``data_files/flow_equation_data.npz``  numerical data used by every selected figure;
                                       reusable by setting ``FORCE_RECOMPUTE = False``.
``figures/Figure_1b``                  Lam_nm(kappa)/beta_nm^2 for both wall conditions,
                                       against its unit bare ceiling (eq. rate_ordering) and
                                       the Bessel floor q_nm/(beta_nm^2 sqrt(kappa))
                                       (eq. rate_large_kappa). Fig. 1b of the article.

Three diagnostic figures, not shown in the article, are drawn when added to ``FIGURES``:
``flow_equation_similarity_ratio``     Lam_nm(kappa)/beta_nm^2 vs kappa.
``flow_equation_similarity_residual``  Lam_nm/Lam_seed - 1 vs kappa: the residual the
                                       closed-form seed leaves, i.e. what the Newton
                                       iteration starts from.
``flow_equation_validation``           the flow equation checked pointwise against central
                                       differences of the root-found rates.

Usage
-----
All run parameters are plain module-level constants below (no command-line
interface) -- edit them directly and run:

    python3 solve_flow_equation.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

import MassDiffusion3Dcpp as md


# This batch script only writes PNG/PDF files.  Use a non-interactive backend so it
# also runs from terminals without a valid X/Wayland display (e.g. remote VS Code).
# This must precede every pyplot import, which deliberately remains local to the
# plotting functions below.
matplotlib.use("Agg")


DEFS = md.definitions

# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
ROOT_DIRECTORY = REPOSITORY
FIGURES_DIRECTORY = REPOSITORY / "figures"
DATA_DIRECTORY = REPOSITORY / "data_files"
DATA_FILE = DATA_DIRECTORY / "flow_equation_data.npz"

# ---------------------------------------------------------------------------
# Run configuration. Edit directly.
# ---------------------------------------------------------------------------
# The same (n, m) grid is used for both wall conditions.  Both indices are zero-based:
# ``m`` is the column of the base-root table and therefore identifies R_nm directly.
# The Neumann (n, m) = (0, 0) mode is the exact constant mode, Lam = beta = 0.  It is
# omitted from normalised-rate plots because Lam / beta^2 is undefined, not because the
# radial index is shifted.
ANGULAR_INDICES = np.linspace(0, 60, 7, dtype=int)
RADIAL_INDICES = [1, 2, 3, 4, 5]
# Keep these low for a paper figure: high m puts every curve deep in the
# kappa*beta^2 >> 1 regime, where they are parallel straight lines and the beta^2
# plateau is off-scale.  High (n, m) values do work; they just make a poor figure.

KAPPA_MAX = 1.0                        # largest kappa = Pe^-2 (1.0 -> Pe = 1)
KAPPA_MIN = 1e-8                       # smallest non-zero kappa on the output grid
KAPPA_MID = 1e-7                       # upper edge of the low-kappa inset
KAPPA_POINTS = 120                     # log-spaced output points above kappa = 0
QUAD_PANELS = 16                       # Gauss-Legendre panels (16 nodes each) for the flow-equation
                                       # quadratures; used only by the pointwise check
ROOT_REL_TOL = 1e-14                   # relative tolerance handed to fp_solve_ladder
ROOT_MAX_ITER = 200                    # iteration cap handed to fp_solve_ladder
CHECK_POINTS = 9                       # kappa values at which the flow equation is checked pointwise
RUN_FLOW_CHECK = True                  # verify eq. (flow_equation) against central differences
FORCE_RECOMPUTE = False                 # False: reuse DATA_FILE when its configuration matches
USE_TEX = True                         # render labels with a real LaTeX installation
SHOW_RATE_GUIDES = True                # draw the beta^2 ceiling and the Bessel-zero floor per mode
MAX_CHECK_CURVES = 6                   # cap on curves in the pointwise-check figure
FIGURES = ("rates",)                   # "rates" is Fig. 1b; also "similarity_ratio", "similarity_residual", "validation"
LEGEND_LABEL_FONT_SIZE = 32            # font size of the compact n/m legend entries
PANEL_TITLE_FONT_SIZE = 38             # font size of the Neumann/Dirichlet panel titles
AXIS_LABEL_FONTSIZE = 42
TICKS_FONTSIZE = 38
INSET_TICKS_FONTSIZE = 30
Y_AXIS_LABEL_PAD = 8

# Each radial branch gets one dash pattern, shared by its two wall-condition curves and
# all angular families.  The dash families remain distinguishable in greyscale and at
# journal-column scale.
RADIAL_LINESTYLES = (
    "-",                              # solid
    (0, (5, 1)),                      # densely dashed
    (0, (1, 1)),                      # densely dotted
    (0, (3, 1, 1, 1)),                # densely dash-dotted
    (0, (3, 1, 1, 1, 1, 1)),          # densely dash-dot-dotted
    (0, (5, 5)),                      # loosely dashed
    (0, (1, 5)),                      # loosely dotted
    (0, (3, 5, 1, 5)),                # loosely dash-dotted
)

# Okabe & Ito (2008), the standard qualitative palette that stays separable under
# deuteranopia, protanopia and tritanopia. Yellow sits last: it is the weakest of the
# eight on a white background, so it is only reached when many curves are drawn.
OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C510A",  # deep brown
    "#F0E442",  # yellow
)

# Markers distinguish the two wall conditions only in figures that overlay them.  In the
# rate figure they are already separated into panels, so the marker is not drawn there.
WALL_MARKERS = {
    "stratified": "o",
    "graetz": "s",
}


# ---------------------------------------------------------------------------
# Problems: which pre-computed base-root table seeds which wall condition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    """One wall condition, its base-root table and the blocks to integrate."""

    name: str
    label: str
    short: str          # compact legend form, for plots that mix both wall conditions
    roots_path: Path
    norms_path: Path
    wall_condition: "md.WallCondition"

    def has_zero_mode(self, n: int) -> bool:
        """True when Lam = 0 is an exact eigenvalue at this n (Neumann, n = 0)."""
        return self.wall_condition == md.WallCondition.Neumann and n == 0


PROBLEMS = (
    Problem(
        name="stratified",
        label=r"Neumann",
        short=r"Neumann",

        roots_path=ROOT_DIRECTORY / "src/CDStratifiedSolution/data/roots_100x100.txt",
        norms_path=ROOT_DIRECTORY / "src/CDStratifiedSolution/data/norms_100x100_cpp.txt",
        wall_condition=md.WallCondition.Neumann,
    ),
    Problem(
        name="graetz",
        label=r"Dirichlet",
        short=r"Dirichlet",

        roots_path=ROOT_DIRECTORY / "src/CDGraetzIsothermalSolution/data/graetz_roots_100.txt",
        norms_path=ROOT_DIRECTORY / "src/CDGraetzIsothermalSolution/data/graetz_norms_100.txt",
        wall_condition=md.WallCondition.Dirichlet,
    ),
)


def solve_base_roots(n: int, wall: "md.WallCondition", count: int,
                     beta_max: float = 0.0, scan: float = 0.25) -> np.ndarray:
    """Bare roots beta_nm from the src/ kernel, for blocks no shipped table covers.

    At kappa = 0 the modified mode collapses onto the bare one (bt == b), so the zeros of
    fp_char(n, beta^2, 0, wall) ARE the base roots. Scan upward in beta for sign changes
    (root spacing is ~pi in beta, far wider than the scan step) and refine each bracket
    with fp_solve_bracketed. Reproduces both shipped tables to every digit they carry.
    """
    # The roots start near n and are spaced by rather more than pi, so a fixed cap is
    # wrong as soon as either index grows. Estimate generously and keep extending rather
    # than failing: the scan is cheap next to everything else here.
    if beta_max <= 0.0:
        beta_max = 2.0 * n + 6.0 * (count + 2) + 40.0
    hard_limit = 20.0 * beta_max + 1.0e4

    roots: list[float] = []
    beta = scan if (wall == md.WallCondition.Neumann and n == 0) else 1e-6
    if wall == md.WallCondition.Neumann and n == 0:
        roots.append(0.0)  # the exact constant mode
    previous = DEFS.fp_char(n, beta * beta, 0.0, wall)
    while len(roots) < count:
        upper = beta + scan
        current = DEFS.fp_char(n, upper * upper, 0.0, wall)
        if (previous > 0.0) != (current > 0.0):
            root, _ = DEFS.fp_solve_bracketed(n, 0.0, wall, beta * beta, upper * upper, 1e-14, 200)
            roots.append(math.sqrt(root))
        beta, previous = upper, current
        if beta > beta_max:
            beta_max *= 2.0          # the estimate was short; keep going from where we are
            if beta > hard_limit:
                raise RuntimeError(
                    f"solve_base_roots: only {len(roots)}/{count} roots for n = {n} below "
                    f"beta = {hard_limit:.0f}; the characteristic function is probably "
                    f"underflowing rather than the roots being absent"
                )
    return np.array(roots)


def bessel_zeros(n: int, wall: "md.WallCondition", count: int) -> np.ndarray:
    """The q_nm of eq. (rate_large_kappa): the exact kappa -> infinity limit of the rates.

    As kappa -> infinity with Lam -> 0 at fixed kappa*Lam^2, the term Lam*(1-r^2) drops out
    of eq. (radial_mode_fp) and the transverse operator degenerates to Bessel's equation
    with eigenvalue kappa*Lam^2, quantised by the same wall condition. Hence
    Lam_nm ~ q_nm / sqrt(kappa), with q_nm the m-th non-negative zero of J_n' (zero flux,
    including the constant mode q_00 = 0) or of J_n (Dirichlet).

    Checked against the exact rates: at kappa = 1e6, Lam*sqrt(kappa)/q deviates from 1 by
    at most 2e-4 for both walls and n = 0, 5.
    """
    from scipy.special import jn_zeros, jnp_zeros

    if wall == md.WallCondition.Dirichlet:
        return jn_zeros(n, count)
    if n == 0:  # the constant mode, absent from jnp_zeros
        return np.concatenate(([0.0], jnp_zeros(0, count - 1)))
    return jnp_zeros(n, count)


def base_roots(problem: Problem, n: int, count: int) -> np.ndarray:
    """Base roots beta_nm for one angular block, preferring the shipped table.

    The tables are one row per angular index, columns ascending in m; these are the roots
    beta_nm, so the bare decay rates -- the initial conditions of the flow -- are their
    squares. The Graetz tables ship only the n = 0 row (a uniform inlet excites the
    axisymmetric family alone), so any n > 0 Dirichlet block is solved from src/ instead.
    """
    data = np.loadtxt(problem.roots_path)
    row = None
    if data.ndim == 1:
        if n == 0:
            row = data
    elif n < data.shape[0]:
        row = data[n]
    if row is None or row.size < count:
        return solve_base_roots(n, problem.wall_condition, count)
    return row


# ---------------------------------------------------------------------------
# Flow-equation right-hand side (used only for the pointwise check, never integrated)
# ---------------------------------------------------------------------------
class FlowRightHandSide:
    """dLam/dkappa of eq. (flow_equation), from the two src/ quadratures.

    N^2_fp = B + 2 kappa Lam C is exactly the denominator (eq. norm_fp), so the
    right-hand side is -Lam^2 * C / N^2_fp with C the unweighted self-overlap
    (eq. U_nm_def). Valid only ON an eigenpair -- see the module docstring.
    """

    def __init__(self, n: int, panels: int = QUAD_PANELS) -> None:
        self.n, self.panels = int(n), int(panels)

    def derivative(self, kappa: float, lam: float) -> float:
        b = math.sqrt(lam)
        bt = DEFS.btilde_from_b(b, kappa)
        overlap = DEFS.fp_unweighted_overlap(self.n, b, bt, b, bt, self.panels)
        norm_fp = DEFS.fp_generalized_norm(self.n, b, bt, kappa, lam, self.panels)
        return -lam * lam * overlap / norm_fp


def bare_alpha(n: int, beta: float, panels: int = QUAD_PANELS) -> float:
    """alpha_nm = U^n_mm / N_nm^2 (eq. seed_root), from the bare mode at kappa = 0.

    Equals 1/(1 - <r^2>^(0)_nm), hence also minus the initial slope of the flow divided
    by beta^4. Exactly 2 for the Neumann axisymmetric family, by the virial identity.
    """
    lam = beta * beta
    overlap = DEFS.fp_unweighted_overlap(n, beta, beta, beta, beta, panels)
    norm = DEFS.fp_generalized_norm(n, beta, beta, 0.0, lam, panels)
    return overlap / norm


def seed_rate(beta_squared: float, kappa: float, alpha: float) -> float:
    """Closed-form seed of eq. (seed_root): 2 beta^2 / (1 + sqrt(1 + 4 kappa alpha beta^2)).

    Deliberately not DEFS.fp_seed, which additionally switches to a diffusion-dominated
    Bessel-zero branch; the object of interest here is the closed form itself.
    """
    return 2.0 * beta_squared / (1.0 + math.sqrt(1.0 + 4.0 * kappa * alpha * beta_squared))


# ---------------------------------------------------------------------------
# The rates themselves: the src/ ladder root-finder
# ---------------------------------------------------------------------------
def exact_rates(problem: Problem, n: int, roots_row: np.ndarray, kappa: float,
                alpha_row: np.ndarray) -> np.ndarray:
    """Lam_nm(kappa) for a whole angular block, from fp_solve_ladder.

    One call returns every m of the block, sweeping upward inside the enclosure
    (eq. rate_enclosure): Lam_{n,m-1} < Lam_nm < beta_nm^2.
    """
    rates, _ = DEFS.fp_solve_ladder(
        n, roots_row * roots_row, kappa, problem.wall_condition,
        problem.has_zero_mode(n), ROOT_REL_TOL, ROOT_MAX_ITER, alpha_row,
    )
    return np.asarray(rates, dtype=float)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class ModeRun:
    """Everything computed for one (problem, n, m) mode."""

    problem: Problem
    n: int
    index: int          # column in the base-root table (0-based)
    m: int              # physical radial index: exactly the zero-based table column
    beta: float
    alpha: float
    bessel_zero: float  # q_nm, the exact kappa -> infinity limit (eq. rate_large_kappa)
    kappa: np.ndarray
    rate: np.ndarray    # Lam_nm(kappa) from the root-finder
    base_roots: np.ndarray
    alpha_row: np.ndarray


@dataclass
class PlotData:
    """Numerical arrays needed to reproduce every requested plot."""

    runs: list[ModeRun]
    validation_kappas: np.ndarray
    validation_gaps: np.ndarray


def build_kappa_grid(kappa_min: float, kappa_max: float, points: int) -> np.ndarray:
    """kappa = 0 (the bare limit) followed by log-spaced points up to kappa_max."""
    return np.concatenate(([0.0], np.logspace(math.log10(kappa_min), math.log10(kappa_max), points)))


def configured_indices(values, name: str) -> tuple[int, ...]:
    """Validate one zero-based index list while preserving its configured order."""
    indices = tuple(int(value) for value in values)
    if not indices:
        raise ValueError(f"{name} must contain at least one index")
    if any(index < 0 for index in indices):
        raise ValueError(f"{name} must contain only non-negative indices: {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not repeat an index: {indices}")
    return indices


def data_configuration() -> dict:
    """Configuration fingerprint used to reject stale or incompatible cached data."""
    return {
        "format_version": 1,
        "angular_indices": list(configured_indices(ANGULAR_INDICES, "ANGULAR_INDICES")),
        "radial_indices": list(configured_indices(RADIAL_INDICES, "RADIAL_INDICES")),
        "kappa_min": KAPPA_MIN,
        "kappa_max": KAPPA_MAX,
        "kappa_points": KAPPA_POINTS,
        "quad_panels": QUAD_PANELS,
        "root_rel_tol": ROOT_REL_TOL,
        "root_max_iter": ROOT_MAX_ITER,
        "include_validation": "validation" in FIGURES and RUN_FLOW_CHECK,
        "check_points": CHECK_POINTS,
        "max_check_curves": MAX_CHECK_CURVES,
    }


def run() -> list[ModeRun]:
    kappa_grid = build_kappa_grid(KAPPA_MIN, KAPPA_MAX, KAPPA_POINTS)
    angular_indices = configured_indices(ANGULAR_INDICES, "ANGULAR_INDICES")
    radial_indices = configured_indices(RADIAL_INDICES, "RADIAL_INDICES")

    print("=" * 104)
    print(f"EXACT RATES from the characteristic equation (fp_solve_ladder)  |  "
          f"rel_tol = {ROOT_REL_TOL:g}")
    print(f"kappa in [0, {KAPPA_MAX:g}]  ({KAPPA_POINTS} points, Pe down to {KAPPA_MAX ** -0.5:.3g})")
    print("=" * 104)
    print(f"{'problem':<12} | {'n':>3} | {'m':>3} | {'beta_nm':>9} | {'alpha_nm':>8} | "
          f"{'Lam(0)=beta^2':>14} | {'Lam(kappa_max)':>14} | {'q_nm':>8} | {'t [s]':>7}")
    print("-" * 104)

    runs: list[ModeRun] = []
    for problem in PROBLEMS:
        for n in angular_indices:
            # A zero-based column m requires m + 1 roots in the row.  This is only the
            # requested array length; the root later selected and plotted remains R_nm.
            roots_row = base_roots(problem, n, max(radial_indices) + 1)
            indices = sorted(index for index in radial_indices if index < roots_row.size)
            if problem.has_zero_mode(n):
                # This is the physical (n, m) = (0, 0) constant mode.  Its zero-based
                # index is retained everywhere else; only its undefined normalised rate
                # is omitted from these figures.
                if any(roots_row[index] == 0.0 for index in indices):
                    print("note: omitting Neumann (n, m) = (0, 0): "
                          "Lambda_nm / beta_nm^2 is 0 / 0")
                indices = [index for index in indices if roots_row[index] > 0.0]
            if not indices:
                continue

            highest = max(indices)
            row = roots_row[: highest + 1]
            alpha_row = np.array(
                [bare_alpha(n, b, QUAD_PANELS) if b > 0.0 else 2.0 for b in row]
            )
            zeros_row = bessel_zeros(n, problem.wall_condition, highest + 1)

            # One ladder solve per kappa gives the whole block at once.
            started = time.perf_counter()
            table = np.empty((kappa_grid.size, row.size))
            table[0] = row * row                       # kappa = 0 is the bare limit
            for position, kappa in enumerate(kappa_grid[1:], start=1):
                table[position] = exact_rates(problem, n, row, float(kappa), alpha_row)
            elapsed = time.perf_counter() - started

            for index in indices:
                beta = float(row[index])
                rate = table[:, index]
                print(f"{problem.name:<12} | {n:>3} | {index:>3} | {beta:>9.4f} | "
                      f"{alpha_row[index]:>8.5f} | {beta * beta:>14.4f} | {rate[-1]:>14.4f} | "
                      f"{zeros_row[index]:>8.4f} | {elapsed / len(indices):>7.2f}")
                runs.append(ModeRun(
                    problem=problem, n=n, index=index, m=index,
                    beta=beta, alpha=float(alpha_row[index]),
                    bessel_zero=float(zeros_row[index]),
                    kappa=kappa_grid, rate=rate,
                    base_roots=row, alpha_row=alpha_row,
                ))
    print("-" * 104)
    return runs


def check_enclosure(runs: list[ModeRun], *, recompute_probes: bool = True) -> None:
    """Verify eq. (rate_enclosure) and the limits, on the root-found rates."""
    print("\nSpectral checks (eq. rate_enclosure and the two limits)")
    print("-" * 104)

    worst_increase = 0.0
    ceiling_violations = 0
    for entry in runs:
        increments = np.diff(entry.rate)
        worst_increase = max(worst_increase, float(increments.max()) if increments.size else 0.0)
        if np.any(entry.rate[1:] >= entry.beta * entry.beta):
            ceiling_violations += 1
    print(f"  monotone decrease (eq. flow_negative):        largest increment = {worst_increase:+.3e}")
    print(f"  ceiling Lam < beta^2 (eq. rate_ordering):     {ceiling_violations} violation(s) / {len(runs)} modes")

    # Floor: within an angular block the rates must stay strictly ordered in m at every kappa.
    floor_violations = 0
    blocks: dict[tuple[str, int], list[ModeRun]] = {}
    for entry in runs:
        blocks.setdefault((entry.problem.name, entry.n), []).append(entry)
    for block in blocks.values():
        block = sorted(block, key=lambda e: e.index)
        for lower, upper in zip(block, block[1:]):
            if np.any(upper.rate[1:] <= lower.rate[1:]):
                floor_violations += 1
    print(f"  ordering preserved in m (eq. rate_enclosure): {floor_violations} crossing(s) over "
          f"{len(blocks)} block(s)")

    if not recompute_probes:
        print("  solver-backed limit and slope probes:        skipped (using cached plot data)")
        print("-" * 104)
        return

    print("  large-kappa limit Lam*sqrt(kappa) -> q_nm (eq. rate_large_kappa), probed at kappa = 1e6:")
    probe = 1.0e6
    for entry in runs[: min(4, len(runs))]:
        if entry.bessel_zero <= 0.0:
            continue
        measured = float(exact_rates(entry.problem, entry.n, entry.base_roots, probe,
                                     entry.alpha_row)[entry.index]) * math.sqrt(probe)
        print(f"    {entry.problem.name:<11} n={entry.n:<3} m={entry.m:<3} "
              f"{measured:11.5f} vs q = {entry.bessel_zero:11.5f}  "
              f"(rel. {abs(measured / entry.bessel_zero - 1.0):.2e})")

    print("  initial slope dLam/dkappa|_0 vs -alpha_nm beta_nm^4 (eq. flow_equation at kappa = 0):")
    for entry in runs[: min(4, len(runs))]:
        measured = FlowRightHandSide(entry.n).derivative(0.0, entry.beta * entry.beta)
        predicted = -entry.alpha * entry.beta**4
        print(f"    {entry.problem.name:<11} n={entry.n:<3} m={entry.m:<3} "
              f"{measured:+.6e} vs {predicted:+.6e}  (rel. {abs(measured / predicted - 1.0):.2e})")
    print("-" * 104)


def flow_versus_finite_differences(entry: ModeRun, kappas: np.ndarray) -> np.ndarray:
    """|FD/flow - 1|: eq. (flow_equation) against central differences of the exact rates.

    Both sides are evaluated ON the branch -- the rate at each kappa comes from the
    root-finder -- which is the only regime in which the mode-integral form of the
    right-hand side is meaningful.
    """
    rhs = FlowRightHandSide(entry.n)
    gaps = np.empty(kappas.size)
    for position, kappa in enumerate(kappas):
        step = max(1e-8, 1e-4 * float(kappa))
        below = exact_rates(entry.problem, entry.n, entry.base_roots,
                            float(kappa) - step, entry.alpha_row)[entry.index]
        above = exact_rates(entry.problem, entry.n, entry.base_roots,
                            float(kappa) + step, entry.alpha_row)[entry.index]
        difference = (above - below) / (2.0 * step)
        analytic = rhs.derivative(float(kappa), 0.5 * (above + below))
        gaps[position] = abs(difference / analytic - 1.0)
    return gaps


def compute_plot_data() -> PlotData:
    """Compute all numerical values required by the selected figures."""
    runs = run()
    validation_entries = runs[:MAX_CHECK_CURVES]
    if "validation" in FIGURES and RUN_FLOW_CHECK and validation_entries:
        validation_kappas = np.logspace(
            math.log10(max(KAPPA_MIN, 1e-3)), math.log10(KAPPA_MAX), CHECK_POINTS,
        )
        validation_gaps = np.vstack([
            flow_versus_finite_differences(entry, validation_kappas)
            for entry in validation_entries
        ])
    else:
        validation_kappas = np.empty(0, dtype=float)
        validation_gaps = np.empty((0, 0), dtype=float)
    return PlotData(runs, validation_kappas, validation_gaps)


def save_plot_data(data: PlotData) -> None:
    """Export the plot data to one compressed, non-pickled NumPy archive."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    max_row_size = max(entry.base_roots.size for entry in data.runs)
    base_roots = np.full((len(data.runs), max_row_size), np.nan)
    alpha_rows = np.full_like(base_roots, np.nan)
    row_sizes = np.empty(len(data.runs), dtype=int)
    for position, entry in enumerate(data.runs):
        row_size = entry.base_roots.size
        row_sizes[position] = row_size
        base_roots[position, :row_size] = entry.base_roots
        alpha_rows[position, :row_size] = entry.alpha_row

    np.savez_compressed(
        DATA_FILE,
        configuration=np.array(json.dumps(data_configuration(), sort_keys=True)),
        problem=np.array([entry.problem.name for entry in data.runs]),
        n=np.array([entry.n for entry in data.runs], dtype=int),
        index=np.array([entry.index for entry in data.runs], dtype=int),
        m=np.array([entry.m for entry in data.runs], dtype=int),
        beta=np.array([entry.beta for entry in data.runs]),
        alpha=np.array([entry.alpha for entry in data.runs]),
        bessel_zero=np.array([entry.bessel_zero for entry in data.runs]),
        kappa=np.stack([entry.kappa for entry in data.runs]),
        rate=np.stack([entry.rate for entry in data.runs]),
        base_roots=base_roots,
        alpha_row=alpha_rows,
        row_size=row_sizes,
        validation_kappas=data.validation_kappas,
        validation_gaps=data.validation_gaps,
    )
    print(f"  saved data {DATA_FILE}")


def load_plot_data() -> PlotData:
    """Load cached plot data and verify that it matches the current run settings."""
    with np.load(DATA_FILE, allow_pickle=False) as archive:
        stored_configuration = json.loads(str(archive["configuration"].item()))
        if stored_configuration != data_configuration():
            raise ValueError("the stored numerical configuration does not match this script")

        problem_by_name = {problem.name: problem for problem in PROBLEMS}
        problem_names = archive["problem"].astype(str)
        kappas = archive["kappa"]
        rates = archive["rate"]
        base_roots = archive["base_roots"]
        alpha_rows = archive["alpha_row"]
        row_sizes = archive["row_size"]
        runs = []
        for position, problem_name in enumerate(problem_names):
            row_size = int(row_sizes[position])
            runs.append(ModeRun(
                problem=problem_by_name[problem_name],
                n=int(archive["n"][position]),
                index=int(archive["index"][position]),
                m=int(archive["m"][position]),
                beta=float(archive["beta"][position]),
                alpha=float(archive["alpha"][position]),
                bessel_zero=float(archive["bessel_zero"][position]),
                kappa=np.array(kappas[position], copy=True),
                rate=np.array(rates[position], copy=True),
                base_roots=np.array(base_roots[position, :row_size], copy=True),
                alpha_row=np.array(alpha_rows[position, :row_size], copy=True),
            ))
        data = PlotData(
            runs=runs,
            validation_kappas=np.array(archive["validation_kappas"], copy=True),
            validation_gaps=np.array(archive["validation_gaps"], copy=True),
        )
    print(f"  loaded data {DATA_FILE}")
    return data


def get_plot_data() -> tuple[PlotData, bool]:
    """Load a compatible cache when allowed, otherwise compute and export fresh data."""
    if not FORCE_RECOMPUTE:
        try:
            return load_plot_data(), True
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
            print(f"  could not reuse {DATA_FILE}: {error}")
            print("  recomputing the plot data instead")

    data = compute_plot_data()
    if data.runs:
        save_plot_data(data)
    return data, False


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(figure, stem: str) -> None:
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = FIGURES_DIRECTORY / f"{stem}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")


def mode_styles(runs: list[ModeRun], points: int = 0, markers_per_curve: int = 9) -> dict[int, dict]:
    """Encode angular families by colour and radial branches by dash pattern.

    Markers distinguish Neumann from Dirichlet only where both wall conditions share an
    axis.  ``points`` staggers the marker positions so they do not pile up in columns.
    """
    angular_indices = configured_indices(ANGULAR_INDICES, "ANGULAR_INDICES")
    radial_indices = configured_indices(RADIAL_INDICES, "RADIAL_INDICES")
    if len(angular_indices) > len(OKABE_ITO):
        raise ValueError(
            f"ANGULAR_INDICES has {len(angular_indices)} entries, but only "
            f"{len(OKABE_ITO)} family colours are defined"
        )
    if len(radial_indices) > len(RADIAL_LINESTYLES):
        raise ValueError(
            f"RADIAL_INDICES has {len(radial_indices)} entries, but only "
            f"{len(RADIAL_LINESTYLES)} radial line styles are defined"
        )

    colour_by_n = dict(zip(angular_indices, OKABE_ITO))
    linestyle_by_m = dict(zip(radial_indices, RADIAL_LINESTYLES))
    styles = {}
    for position, entry in enumerate(runs):
        style = {
            "color": colour_by_n[entry.n],
            "linestyle": linestyle_by_m[entry.m],
            "marker": WALL_MARKERS[entry.problem.name],
        }
        if points > 0:
            step = max(1, points // markers_per_curve)
            style["markevery"] = (position % step, step)
        styles[id(entry)] = style
    return styles


def apply_log_axis_style(axis, *, log_y: bool = False) -> None:
    """Apply the shared heavy-frame, mirrored-tick treatment to logarithmic axes."""
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(axis="x", which="major", direction="in", top=True, bottom=True,
                     width=1.6, length=8)
    axis.tick_params(axis="x", which="minor", direction="in", top=True, bottom=True,
                     width=1.1, length=4.5)
    axis.tick_params(axis="y", which="major", direction="in", left=True, right=True,
                     width=1.6, length=8)
    if log_y:
        axis.tick_params(axis="y", which="minor", direction="in", left=True, right=True,
                         width=1.1, length=4.5)


def add_encoding_legend(figure, runs: list[ModeRun], *, show_wall_markers: bool) -> None:
    """Add a compact encoding legend below the plotting area."""
    from matplotlib.lines import Line2D

    angular_indices = configured_indices(ANGULAR_INDICES, "ANGULAR_INDICES")
    radial_indices = configured_indices(RADIAL_INDICES, "RADIAL_INDICES")
    if len(angular_indices) > len(OKABE_ITO):
        raise ValueError(
            f"ANGULAR_INDICES has {len(angular_indices)} entries, but only "
            f"{len(OKABE_ITO)} family colours are defined"
        )
    colour_by_n = dict(zip(angular_indices, OKABE_ITO))
    linestyle_by_m = dict(zip(radial_indices, RADIAL_LINESTYLES))
    shown_n = [n for n in angular_indices if any(entry.n == n for entry in runs)]
    shown_m = [m for m in radial_indices if any(entry.m == m for entry in runs)]
    colour_handles = [
        Line2D([0], [0], color=colour_by_n[n], linewidth=2.4, label=rf"$n = {n}$")
        for n in shown_n
    ]
    linestyle_handles = [
        Line2D([0], [0], color="#303030", linestyle=linestyle_by_m[m], linewidth=2.4,
               label=rf"$m = {m}$")
        for m in shown_m
    ]

    handles = colour_handles + linestyle_handles
    if show_wall_markers:
        shown_problems = {entry.problem.name for entry in runs}
        handles += [
            Line2D([0], [0], color="#303030", linestyle="None", marker=WALL_MARKERS[problem.name],
                   markerfacecolor="none", markersize=8, markeredgewidth=1.3,
                   label=problem.short)
            for problem in PROBLEMS if problem.name in shown_problems
        ]

    columns = min(7, max(1, math.ceil(len(handles) / 2)))
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=columns,
        framealpha=0.92, fontsize=LEGEND_LABEL_FONT_SIZE,
        handlelength=2.4, labelspacing=0.55, columnspacing=1.1, borderpad=0.55,
    )


def add_rate_inset(axis, entries: list[ModeRun], styles: dict[int, dict]) -> None:
    """Add a true low-kappa magnification of one normalised-rate panel."""
    if not KAPPA_MIN < KAPPA_MID < KAPPA_MAX:
        raise ValueError(
            "the rate inset requires KAPPA_MIN < KAPPA_MID < KAPPA_MAX; "
            f"got {KAPPA_MIN:g}, {KAPPA_MID:g}, {KAPPA_MAX:g}"
        )

    # Axes-relative placement keeps the inset in the data-free lower-left region when
    # the outer figure size or manuscript column width changes.
    inset = axis.inset_axes((0.055, 0.145, 0.44, 0.40), zorder=5)
    for entry in entries:
        kappa = entry.kappa[1:]
        visible = (kappa >= KAPPA_MIN) & (kappa <= KAPPA_MID)
        if not np.any(visible):
            continue
        style = styles[id(entry)]
        inset.plot(
            kappa[visible], entry.rate[1:][visible] / entry.beta**2,
            color=style["color"], linestyle=style["linestyle"], linewidth=1.25,
        )

    # Linear in the ratio: over this range the rates stay within a fraction of a percent
    # of their bare values, which a logarithmic axis would flatten into one line.
    inset.set_xscale("log")
    inset.set_xlim(KAPPA_MIN, KAPPA_MID)
    inset.margins(y=0.08)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.97)
    inset.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.35)
    inset.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.18)
    for spine in inset.spines.values():
        spine.set_linewidth(1.15)
    inset.tick_params(which="major", direction="in", top=True, right=True,
                      width=1.0, length=4.0, labelsize=INSET_TICKS_FONTSIZE)
    inset.tick_params(which="minor", direction="in", top=True, right=True,
                      width=0.7, length=2.3)
    # The labels go on the right, where the panel has no curves.
    inset.tick_params(axis="y", which="both", labelleft=False, labelright=True)
    inset.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.001))
    inset.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    inset.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # A restrained dashed callout guides the eye from the low-kappa region to the inset.
    # Matplotlib selects the two unobtrusive connector corners automatically.
    _, connectors = axis.indicate_inset_zoom(
        inset, edgecolor="#303030", linewidth=0.9, linestyle=(0, (5, 4)),
        alpha=0.52, zorder=4,
    )
    # Matplotlib 3.6 forwards ``linestyle`` to the rectangle but not to the four
    # ConnectionPatch artists returned alongside it.
    for connector in connectors:
        connector.set_linestyle((0, (5, 4)))


def figure_rates(runs: list[ModeRun]) -> None:
    """Normalised rates, split by wall condition and grouped by angular family."""
    import matplotlib.pyplot as plt

    styles = mode_styles(runs)
    figure, axes = plt.subplots(
        2, 1, figsize=(20.0, 13.75), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    axes_by_problem = {problem.name: axis for problem, axis in zip(PROBLEMS, axes)}
    for entry in runs:
        style = styles[id(entry)]
        kappa = entry.kappa[1:]
        axis = axes_by_problem[entry.problem.name]
        axis.plot(
            kappa, entry.rate[1:] / entry.beta**2,
            color=style["color"], linestyle=style["linestyle"],
        )
        if SHOW_RATE_GUIDES:
            if entry.bessel_zero > 0.0:
                asymptote = entry.bessel_zero / (entry.beta**2 * np.sqrt(kappa))
                visible = asymptote < 1.0
                axis.plot(kappa[visible], asymptote[visible], color=style["color"],
                          linewidth=1.4, alpha=0.32, linestyle=style["linestyle"])
    for axis in axes:
        if SHOW_RATE_GUIDES:
            # All normalised branches share the same bare limit at kappa = 0.
            axis.axhline(1.0, color="#202020", linewidth=0.9, alpha=0.5,
                         linestyle=(0, (1, 2)))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(
            r"$\Lambda_{nm}(\kappa)\,/\,\beta_{nm}^2$",
            fontsize=AXIS_LABEL_FONTSIZE,
            labelpad=Y_AXIS_LABEL_PAD,
        )
        axis.tick_params(axis="y", which="both", labelsize=TICKS_FONTSIZE)
        axis.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.4)
        axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
        apply_log_axis_style(axis, log_y=True)
    for problem, axis in zip(PROBLEMS, axes):
        axis.text(
            1.0, 1.0, problem.short, transform=axis.transAxes,
            rotation=270, rotation_mode="anchor", ha="left", va="bottom",
            fontsize=PANEL_TITLE_FONT_SIZE, clip_on=False,
        )
        add_rate_inset(
            axis,
            [entry for entry in runs if entry.problem.name == problem.name],
            styles,
        )
    axes[-1].set_xlabel(r"$\kappa = \mathrm{Pe}^{-2}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[-1].tick_params(axis="x", which="both", labelsize=TICKS_FONTSIZE)

    add_encoding_legend(figure, runs, show_wall_markers=False)
    figure.subplots_adjust(left=0.10, right=0.96, top=0.95, bottom=0.27)
    save(figure, "Figure_1b")
    plt.close(figure)


def figure_similarity_residual(runs: list[ModeRun]) -> None:
    """Seed residual in its own wide, rectangular figure."""
    import matplotlib.pyplot as plt

    points = runs[0].kappa.size - 1
    styles = mode_styles(runs, points=points)
    figure, axis = plt.subplots(figsize=(20.0, 7.5))
    for entry in runs:
        style = styles[id(entry)]
        kappa = entry.kappa[1:]
        seed = np.array([seed_rate(entry.beta**2, float(k), entry.alpha) for k in kappa])
        axis.plot(
            kappa, entry.rate[1:] / seed - 1.0,
            color=style["color"], linestyle=style["linestyle"], linewidth=1.8,
            marker=style["marker"], markevery=style["markevery"], markerfacecolor="none",
            markersize=10, markeredgewidth=1.3,
        )
    axis.axhline(0.0, color="#202020", linewidth=1.0, alpha=0.6)
    axis.set_xscale("log")
    axis.set_xlabel(r"$\kappa = \mathrm{Pe}^{-2}$")
    axis.set_ylabel(r"$\Lambda_{nm}(\kappa)/\Lambda^{\mathrm{seed}}_{nm} - 1$", fontsize=24, labelpad=Y_AXIS_LABEL_PAD)
    axis.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.4)
    apply_log_axis_style(axis)
    add_encoding_legend(figure, runs, show_wall_markers=True)
    figure.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.36)
    save(figure, "flow_equation_similarity_residual")
    plt.close(figure)


def figure_similarity_ratio(runs: list[ModeRun]) -> None:
    """Lam_nm(kappa) / beta_nm^2 vs kappa, one curve per mode."""
    import matplotlib.pyplot as plt

    points = runs[0].kappa.size - 1
    styles = mode_styles(runs, points=points)
    figure, axis = plt.subplots(figsize=(17.5, 9.375))
    for entry in runs:
        style = styles[id(entry)]
        axis.plot(
            entry.kappa[1:], entry.rate[1:] / entry.beta**2,
            color=style["color"], linestyle=style["linestyle"], linewidth=1.8,
            marker=style["marker"], markevery=style["markevery"], markerfacecolor="none",
            markersize=7, markeredgewidth=1.3,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\kappa = \mathrm{Pe}^{-2}$")
    axis.set_ylabel(r"$\Lambda_{nm}(\kappa)\,/\,\beta_{nm}^2$", labelpad=Y_AXIS_LABEL_PAD)
    axis.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.4)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
    add_encoding_legend(figure, runs, show_wall_markers=True)
    figure.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.35)
    save(figure, "flow_equation_similarity_ratio")
    plt.close(figure)


def figure_validation(
    runs: list[ModeRun], validation_kappas: np.ndarray, validation_gaps: np.ndarray,
) -> None:
    """The flow equation checked pointwise against central differences of the exact rates.

    This is the verification the manuscript reports. It never integrates eq.
    (flow_equation); it evaluates both sides on the branch and compares. The floor of the
    curves is the finite-difference truncation/round-off floor, not the flow equation's.
    """
    import matplotlib.pyplot as plt

    shown = runs[:MAX_CHECK_CURVES]
    styles = mode_styles(shown)

    if validation_gaps.shape != (len(shown), validation_kappas.size):
        raise ValueError("cached validation data has an unexpected shape")

    figure, axis = plt.subplots(figsize=(17.5, 9.375))
    for entry, gaps in zip(shown, validation_gaps):
        style = styles[id(entry)]
        axis.plot(
            validation_kappas, np.maximum(gaps, 1e-17),
            color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
            markersize=6, markerfacecolor="none", markeredgewidth=1.3, linewidth=1.6,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\kappa = \mathrm{Pe}^{-2}$")
    axis.set_ylabel(r"$\left|\mathrm{FD}/\dot\Lambda_{\mathrm{flow}} - 1\right|$", labelpad=Y_AXIS_LABEL_PAD)
    axis.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.4)
    add_encoding_legend(figure, shown, show_wall_markers=True)
    figure.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.35)
    save(figure, "flow_equation_validation")
    plt.close(figure)


# ---------------------------------------------------------------------------
def main() -> None:
    matplotlib.rcParams.update({
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": LEGEND_LABEL_FONT_SIZE,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })

    data, loaded_from_cache = get_plot_data()
    runs = data.runs
    if not runs:
        raise SystemExit("no modes selected (check ANGULAR_INDICES / RADIAL_INDICES)")
    check_enclosure(runs, recompute_probes=not loaded_from_cache)

    print("\nFigures")
    print("-" * 104)
    if "rates" in FIGURES:
        figure_rates(runs)
    if "similarity_ratio" in FIGURES:
        figure_similarity_ratio(runs)
    if "similarity_residual" in FIGURES:
        figure_similarity_residual(runs)
    if "validation" in FIGURES and RUN_FLOW_CHECK:
        figure_validation(runs, data.validation_kappas, data.validation_gaps)
    print("-" * 104)


if __name__ == "__main__":
    main()
