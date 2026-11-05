"""Exact finite-Peclet decay rates against the bare-basis matrix method of Neuhauser et al.

Neuhauser et al. (Int. J. Heat Mass Transfer 249 (2025) 127198, Sec. 2.4) expand the
finite-Peclet solution of each angular block in the N bare (Pe = inf) eigenfunctions and
obtain the decay rates as eigenvalues of a quadratic matrix problem. With the bare modes
R_m normalised in L2_omega, projecting the radial equation onto them gives

    (kappa Lam^2 U + Lam I - B) v = 0,   U_km = int_0^1 R_k R_m r dr,   B = diag(beta_m^2),

and its N non-negative eigenvalues Lam^(N)_nm approximate the decaying rates. The rates
Lam_nm of the article are the roots of the characteristic equations (char_dirichlet_fp,
char_neumann_fp), found by the compiled ladder solver. The figure shows
(Lam^(N)_nm - Lam_nm) / Lam_nm against N for each Peclet number and wall condition.

The matrix problem is assembled here, following Sec. 2.4 of that paper; it is not their
code. Each eigenvalue is re-evaluated as the positive root of the Rayleigh functional of
its eigenvector, which is the same Ritz value with less round-off. The constant mode of
the zero-flux n = 0 block (Lam = 0 in both methods) is kept in the basis and not plotted.

Outputs, at the root of the repository (Fig. 10 of the article):
    data_files/qep_rates_convergence_data.npz   bare roots, exact rates and matrix rates
                                                for every N; reused while its stored
                                                configuration matches.
    figures/Figure_10.pdf, .png                  the figure.

All run parameters are module-level constants below. Run with
    python3 qep_rates_convergence.py
The default configuration (N up to 300) takes about ten minutes; restyling the figure
with FORCE_RECOMPUTE = False only reloads the data file.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import scipy.linalg as sla

import MassDiffusion3Dcpp as md


DEFS = md.definitions
# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
ROOT_DIRECTORY = REPOSITORY
FIGURES_DIRECTORY = REPOSITORY / "figures"
DATA_DIRECTORY = REPOSITORY / "data_files"
DATA_FILE = DATA_DIRECTORY / "qep_rates_convergence_data.npz"
FIGURE_STEM = "Figure_10"

# ---------------------------------------------------------------------------
# Run configuration. Edit directly.
# ---------------------------------------------------------------------------
FORCE_RECOMPUTE = False
PE_VALUES = (1.0, 10.0, 100.0)
# One figure row per entry: (wall condition, angular index n).
FAMILIES = (("Neumann", 1), ("Dirichlet", 0))
MAX_BASIS_SIZE = 300
BASIS_SIZE_POINTS = 36
QUADRATURE_POINTS_PER_MODE = 10
# The double-precision bare mode loses accuracy for beta above ~700 (ln DBL_MAX = 709).
LONG_DOUBLE_ABOVE_BETA = 600.0
ROOT_REL_TOL = 1e-15
ROOT_MAX_ITER = 300
ROOT_TABLES = {
    "Neumann": "src/CDStratifiedSolution/data/roots_400x400.txt",
    "Dirichlet": "src/CDGraetzIsothermalSolution/data/graetz_roots_1000.txt",
}

# Presentation only: changing these never invalidates the data file.
RADIAL_INDICES = (0, 1, 4, 19, 49, 99)
ROUNDOFF_FLOOR = 1e-13
Y_LIMITS = (1e-14, 3.0)
X_TICKS = (2, 5, 10, 30, 100, 300)
# Each row's slope guide uses the exponent fitted to the curve of radial index
# GUIDE_RADIAL_INDEX in the panel where it is drawn (first Peclet number).
GUIDE_RADIAL_INDEX = 0
SLOPE_FIT_MIN_N = 100
GUIDE_RANGE = (10.0, 80.0)
GUIDE_OFFSET = 1e-2
ROW_LABELS = {"Neumann": "zero-flux wall", "Dirichlet": "isothermal wall"}
USE_TEX = True
# Printed size: the article's text width is 160 mm, so the sizes below are printed points.
FIGURE_WIDTH_MM = 160.0
FIGURE_HEIGHT_MM = 115.0
LABEL_FONT_SIZE = 10.5
TICK_FONT_SIZE = 9.5
LEGEND_FONT_SIZE = 9.5
CURVE_LINE_WIDTH = 1.2
GUIDE_LINE_WIDTH = 0.8
AXES_LINE_WIDTH = 0.8
GRID_LINE_WIDTH = 0.5
CURVE_COLORMAP = "Blues"
CURVE_COLOR_RANGE = (1.0, 0.45)
# Same dash sequence as the rate figure (solve_flow_equation.py), assigned by position.
RADIAL_LINESTYLES = (
    "-",
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 5)),
    (0, (1, 5)),
    (0, (3, 5, 1, 5)),
)
# Sub- and superscripts at 9 pt keep every printed capital and numeral at least 2 mm high.
TEX_PREAMBLE = "".join(rf"\DeclareMathSizes{{{size:g}}}{{{size:g}}}{{9}}{{9}}"
                       for size in (9, 9.5, 10, 10.5, 11, 12))

matplotlib.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": LABEL_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": AXES_LINE_WIDTH,
    "text.latex.preamble": TEX_PREAMBLE,
})


def wall_condition(wall: str):
    return {"Neumann": md.WallCondition.Neumann, "Dirichlet": md.WallCondition.Dirichlet}[wall]


def has_zero_mode(wall: str, n: int) -> bool:
    return wall == "Neumann" and n == 0


def basis_sizes() -> np.ndarray:
    return np.unique(np.round(np.geomspace(2, MAX_BASIS_SIZE, BASIS_SIZE_POINTS)).astype(int))


def data_configuration() -> dict:
    return {
        "format_version": 1,
        "pe_values": [float(pe) for pe in PE_VALUES],
        "families": [[wall, int(n)] for wall, n in FAMILIES],
        "basis_sizes": basis_sizes().tolist(),
        "quadrature_points_per_mode": QUADRATURE_POINTS_PER_MODE,
        "long_double_above_beta": LONG_DOUBLE_ABOVE_BETA,
        "root_rel_tol": ROOT_REL_TOL,
        "root_max_iter": ROOT_MAX_ITER,
        "root_tables": {wall: Path(name).name for wall, name in ROOT_TABLES.items()},
    }


def solve_base_roots(wall: str, n: int, count: int, scan: float = 0.25) -> np.ndarray:
    """Bare roots from the src/ kernel, for blocks the shipped tables do not cover."""
    condition = wall_condition(wall)
    roots = [0.0] if has_zero_mode(wall, n) else []
    beta = scan if roots else 1e-6
    previous = DEFS.fp_char_ld(n, beta * beta, 0.0, condition)
    while len(roots) < count:
        upper = beta + scan
        current = DEFS.fp_char_ld(n, upper * upper, 0.0, condition)
        if (previous > 0.0) != (current > 0.0):
            root, _ = DEFS.fp_solve_bracketed_ld(n, 0.0, condition, beta * beta, upper * upper,
                                                 ROOT_REL_TOL, ROOT_MAX_ITER)
            roots.append(math.sqrt(root))
        beta, previous = upper, current
    return np.array(roots)


def base_roots(wall: str, n: int, count: int) -> np.ndarray:
    table = np.loadtxt(ROOT_DIRECTORY / ROOT_TABLES[wall])
    row = None
    if table.ndim == 1 and n == 0:
        row = table
    elif table.ndim == 2 and n < table.shape[0]:
        row = table[n]
    if row is not None and row.size >= count:
        return np.array(row[:count], dtype=float)
    return solve_base_roots(wall, n, count)


def bare_modes(n: int, beta: np.ndarray, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Bare radial modes on the quadrature nodes, normalised in L2_omega."""
    modes = np.empty((beta.size, r.size))
    for row, b in enumerate(beta):
        if b == 0.0:
            modes[row] = 1.0
            continue
        kernel = DEFS.psinm_r_ld if b > LONG_DOUBLE_ABOVE_BETA else DEFS.psinm_r
        modes[row] = [kernel(n, float(b), float(x)) for x in r]
    modes /= np.sqrt((modes ** 2 * w * r * (1.0 - r * r)).sum(axis=1))[:, None]
    return modes


def matrix_rates(overlap: np.ndarray, beta2: np.ndarray, kappa: float) -> tuple[np.ndarray, float]:
    """Non-negative eigenvalues of (kappa Lam^2 U + Lam I - B) v = 0, ascending."""
    size = beta2.size
    zero, identity = np.zeros((size, size)), np.eye(size)
    stiffness = np.diag(beta2)
    values, vectors = sla.eig(np.block([[zero, identity], [stiffness, -identity]]),
                              np.block([[identity, zero], [zero, kappa * overlap]]))
    decaying = np.argsort(-values.real)[:size]
    values, vectors = values[decaying], vectors[:size, decaying]
    imaginary = float(np.max(np.abs(values.imag) / np.maximum(np.abs(values), 1.0)))
    a = kappa * np.einsum("im,ij,jm->m", vectors.conj(), overlap, vectors).real
    b = np.einsum("im,im->m", vectors.conj(), vectors).real
    c = np.einsum("im,i,im->m", vectors.conj(), beta2, vectors).real
    rates = 2.0 * c / (b + np.sqrt(b * b + 4.0 * a * c))
    return np.sort(rates), imaginary


def compute_data() -> dict:
    sizes = basis_sizes()
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_POINTS_PER_MODE * MAX_BASIS_SIZE)
    r, w = 0.5 * (nodes + 1.0), 0.5 * weights
    beta_all = np.full((len(FAMILIES), MAX_BASIS_SIZE), np.nan)
    exact = np.full((len(FAMILIES), len(PE_VALUES), MAX_BASIS_SIZE), np.nan)
    ritz = np.full((len(FAMILIES), len(PE_VALUES), sizes.size, MAX_BASIS_SIZE), np.nan)
    worst_imaginary = 0.0
    started = time.perf_counter()
    for f, (wall, n) in enumerate(FAMILIES):
        beta = base_roots(wall, n, MAX_BASIS_SIZE)
        beta_all[f] = beta
        modes = bare_modes(n, beta, r, w)
        overlap = (modes * w * r) @ modes.T
        print(f"  {wall} n={n}: {MAX_BASIS_SIZE} bare modes on {r.size} nodes, "
              f"beta up to {beta[-1]:.1f} ({time.perf_counter() - started:.0f}s)", flush=True)
        for p, pe in enumerate(PE_VALUES):
            kappa = 1.0 / pe ** 2
            roots, _ = DEFS.fp_solve_ladder_ld(n, list(beta ** 2), kappa, wall_condition(wall),
                                               has_zero_mode(wall, n), ROOT_REL_TOL, ROOT_MAX_ITER)
            exact[f, p] = roots
            for i, size in enumerate(sizes):
                rates, imaginary = matrix_rates(overlap[:size, :size], beta[:size] ** 2, kappa)
                ritz[f, p, i, :size] = rates
                worst_imaginary = max(worst_imaginary, imaginary)
            print(f"    Pe={pe:g} done ({time.perf_counter() - started:.0f}s)", flush=True)
    print(f"  largest relative imaginary part of a decaying eigenvalue: {worst_imaginary:.1e}")
    return {"basis_sizes": sizes, "beta": beta_all, "exact": exact, "ritz": ritz}


def save_data(data: dict) -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_FILE,
                        configuration=np.array(json.dumps(data_configuration(), sort_keys=True)),
                        **data)
    print(f"  saved data {DATA_FILE.name}")


def load_data() -> dict:
    with np.load(DATA_FILE, allow_pickle=False) as archive:
        if json.loads(str(archive["configuration"].item())) != data_configuration():
            raise ValueError("the stored configuration does not match this script")
        data = {key: np.array(archive[key]) for key in ("basis_sizes", "beta", "exact", "ritz")}
    print(f"  loaded data {DATA_FILE.name}")
    return data


def get_data() -> dict:
    if not FORCE_RECOMPUTE:
        try:
            return load_data()
        except FileNotFoundError:
            print(f"  no {DATA_FILE.name} yet; computing")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"  could not reuse {DATA_FILE.name} ({type(error).__name__}: {error}); recomputing")
    data = compute_data()
    save_data(data)
    return data


def relative_difference(data: dict, f: int, p: int) -> np.ndarray:
    exact = data["exact"][f, p]
    with np.errstate(invalid="ignore", divide="ignore"):
        difference = (data["ritz"][f, p] - exact) / exact
    difference[:, exact == 0.0] = np.nan
    return difference


def fitted_slope(sizes: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of log(values) against log(N) over N >= SLOPE_FIT_MIN_N."""
    keep = (sizes >= SLOPE_FIT_MIN_N) & np.isfinite(values) & (values > 10 * ROUNDOFF_FLOOR)
    if keep.sum() < 3:
        return math.nan
    return float(np.polyfit(np.log(sizes[keep]), np.log(values[keep]), 1)[0])


def report(data: dict) -> None:
    sizes = data["basis_sizes"]
    for f, (wall, n) in enumerate(FAMILIES):
        for p, pe in enumerate(PE_VALUES):
            difference = relative_difference(data, f, p)
            below = np.nanmin(difference)
            slopes = []
            for m in RADIAL_INDICES[:3]:
                slope = fitted_slope(sizes, difference[:, m])
                if math.isfinite(slope):
                    slopes.append(f"m={m}: {slope:.3f}")
            inaccurate = np.nan_to_num(difference[-1, :sizes[-1]], nan=0.0) >= 1e-3
            accurate = int(np.argmax(inaccurate)) if inaccurate.any() else int(sizes[-1])
            print(f"  {wall} n={n}, Pe={pe:g}: slopes over N >= {SLOPE_FIT_MIN_N} "
                  f"({', '.join(slopes)}); with N={sizes[-1]}, the first {accurate} rates are "
                  f"within 1e-3; smallest difference {below:.1e}")
            if below < -10 * ROUNDOFF_FLOOR:
                print("    WARNING: a matrix rate lies below the exact rate beyond round-off")


def logarithmic_guide(ax, sizes, curves, exponent: float, label: str) -> None:
    lower, upper = GUIDE_RANGE
    anchors = [np.exp(np.interp(np.log(lower), np.log(sizes[np.isfinite(c)]),
                                np.log(c[np.isfinite(c)])))
               for c in curves if np.sum(np.isfinite(c)) >= 2 and np.nanmin(sizes[np.isfinite(c)]) <= lower]
    if not anchors:
        return
    start = min(anchors) * GUIDE_OFFSET
    end = start * (upper / lower) ** (-exponent)
    if end < Y_LIMITS[0] * 10:
        start *= Y_LIMITS[0] * 10 / end
        end = Y_LIMITS[0] * 10
    ax.plot((lower, upper), (start, end), color="#505050", linewidth=GUIDE_LINE_WIDTH)
    middle = math.sqrt(lower * upper)
    ax.text(middle, math.sqrt(start * end) / 4.0, label, fontsize=TICK_FONT_SIZE,
            color="#303030", ha="right", va="top")


def style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.8, MAX_BASIS_SIZE * 1.12)
    ax.set_ylim(*Y_LIMITS)
    ax.xaxis.set_major_locator(FixedLocator([t for t in X_TICKS if t <= MAX_BASIS_SIZE]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    decades = np.arange(math.ceil(math.log10(Y_LIMITS[0])), math.floor(math.log10(Y_LIMITS[1])) + 1)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** decades[(decades % 3) == 0]))
    ax.yaxis.set_minor_locator(FixedLocator(10.0 ** decades))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.4)
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   width=AXES_LINE_WIDTH, length=4)
    ax.tick_params(which="minor", direction="in", top=True, right=True, width=0.6, length=2)


def plot(data: dict) -> None:
    mm = 1.0 / 25.4
    sizes = data["basis_sizes"]
    colors = plt.get_cmap(CURVE_COLORMAP)(np.linspace(*CURVE_COLOR_RANGE, len(RADIAL_INDICES)))
    fig, axes = plt.subplots(len(FAMILIES), len(PE_VALUES), sharex=True, sharey=True,
                             figsize=(FIGURE_WIDTH_MM * mm, FIGURE_HEIGHT_MM * mm), squeeze=False)
    for f, (wall, n) in enumerate(FAMILIES):
        for p, pe in enumerate(PE_VALUES):
            ax = axes[f, p]
            style_axis(ax)
            difference = relative_difference(data, f, p)
            drawn = []
            for k, m in enumerate(RADIAL_INDICES):
                if m >= difference.shape[1]:
                    continue
                curve = np.where(difference[:, m] > ROUNDOFF_FLOOR, difference[:, m], np.nan)
                if np.sum(np.isfinite(curve)) < 2:
                    continue
                ax.plot(sizes, curve, color=colors[k], linestyle=RADIAL_LINESTYLES[k],
                        linewidth=CURVE_LINE_WIDTH)
                drawn.append(curve)
            if p == 0:
                exponent = -fitted_slope(sizes, difference[:, GUIDE_RADIAL_INDEX])
                if math.isfinite(exponent):
                    logarithmic_guide(ax, sizes, drawn, exponent,
                                      rf"$\propto N^{{-{exponent:.3f}}}$")
            if f == 0:
                ax.set_title(rf"$\mathrm{{Pe}} = {pe:g}$", fontsize=LABEL_FONT_SIZE)
        axes[f, -1].text(1.07, 0.5, rf"{ROW_LABELS[wall]}, $n = {n}$",
                         transform=axes[f, -1].transAxes, rotation=270,
                         ha="center", va="center", fontsize=LABEL_FONT_SIZE)
    for ax in axes[-1]:
        ax.set_xlabel(r"bare modes, $N$")
    fig.supylabel(r"$(\Lambda^{(N)}_{nm}-\Lambda_{nm})\,/\,\Lambda_{nm}$", x=0.012,
                  fontsize=LABEL_FONT_SIZE)
    handles = [Line2D([], [], color=colors[k], linestyle=RADIAL_LINESTYLES[k],
                      linewidth=CURVE_LINE_WIDTH, label=rf"$m = {m}$")
               for k, m in enumerate(RADIAL_INDICES)]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="lower center",
               bbox_to_anchor=(0.52, 0.0), handlelength=2.4, columnspacing=1.1)
    fig.subplots_adjust(left=0.115, right=0.925, top=0.94, bottom=0.185, wspace=0.13, hspace=0.08)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES_DIRECTORY / f"{FIGURE_STEM}.{extension}", dpi=300)
        print(f"  saved {FIGURE_STEM}.{extension}")
    plt.close(fig)


def main() -> None:
    if max(RADIAL_INDICES) >= MAX_BASIS_SIZE:
        print("  note: radial indices at or above MAX_BASIS_SIZE are not drawn")
    data = get_data()
    report(data)
    plot(data)


if __name__ == "__main__":
    main()
