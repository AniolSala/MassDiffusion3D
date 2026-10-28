"""Generate local isothermal Graetz Nusselt-number curves and its comparison tables.

Besides the curves of Fig. 3, the script writes the fully developed Nusselt numbers and
the comparison of the bare series with the classical values of Shah and London (Table S2).

Edit the configuration block below, then run:
    python3 nusselt_curves.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

import MassDiffusion3Dcpp as md

# ---------------------------------------------------------------------------
# Run configuration. Edit these values directly; there are no CLI arguments.
# ---------------------------------------------------------------------------
# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
FIGURES_DIRECTORY = REPOSITORY / "figures"
DATA_DIRECTORY = REPOSITORY / "data_files"
FORCE_RECOMPUTE = False

# Pe=inf uses setup_bare_solution; finite values use setup_fp_solution.
SOLVER_CLASS = md.CDGraetzIsothermalSolution  # Or md.CDGraetzIsothermalSolutionLD.
SOLVER_CONFIG = {
    "inlet_temperature": 1.0,
    "wall_temperature": 0.0,
    "verbose_level": 0,
    "max_root": 400.0,
    "n_gauss_points": 100,
    "fp_tolerance": 1e-13,
    "fp_max_iter": 200,
    "fp_quad_panels": 64,             # Used by QEP only; retained for a complete solver setup.
    "fp_cap_quad_margin": 100,        # Sizes the Nusselt bulk-moment quadrature.
    "gram_method": md.GramMethod.GaussJacobiQR,
    "gram_oversampling_factor": 2,
    "gram_oversampling_margin": 20,
    "gram_minimum_factor": 2,
    "gram_order_check": False,
    "rhs_method": md.RhsMethod.Representer,
    "rhs_representer_margin": 40,
    # Radial: the Weighted projection leaves the wall slopes of the last retained
    # modes short of their Bessel limit, which the split evaluation relies on.
    "projection_space": md.ProjectionSpace.Radial,
}
# Finite Pe only; the bare series (Pe=inf) always uses Plain.
NUSSELT_EVALUATION = md.NusseltEvaluation.SlugSplit
PE_VALUES = (1, 2, 5, 10, 20, 50, 100, 1000, math.inf)
CURVE_X_MIN = 1e-4
CURVE_X_MAX = 5.0
CURVE_POINT_COUNT = 240
JSON_X_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
JSON_FILENAME = "nusselt_at_decades.json"
CLASSICAL_TABLE_FILENAME = "nusselt_classical_table.csv"
USE_TEX = True
FIGURE_SIZE = (8.2, 5.6)
FONT_SIZE = 20
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
GRID_MAJOR_ALPHA = 0.40
GRID_MINOR_ALPHA = 0.20
LEGEND_COLUMNS = 2
MARKER_SIZE = 1.0
CURVE_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "h", "p")
# Keep all Péclet-number curves in one visual family.  The increasing shade
# follows the order in PE_VALUES, while markers retain an additional cue.
PE_COLORS = plt.get_cmap("Blues")(np.linspace(0.30, 0.99, len(PE_VALUES)))
# Near-inlet asymptotes (dotted, curve colour): corner law 4/(pi Pe x), finite Pe only;
# drawn while above the fully developed value.
DRAW_ASYMPTOTES = True
# Each curve carries its Peclet number, written along it where the curves are still apart;
# this replaces their legend entries, which nine shades of one blue cannot support.
# Pe = 1000 and Pe = inf nearly coincide there, so their labels sit above and below the pair.
CURVE_LABEL_X = 2.5e-4
CURVE_LABEL_X_BY_PE = {100: 3.2e-4, 1000: 2.15e-4}     # where the default is crowded
CURVE_LABEL_OFFSETS = {1000: 9.0, math.inf: -10.0}  # points
CURVE_LABEL_COLOR = "#202020"
ASYMPTOTE_LINEWIDTH = 1.0

# Shah and London (1978), circular duct Table 6, retained for the generated
# fully developed comparison table below. No external data are drawn in the figure.
DEVELOPED_TABLE = {1: 4.030, 2: 3.925, 5: 3.769, 10: 3.697,
                   20: 3.670, 50: 3.660, math.inf: 3.6568}

# Shah and London (1978), Table 15, Pe -> inf column as (x*, Nu); drawn at x = 2 x*. That
# table is read there from the graphical results of Hennecke (1968).
CLASSICAL_LOCAL_TABLE = ((0.001, 10.13), (0.002, 8.04), (0.003, 7.04), (0.004, 6.43),
                         (0.005, 6.00), (0.010, 4.92), (0.020, 4.17), (0.050, 3.71),
                         (0.100, 3.66), (0.200, 3.66))
# Digitised two-sided-problem references as {Pe: ((x, Nu), ...)}, abscissa already in x.
# Hennecke (1968) Fig. 2 = Shah and London Table 15 (trust x >= 0.1/Pe; entrance values at
# Pe <= 10 differ from the semi-infinite problem by construction), and Lahjomri and
# Oubarra (1999) Fig. 2. Empty until transcribed.
HENNECKE_TABLE_15 = {}
LAHJOMRI_FIG2 = {}

matplotlib.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": 1.6,
})


def nusselt(obj, pe: float, xs):
    if math.isinf(pe):
        return obj.get_nusselt_number(list(xs))
    return obj.get_nusselt_number(list(xs), NUSSELT_EVALUATION)


def build_solver(pe: float):
    """Build one solution with all editable solver controls applied before setup."""
    obj = SOLVER_CLASS(SOLVER_CONFIG["inlet_temperature"],
                       SOLVER_CONFIG["wall_temperature"],
                       SOLVER_CONFIG["verbose_level"])
    obj.set_max_root(SOLVER_CONFIG["max_root"])
    obj.set_number_of_gauss_points(SOLVER_CONFIG["n_gauss_points"])
    obj.set_fp_tolerance(SOLVER_CONFIG["fp_tolerance"])
    obj.set_fp_max_iter(SOLVER_CONFIG["fp_max_iter"])
    obj.set_fp_quad_panels(SOLVER_CONFIG["fp_quad_panels"])
    obj.set_fp_cap_quad_margin(SOLVER_CONFIG["fp_cap_quad_margin"])
    obj.set_gram_method(SOLVER_CONFIG["gram_method"])
    obj.set_gram_oversampling(SOLVER_CONFIG["gram_oversampling_factor"],
                              SOLVER_CONFIG["gram_oversampling_margin"])
    obj.set_gram_minimum_factor(SOLVER_CONFIG["gram_minimum_factor"])
    obj.set_gram_order_check(SOLVER_CONFIG["gram_order_check"])
    obj.set_rhs_method(SOLVER_CONFIG["rhs_method"])
    obj.set_rhs_representer_margin(SOLVER_CONFIG["rhs_representer_margin"])
    obj.set_projection_space(SOLVER_CONFIG["projection_space"])
    if math.isinf(pe):
        obj.setup_bare_solution()
    else:
        obj.setup_fp_solution(float(pe))
    return obj


def write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def curve_column(pe: float) -> str:
    """Return the CSV column used by one Péclet-number curve."""
    return "bare" if math.isinf(pe) else f"Pe_{pe:g}"


def load_curves_csv(path: Path):
    """Load cached curve coordinates for the configured Péclet numbers."""
    expected_header = ("x",) + tuple(curve_column(pe) for pe in PE_VALUES)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(
                f"{path} columns do not match the configured PE_VALUES: "
                f"expected {', '.join(expected_header)}"
            )
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two curve points")
    xs = np.asarray([float(row["x"]) for row in rows])
    curves = {
        pe: np.asarray([float(row[curve_column(pe)]) for row in rows])
        for pe in PE_VALUES
    }
    if (not np.all(np.isfinite(xs)) or np.any(xs <= 0)
            or np.any(np.diff(xs) <= 0)):
        raise ValueError(f"{path} must contain finite, positive, increasing x values")
    if any(not np.all(np.isfinite(nu)) or np.any(nu <= 0)
           for nu in curves.values()):
        raise ValueError(f"{path} must contain finite, positive Nusselt numbers")
    return xs, curves


def label_curve(ax, xs, nu, pe: float) -> None:
    """Write the Peclet number on its curve at CURVE_LABEL_X, rotated along the curve."""
    i = int(np.searchsorted(xs, CURVE_LABEL_X_BY_PE.get(pe, CURVE_LABEL_X)))
    start, end = ax.transData.transform([(xs[i - 1], nu[i - 1]), (xs[i + 1], nu[i + 1])])
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    text = r"$\mathrm{Pe}=\infty$" if math.isinf(pe) else rf"$\mathrm{{Pe}}={pe:g}$"
    ax.annotate(text, (xs[i], nu[i]), xytext=(0.0, CURVE_LABEL_OFFSETS.get(pe, 0.0)),
                textcoords="offset points", rotation=angle, rotation_mode="anchor",
                ha="center", va="center", fontsize=LEGEND_FONT_SIZE, color=CURVE_LABEL_COLOR,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"),
                zorder=5)


def write_classical_table(path: Path) -> None:
    """Bare series (Pe = inf) at the abscissae of Shah and London, and its relative difference.

    Their abscissa x_hat is x/2 here, and Delta_rel = (Nu_Shah-London - Nu_K) / Nu_K.
    """
    xs = [2.0 * x_star for x_star, _ in CLASSICAL_LOCAL_TABLE]
    present = build_solver(math.inf).get_nusselt_number(xs)
    rows = [(x_star, x, reference, value, 1e3 * (reference - value) / value)
            for (x_star, reference), x, value in zip(CLASSICAL_LOCAL_TABLE, xs, present)]
    write_csv(path, ("x_hat", "x", "Shah_London_1978", "present", "1e3_Delta_rel"), rows)
    print(f"Saved {path}")


def plot_curves(xs, curves, developed_by_pe=None) -> None:
    """Render the configured figure from computed or cached curve data."""
    if len(CURVE_MARKERS) < len(PE_VALUES):
        raise ValueError("CURVE_MARKERS must provide a distinct marker for every Péclet value")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for pe, marker, color in zip(PE_VALUES, CURVE_MARKERS, PE_COLORS):
        nu = curves[pe]
        developed = (nu[-1] if developed_by_pe is None else developed_by_pe[pe])
        line, = ax.loglog(xs, nu, "--" if math.isinf(pe) else "-", linewidth=2.0,
                          color=color, marker=marker,
                          markersize=MARKER_SIZE)
        if DRAW_ASYMPTOTES and not math.isinf(pe):
            asymptote = 4.0 / (math.pi * pe * xs)
            above = asymptote >= developed
            ax.loglog(xs[above], asymptote[above], ":", color=line.get_color(),
                      linewidth=ASYMPTOTE_LINEWIDTH)
        for source, marker in ((HENNECKE_TABLE_15, "s"), (LAHJOMRI_FIG2, "^")):
            if source.get(pe):
                ax.loglog(*zip(*source[pe]), marker, color=line.get_color(), fillstyle="none")

    ax.loglog([2.0 * x_star for x_star, _ in CLASSICAL_LOCAL_TABLE],
              [value for _, value in CLASSICAL_LOCAL_TABLE], "ko", fillstyle="none",
              label="Shah--London" if USE_TEX else "Shah-London")
    ax.set(xlabel=r"$x$", ylabel=r"$\mathrm{Nu}_K(x)$", xlim=(xs[0], xs[-1]), ylim=(1.0, 3e4))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=GRID_MAJOR_ALPHA)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=GRID_MINOR_ALPHA)
    ax.tick_params(which="major", direction="in", top=True, right=True, width=1.6, length=8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, width=1.1, length=4.5)
    ax.legend(ncol=LEGEND_COLUMNS, frameon=False, columnspacing=1.2, handlelength=2.4,
              loc="upper right")
    fig.tight_layout()
    # After the layout is final, so that the label angles match the drawn curves.
    for pe in PE_VALUES:
        label_curve(ax, xs, curves[pe], pe)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIRECTORY / "Figure_3.pdf")
    plt.close(fig)


def run() -> None:
    if SOLVER_CONFIG["max_root"] <= 0 or CURVE_X_MIN <= 0 or CURVE_X_MAX <= CURVE_X_MIN:
        raise ValueError("max_root and curve bounds must be positive")
    if CURVE_POINT_COUNT < 2 or not PE_VALUES or any(pe <= 0 for pe in PE_VALUES):
        raise ValueError("at least two curve points and positive Péclet numbers are required")

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    classical_table_path = DATA_DIRECTORY / CLASSICAL_TABLE_FILENAME
    if FORCE_RECOMPUTE or not classical_table_path.is_file():
        write_classical_table(classical_table_path)
    curves_csv_path = DATA_DIRECTORY / "nusselt_curves.csv"
    if not FORCE_RECOMPUTE and curves_csv_path.is_file():
        print(f"Loading existing curve data from {curves_csv_path}...")
        xs, curves = load_curves_csv(curves_csv_path)
        plot_curves(xs, curves)
        print(f"Saved {FIGURES_DIRECTORY / 'Figure_3.pdf'} without recomputing the curves.")
        return
    if not FORCE_RECOMPUTE:
        print(f"No cached curve data found at {curves_csv_path}; computing the curves.")

    xs = np.geomspace(CURVE_X_MIN, CURVE_X_MAX, CURVE_POINT_COUNT)
    curves = {}
    developed_by_pe = {}
    nusselt_at_decades = {}
    developed_rows = []
    for pe in PE_VALUES:
        obj = build_solver(pe)
        nu = np.asarray(nusselt(obj, pe, xs))
        developed = obj.get_fully_developed_nusselt_number()
        curves[pe] = nu
        developed_by_pe[pe] = developed
        label = r"$\mathrm{Pe}=\infty$" if math.isinf(pe) else rf"$\mathrm{{Pe}}={pe:g}$"
        json_label = "Pe=inf" if math.isinf(pe) else f"Pe={pe:g}"
        nusselt_at_decades[json_label] = [float(value) for value in nusselt(obj, pe, JSON_X_VALUES)]
        developed_rows.append((label, developed, DEVELOPED_TABLE.get(pe, "")))
        print(f"{label}: Nu_inf={developed:.8f}")

    plot_curves(xs, curves, developed_by_pe)

    write_csv(curves_csv_path, ("x",) + tuple(curve_column(pe) for pe in PE_VALUES),
        ((x, *(curves[pe][j] for pe in PE_VALUES)) for j, x in enumerate(xs)))
    json_report = {
        "max_root": SOLVER_CONFIG["max_root"],
        "evaluation": str(NUSSELT_EVALUATION),
        "projection_space": str(SOLVER_CONFIG["projection_space"]),
        "x_values": list(JSON_X_VALUES),
        "nusselt_number": nusselt_at_decades,
    }
    with (DATA_DIRECTORY / JSON_FILENAME).open("w") as stream:
        json.dump(json_report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(json_report, indent=2, allow_nan=False))
    write_csv(DATA_DIRECTORY / "nusselt_developed.csv",
              ("Pe", "computed", "Shah_London_1978"), developed_rows)
    row_break = r"\\"
    with (DATA_DIRECTORY / "nusselt_developed.tex").open("w") as stream:
        stream.write("\\begin{tabular}{lrr}\n\\toprule\n"
                     "$Pe$ & computed $Nu_\\infty$ & Shah--London " + row_break + "\n\\midrule\n")
        for pe, (_, computed, literature) in zip(PE_VALUES, developed_rows):
            table_pe = r"\infty" if math.isinf(pe) else f"{pe:g}"
            stream.write(f"${table_pe}$ & {computed:.5f} & {literature} {row_break}\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    run()
