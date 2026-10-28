"""Plot the Nusselt-number error against the eigenvalue cutoff, plain sum vs SlugSplit.

Run with:
    python3 nusselt_convergence.py

The reference Nu_N is the SlugSplit evaluation (eq. nusselt_split_evaluation) at the
large cutoff REFERENCE_BETA. For every beta_K in BETA_CUTOFFS the plotted ordinate is
|Nu_K - Nu_N| / Nu_N, for the plain sum of the retained modes and for SlugSplit,
both evaluated on the same finite-Péclet solution. Rows show
the axial positions in X_VALUES and columns separate the two evaluations.

The figure is drawn at its printed width (the text width of the supplement), so the
font sizes below are the printed sizes in points.

Every solve is written to the CSV as soon as it finishes, and an interrupted run
resumes from it. Differences that are exactly zero are not drawn on the log axis.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

import MassDiffusion3Dcpp as md


# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = Path(__file__).resolve().parents[1]
FIGURES_DIRECTORY = REPOSITORY / "figures"
DATA_DIRECTORY = REPOSITORY / "data_files"
CSV_FILENAME = "nusselt_convergence_plain_vs_split.csv"
PDF_FILENAME = "Figure_4.pdf"
X_VALUES = (1e-2, 1e-4)
PE_VALUES = (1, 10, 100, 1000)
REFERENCE_BETA = 1600.0
BETA_CUTOFFS = list(np.geomspace(50, 700, 10))
CURVE_MARKERS = ("o", "s", "^", "v", "D", "P")
PE_COLORS = plt.get_cmap("Blues")(np.linspace(0.30, 0.99, len(PE_VALUES)))
X_AXIS_PADDING_FRACTION = 0.025

SOLVER_CLASS = md.CDGraetzIsothermalSolution
# Above beta = 800 the Pe = 1000 rates exceed the double-precision range of the solver.
SOLVER_CLASS_BY_PE = {1000: md.CDGraetzIsothermalSolutionLD}
SOLVER_CONFIG = {
    "inlet_temperature": 1.0,
    "wall_temperature": 0.0,
    "verbose_level": 0,
    "n_gauss_points": 100,
    "fp_tolerance": 1e-13,
    "fp_max_iter": 200,
    "fp_quad_panels": 64,
    "fp_cap_quad_margin": 100,
    "gram_method": md.GramMethod.GaussJacobiQR,
    "gram_oversampling_factor": 2,
    "gram_oversampling_margin": 20,
    "gram_minimum_factor": 2,
    "gram_order_check": False,
    "rhs_method": md.RhsMethod.Representer,
    "rhs_representer_margin": 40,
    "projection_space": md.ProjectionSpace.Radial,
}

# Printed size: the supplement's text width (A4, 2.5 cm margins) is 160 mm.
FIGURE_WIDTH_MM = 160.0
FIGURE_HEIGHT_MM = 130.0
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
PANEL_FONT_SIZE = 11
LINE_WIDTH = 1.0
MARKER_SIZE = 4.0
AXES_LINE_WIDTH = 0.8
GRID_MAJOR_ALPHA = 0.40
GRID_MINOR_ALPHA = 0.20
LEGEND_COLUMNS = len(PE_VALUES)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": LABEL_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": AXES_LINE_WIDTH,
})

CSV_COLUMNS = ("Pe", "x", "beta", "K", "Nu_plain", "Nu_split")


def build_solver(pe: float, beta_cutoff: float):
    config = SOLVER_CONFIG
    obj = SOLVER_CLASS_BY_PE.get(pe, SOLVER_CLASS)(config["inlet_temperature"],
                                                   config["wall_temperature"],
                                                   config["verbose_level"])
    obj.set_max_root(beta_cutoff)
    obj.set_number_of_gauss_points(config["n_gauss_points"])
    obj.set_fp_tolerance(config["fp_tolerance"])
    obj.set_fp_max_iter(config["fp_max_iter"])
    obj.set_fp_quad_panels(config["fp_quad_panels"])
    obj.set_fp_cap_quad_margin(config["fp_cap_quad_margin"])
    obj.set_gram_method(config["gram_method"])
    obj.set_gram_oversampling(config["gram_oversampling_factor"],
                              config["gram_oversampling_margin"])
    obj.set_gram_minimum_factor(config["gram_minimum_factor"])
    obj.set_gram_order_check(config["gram_order_check"])
    obj.set_rhs_method(config["rhs_method"])
    obj.set_rhs_representer_margin(config["rhs_representer_margin"])
    obj.set_projection_space(config["projection_space"])
    obj.setup_fp_solution(float(pe))
    return obj


def logarithmic_ticks(lower: float, upper: float) -> list[float]:
    ticks = {lower, upper}
    for exponent in range(math.floor(math.log10(lower)), math.ceil(math.log10(upper)) + 1):
        scale = 10.0 ** exponent
        ticks.update(value for value in (scale, 2.0 * scale, 5.0 * scale) if lower <= value <= upper)
    return sorted(ticks)


def key(pe: float, beta: float, x: float) -> tuple[str, float, float]:
    return f"{pe:g}", round(float(beta), 9), float(f"{x:.6g}")


def load_rows(csv_path: Path) -> dict:
    rows = {}
    if not csv_path.is_file():
        return rows
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows[key(float(row["Pe"]), float(row["beta"]), float(row["x"]))] = (
                int(row["K"]), float(row["Nu_plain"]), float(row["Nu_split"]))
    return rows


def write_rows(csv_path: Path, rows: dict) -> None:
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for (pe_label, beta, x), (terms, plain, split) in sorted(
                rows.items(), key=lambda item: (-item[0][2], float(item[0][0]), item[0][1])):
            writer.writerow((pe_label, x, beta, terms, plain, split))


def compute_missing(csv_path: Path, rows: dict) -> None:
    jobs = [(pe, beta) for pe in PE_VALUES for beta in [REFERENCE_BETA] + BETA_CUTOFFS]
    missing = [job for job in jobs if any(key(*job, x) not in rows for x in X_VALUES)]
    started = perf_counter()
    for count, (pe, beta) in enumerate(missing, start=1):
        case_started = perf_counter()
        obj = build_solver(pe, float(beta))
        terms = len(obj.get_nusselt_mode_data())
        values = []
        for x in X_VALUES:
            plain = float(obj.get_nusselt_number(x, md.NusseltEvaluation.Plain))
            split = float(obj.get_nusselt_number(x, md.NusseltEvaluation.SlugSplit))
            rows[key(pe, beta, x)] = (terms, plain, split)
            values.append(f"x={x:g}: plain={plain:.12g}, split={split:.12g}")
        write_rows(csv_path, rows)
        print(f"  [{count}/{len(missing)}] Pe={pe:g}, beta={beta:.4g}, K={terms}; "
              + "; ".join(values)
              + f" ({perf_counter() - case_started:.1f}s, {perf_counter() - started:.0f}s total)",
              flush=True)


def panel_label(x: float) -> str:
    exponent = math.log10(x)
    if math.isclose(exponent, round(exponent)):
        return rf"$x = 10^{{{int(round(exponent))}}}$"
    return rf"$x = {x:g}$"


def plot(rows: dict) -> None:
    mm = 1.0 / 25.4
    fig, axes = plt.subplots(len(X_VALUES), 2, sharex=True, sharey=True,
                             figsize=(FIGURE_WIDTH_MM * mm, FIGURE_HEIGHT_MM * mm),
                             constrained_layout=True)
    # Keep room for the vertical row labels at the right and the legend above the plots.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 0.94, 0.90), hspace=0.0,
                                wspace=0.06, h_pad=0.02)
    lower, upper = min(BETA_CUTOFFS), max(BETA_CUTOFFS)
    for row, x in enumerate(X_VALUES):
        plain_ax, asymptotic_ax = axes[row]
        for pe, marker, color in zip(PE_VALUES, CURVE_MARKERS, PE_COLORS):
            reference = rows[key(pe, REFERENCE_BETA, x)][2]
            plain_errors = np.array([
                abs(rows[key(pe, b, x)][1] - reference) / abs(reference)
                for b in BETA_CUTOFFS
            ])
            asymptotic_errors = np.array([
                abs(rows[key(pe, b, x)][2] - reference) / abs(reference)
                for b in BETA_CUTOFFS
            ])
            plain_errors[plain_errors == 0.0] = np.nan
            asymptotic_errors[asymptotic_errors == 0.0] = np.nan
            plain_ax.plot(BETA_CUTOFFS, plain_errors, linestyle="-", marker=marker,
                          color=color, markerfacecolor=color, linewidth=LINE_WIDTH,
                          markersize=MARKER_SIZE)
            asymptotic_ax.plot(BETA_CUTOFFS, asymptotic_errors, linestyle="-", marker=marker,
                               color=color, markerfacecolor=color, linewidth=LINE_WIDTH,
                               markersize=MARKER_SIZE)

        for ax in (plain_ax, asymptotic_ax):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=100))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=GRID_MAJOR_ALPHA)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=GRID_MINOR_ALPHA)
            ax.tick_params(which="major", direction="in", top=True, right=True,
                           width=AXES_LINE_WIDTH, length=4)
            ax.tick_params(which="minor", direction="in", top=True, right=True,
                           width=0.6, length=2)

        plain_ax.set_ylabel(r"$|Nu_K-Nu_N|\,/\,Nu_N$", labelpad=12)
        asymptotic_ax.tick_params(axis="y", labelleft=False)
        if row == 0:
            plain_ax.set_title("Plain", fontsize=PANEL_FONT_SIZE)
            asymptotic_ax.set_title("Asymptotic", fontsize=PANEL_FONT_SIZE)
        asymptotic_ax.text(1.06, 0.5, panel_label(x), transform=asymptotic_ax.transAxes,
                           fontsize=PANEL_FONT_SIZE, ha="center", va="center", rotation=90)

    bottom = axes[-1]
    for ax in bottom:
        ax.set_xlabel(r"$\beta_K$")
    log_padding = math.exp(X_AXIS_PADDING_FRACTION * math.log(upper / lower))
    bottom[0].set_xlim(lower / log_padding, upper * log_padding)
    for ax in bottom:
        ax.xaxis.set_major_locator(FixedLocator(logarithmic_ticks(lower, upper)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())

    pe_handles = [Line2D([], [], color=color, marker=marker, linewidth=LINE_WIDTH,
                         markersize=MARKER_SIZE, label=rf"$\mathrm{{Pe}}={pe:g}$")
                  for pe, marker, color in zip(PE_VALUES, CURVE_MARKERS, PE_COLORS)]
    fig.legend(handles=pe_handles, ncol=LEGEND_COLUMNS, frameon=False,
               columnspacing=1.2, handlelength=2.2, loc="upper center",
               bbox_to_anchor=(0.47, 1.0))
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIRECTORY / PDF_FILENAME)
    plt.close(fig)


def run() -> None:
    if any(beta >= REFERENCE_BETA for beta in BETA_CUTOFFS):
        raise ValueError("every beta cutoff must lie below the reference cutoff")
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIRECTORY / CSV_FILENAME
    rows = load_rows(csv_path)
    print(f"Nusselt error at x={X_VALUES}: reference SlugSplit at beta_N={REFERENCE_BETA:g}; "
          f"{len(rows)} evaluations cached in {csv_path}.", flush=True)
    compute_missing(csv_path, rows)
    plot(rows)
    print(f"Saved {FIGURES_DIRECTORY / PDF_FILENAME}.", flush=True)


if __name__ == "__main__":
    run()
