"""
plot_solution_3D.py
===================

Perspective views of the exact finite-Peclet scalar field over the circular
cross-section.  Columns are axial positions and rows are solution families:

    * top row    -> isothermal Graetz problem,
    * bottom row -> stratified inlet with a zero-flux wall.

The figure contains only the finite-Peclet solution.  Its coefficients are
computed with the default weighted L2 projection.  The surfaces are drawn on a
triangulated disk, with ``Spectral_r`` colours and fine dark mesh lines.  The
output is ``figures/Figure_5`` at the root of the repository.

Everything normally worth changing is an ALL-CAPS parameter below.  The evaluated
fields are stored in ``DATA_FILE`` and reused on the next run while the stored
configuration matches the script's; set ``FORCE_RECOMPUTE = True`` to recompute them.
"""

import json
import os

import matplotlib
import numpy as np


# Repository root: data_scripts/ sits in it, next to data_files/ and figures/.
REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Physics
# ============================================================

PECLET = 5.0
MAX_ROOT = 300.0
VERBOSE_SOLVER = 0

# Graetz data follow the paper.  The stratified inlet uses four equal-height
# layers whose values decrease uniformly from bottom to top.
GRAETZ_T0 = 1.0
GRAETZ_T_WALL = 0.0
STRATIFIED_ZI = (-0.5, 0.0, 0.5)
STRATIFIED_UI = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)  # bottom to top

# Four representative planes.  Log-like spacing resolves the rapid smoothing
# near the inlet while retaining the requested downstream endpoint x = 0.5.
X_VALUES = np.array([0.0, 1.0e-2, 1.0e-1, 5.0e-1])


# ============================================================
# Evaluation mesh
# ============================================================

# The points lie on polar rings, which lets the C++ solver reuse each radial
# mode evaluation.  Delaunay triangulation is performed once and shared by all
# eight panels.
N_RADIAL = 72

# Visual surface smoothing. This does not alter the analytical field:
#   0.0 -> fully visible black triangulation (the original detailed rendering),
#   1.0 -> no triangulation texture (the clean, cartoon-like cylinder style).
# Intermediate values continuously fade and thin the mesh; 0.5 is halfway.
SMOOTHING_INTENSITY = 0.7


# ============================================================
# Style and layout
# ============================================================

USE_LATEX = True
FONT_SIZE = 22
FIGURE_SIZE = (14.0, 7.8)

CMAP_NAME = "Spectral_r"
SURFACE_EDGE_COLOR = (0.0, 0.0, 0.0)
SURFACE_EDGE_ALPHA = 0.24
SURFACE_LINE_WIDTH = 0.10
SURFACE_ALPHA = 0.96
RASTERIZE_SURFACES = True

VIEW_ELEV = 30.0
VIEW_AZIM = 25.0
PROJECTION_TYPE = "persp"
BOX_ASPECT = (1.0, 1.0, 0.72)
BOX_ZOOM = 1.18

ROW_LABELS = (r"Graetz", r"Stratified")
ROW_LABEL_X = 0.975

# Minimal coordinate axes drawn on every surface panel.
AXIS_COLOR = "#777777"
AXIS_LINE_WIDTH = 0.65
AXIS_LABEL_SIZE = 20
AXIS_LABEL_PAD = -13
PSI_LABEL_POSITION = (-0.005, 0.735)  # axes coordinates, at the z-axis tip

# Perspective guide grid.  The alpha is intentionally low so the grid explains
# depth without competing with the coloured solution surface.
GRID_COLOR = (0.45, 0.45, 0.45, 0.20)
GRID_LINE_WIDTH = 0.50
GRID_LINE_STYLE = "-"
GRID_N_PARTITIONS = 4

# Axial annotation, drawn in figure coordinates below the bottom row.
X_FORMAT = r"${:g}$"
X_ARROW_LABEL = r"$x$"
X_ARROW_COLOR = "#3A3A3A"
X_ARROW_ALPHA = 0.10
X_ARROW_SHAFT_HALF_HEIGHT = 0.022
X_ARROW_HEAD_LENGTH = 0.038
X_ARROW_HEAD_HALF_HEIGHT = 0.033
X_ARROW_VERTICAL_GAP = 0.075

# Shared colour scale (the "legend") above the top row.
COLORBAR_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
COLORBAR_RECT = (0.12, 0.925, 0.76, 0.022)

OUTPUT_DIR = os.path.join(REPOSITORY, "figures")
OUTPUT_NAME = "Figure_5"
DATA_FILE = os.path.join(REPOSITORY, "data_files", "plot_solution_3D.npz")
FORCE_RECOMPUTE = False  # False: reuse DATA_FILE when its configuration matches
SAVE_PDF = True
SAVE_PNG = True
DPI = 300
SHOW_PLOT = False

if not SHOW_PLOT:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import Delaunay

import MassDiffusion3Dcpp as MD


# ============================================================
# Solver construction
# ============================================================

def _setup_finite_peclet(sol):
    """Set up the exact finite-Peclet series with weighted-L2 coefficients."""
    sol.set_max_root(float(MAX_ROOT))

    # ``weighted`` is the library default.  Selecting it explicitly documents
    # the intended coefficient projection and protects the figure from a local
    # or future change of solver defaults.
    if hasattr(sol, "set_projection_space"):
        sol.set_projection_space("weighted")

    print(f"Building {type(sol).__name__} at Pe = {PECLET:g} "
          f"(weighted L2 projection, max root = {MAX_ROOT:g})...")
    sol.setup_fp_solution(float(PECLET))
    print(f"  axial diffusion active: {sol.using_axial_diffusion()}, "
          f"active modes: {sol.get_max_K()}")
    return sol


def build_graetz():
    """Finite-Peclet isothermal Graetz solution used by the top row."""
    sol = MD.CDGraetzIsothermalSolution(
        float(GRAETZ_T0), float(GRAETZ_T_WALL), VERBOSE_SOLVER)
    return _setup_finite_peclet(sol)


def build_stratified():
    """Finite-Peclet, zero-flux stratified solution used by the bottom row."""
    sol = MD.CDStratifiedSolution(
        list(STRATIFIED_ZI), list(STRATIFIED_UI), VERBOSE_SOLVER)
    return _setup_finite_peclet(sol)


# ============================================================
# Mesh and solution evaluation
# ============================================================

def make_disk_mesh():
    """Return disk points in solver coordinates and their triangulation."""
    radii = np.linspace(0.0, 1.0, N_RADIAL)
    radial_spacing = radii[1] - radii[0]

    # Keep triangles approximately equilateral by increasing the number of
    # angular points with circumference.  Using the same large angular count on
    # every ring makes hundreds of dark edges converge at the origin, producing
    # a false black spot in the rendered solution.
    # As in the benchmark mesh, there is no vertex exactly at the origin.  The
    # innermost hexagon is triangulated across its interior by Delaunay, so no
    # collection of mesh edges can converge into a central black hub.
    y_parts = []
    z_parts = []
    r_parts = []
    phi_parts = []
    for ring_index, radius in enumerate(radii[1:], start=1):
        n_phi = max(6, int(np.ceil(2.0 * np.pi * radius / radial_spacing)))
        offset = (ring_index % 2) * np.pi / n_phi
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi,
                          endpoint=False) + offset
        y_parts.append(radius * np.sin(phi))
        z_parts.append(radius * np.cos(phi))
        r_parts.append(np.full(phi.shape, radius))
        phi_parts.append(phi)

    y = np.concatenate(y_parts)
    z = np.concatenate(z_parts)
    r = np.concatenate(r_parts)
    phi = np.concatenate(phi_parts)
    triangles = Delaunay(np.column_stack((y, z))).simplices
    return y, z, r, phi, triangles


def evaluate_planes(sol, r, phi):
    """Evaluate all requested axial planes in one call to the C++ solver."""
    print(f"Computing {len(X_VALUES)} planes on {r.size} disk points "
          f"({np.unique(r).size} distinct radii)...")
    values = sol.get_solution_at_planes(
        X_VALUES.astype(float).tolist(), r.tolist(), phi.tolist())
    fields = np.asarray(values, dtype=float)
    expected_shape = (len(X_VALUES), r.size)
    if fields.shape != expected_shape:
        fields = fields.reshape(expected_shape)
    if not np.all(np.isfinite(fields)):
        raise RuntimeError("Finite-Peclet evaluation produced non-finite values")
    print(f"  field range: {fields.min():.5f} to {fields.max():.5f}\n")
    return fields


# ============================================================
# Drawing
# ============================================================

def surface_edge_style():
    """Interpolate from the detailed mesh to an edge-free smooth surface."""
    intensity = float(SMOOTHING_INTENSITY)
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("SMOOTHING_INTENSITY must lie between 0 and 1")

    edge_alpha = SURFACE_EDGE_ALPHA * (1.0 - intensity)
    edge_width = SURFACE_LINE_WIDTH * (1.0 - 0.70 * intensity)
    edge_color = (*SURFACE_EDGE_COLOR, edge_alpha)
    return edge_color, edge_width


def draw_surface(ax, y, z, triangles, field, cmap, norm):
    """Draw one benchmark-style triangulated solution surface."""
    edge_color, edge_width = surface_edge_style()
    surface = ax.plot_trisurf(
        y, z, field,
        triangles=triangles,
        cmap=cmap,
        norm=norm,
        linewidth=edge_width,
        edgecolor=edge_color,
        shade=False,
        antialiased=False,
        alpha=SURFACE_ALPHA,
    )
    surface.set_rasterized(RASTERIZE_SURFACES)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(norm.vmin, norm.vmax)
    try:
        ax.set_box_aspect(BOX_ASPECT, zoom=BOX_ZOOM)
    except TypeError:  # matplotlib versions predating the ``zoom`` keyword
        ax.set_box_aspect(BOX_ASPECT)
    ax.set_proj_type(PROJECTION_TYPE)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    style_coordinate_axes(ax)
    return surface


def style_coordinate_axes(ax):
    """Draw thin grey axes and a very faint perspective guide grid."""
    if GRID_N_PARTITIONS < 1:
        raise ValueError("GRID_N_PARTITIONS must be at least 1")

    # Five nodes, including both exact limits, produce exactly four cells.  The
    # limits are restored after set_*ticks because Matplotlib may otherwise
    # expand an axis slightly to accommodate a tick at its boundary.
    x_limits = (-1.0, 1.0)
    y_limits = (-1.0, 1.0)
    z_limits = (0.0, 1.0)
    ax.set_xticks(np.linspace(*x_limits, GRID_N_PARTITIONS + 1))
    ax.set_yticks(np.linspace(*y_limits, GRID_N_PARTITIONS + 1))
    ax.set_zticks(np.linspace(*z_limits, GRID_N_PARTITIONS + 1))
    ax.set_xlim(*x_limits, auto=False)
    ax.set_ylim(*y_limits, auto=False)
    ax.set_zlim(*z_limits, auto=False)
    ax.set_autoscale_on(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, width=0, pad=0)
    ax.grid(True)

    ax.set_xlabel(r"$y$", color=AXIS_COLOR, fontsize=AXIS_LABEL_SIZE,
                  labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel(r"$z$", color=AXIS_COLOR, fontsize=AXIS_LABEL_SIZE,
                  labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("")
    ax.text2D(*PSI_LABEL_POSITION, r"$\psi$", transform=ax.transAxes,
              color=AXIS_COLOR, fontsize=AXIS_LABEL_SIZE,
              ha="center", va="bottom", rotation=0)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(True)
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.fill = False
        axis.line.set_color(AXIS_COLOR)
        axis.line.set_linewidth(AXIS_LINE_WIDTH)
        # Retain major-tick locations for the grid, but suppress the short tick
        # strokes themselves (3D axes do not honour tick_params(length=0)).
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["grid"].update({
            "color": GRID_COLOR,
            "linewidth": GRID_LINE_WIDTH,
            "linestyle": GRID_LINE_STYLE,
        })


def add_x_arrow(fig, bottom_axes):
    """Cylinder-style axial arrow and x values below the bottom row."""
    positions = [ax.get_position() for ax in bottom_axes]
    centers = [box.x0 + 0.5 * box.width for box in positions]
    y_arrow = min(box.y0 for box in positions) - X_ARROW_VERTICAL_GAP
    x_start = centers[0] - 0.052
    head_tip = centers[-1] + 0.052
    head_base = head_tip - X_ARROW_HEAD_LENGTH

    # Match cylinder_plot_solution.py: a continuous rectangular shaft and a
    # separate, elongated triangular head, both broad and very translucent.
    shaft = Rectangle(
        (x_start, y_arrow - X_ARROW_SHAFT_HALF_HEIGHT),
        head_base - x_start,
        2.0 * X_ARROW_SHAFT_HALF_HEIGHT,
        transform=fig.transFigure,
        facecolor=X_ARROW_COLOR,
        edgecolor="none",
        alpha=X_ARROW_ALPHA,
        zorder=20,
    )
    head = Polygon(
        [[head_tip, y_arrow],
         [head_base, y_arrow + X_ARROW_HEAD_HALF_HEIGHT],
         [head_base, y_arrow - X_ARROW_HEAD_HALF_HEIGHT]],
        closed=True,
        transform=fig.transFigure,
        facecolor=X_ARROW_COLOR,
        edgecolor="none",
        alpha=X_ARROW_ALPHA,
        zorder=20,
    )
    fig.add_artist(shaft)
    fig.add_artist(head)

    for center, x_value in zip(centers, X_VALUES):
        fig.text(center, y_arrow, X_FORMAT.format(x_value),
                 ha="center", va="center", color=X_ARROW_COLOR,
                 fontsize=FONT_SIZE, zorder=21)
    fig.text(head_tip + 0.010, y_arrow, X_ARROW_LABEL,
             ha="left", va="center", color=X_ARROW_COLOR,
             fontsize=FONT_SIZE, zorder=21)


def data_configuration():
    """Everything the fields depend on; a cache computed otherwise is rejected."""
    return json.loads(json.dumps({
        "format_version": 1,
        "peclet": PECLET,
        "max_root": MAX_ROOT,
        "graetz_t0": GRAETZ_T0,
        "graetz_t_wall": GRAETZ_T_WALL,
        "stratified_zi": STRATIFIED_ZI,
        "stratified_ui": STRATIFIED_UI,
        "x_values": X_VALUES.tolist(),
        "n_radial": N_RADIAL,
    }, sort_keys=True))


def get_fields():
    """Disk mesh and fields; reuse DATA_FILE when allowed and compatible, else compute and store."""
    if not FORCE_RECOMPUTE:
        try:
            with np.load(DATA_FILE, allow_pickle=False) as archive:
                if json.loads(str(archive["configuration"].item())) != data_configuration():
                    raise ValueError("the stored configuration does not match this script")
                y, z, triangles = archive["y"], archive["z"], archive["triangles"]
                rows = [archive["graetz"], archive["stratified"]]
            print(f"Reusing the fields stored in {DATA_FILE}")
            return y, z, triangles, rows
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not reuse {DATA_FILE}: {error}; recomputing the fields")

    y, z, r, phi, triangles = make_disk_mesh()
    rows = [
        evaluate_planes(build_graetz(), r, phi),
        evaluate_planes(build_stratified(), r, phi),
    ]
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    np.savez_compressed(
        DATA_FILE,
        configuration=np.array(json.dumps(data_configuration(), sort_keys=True)),
        x=X_VALUES, y=y, z=z, r=r, phi=phi, triangles=triangles,
        graetz=rows[0], stratified=rows[1],
    )
    print(f"Saved data {DATA_FILE}")
    return y, z, triangles, rows


def make_figure():
    plt.rcParams.update({
        "text.usetex": USE_LATEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": FONT_SIZE,
    })

    y, z, triangles, rows = get_fields()
    print(f"Visual surface smoothing intensity: {SMOOTHING_INTENSITY:g}")

    # A fixed physical [0, 1] scale makes all panels directly comparable.  The
    # truncated inlet series can ring slightly outside it; the colour mapping
    # clips those small overshoots while the surface height remains untouched.
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    cmap = plt.get_cmap(CMAP_NAME)

    fig, axes = plt.subplots(
        2, len(X_VALUES),
        figsize=FIGURE_SIZE,
        subplot_kw={"projection": "3d"},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.945, bottom=0.145, top=0.895,
                        wspace=-0.20, hspace=0.06)

    for row_index, fields in enumerate(rows):
        for column_index, field in enumerate(fields):
            draw_surface(axes[row_index, column_index], y, z, triangles,
                         field, cmap, norm)

    # Solution-family labels live just to the right of the surfaces.
    for label, row_axes in zip(ROW_LABELS, axes):
        boxes = [ax.get_position() for ax in row_axes]
        center_y = 0.5 * (min(box.y0 for box in boxes) + max(box.y1 for box in boxes))
        fig.text(ROW_LABEL_X, center_y, label, rotation=270,
                 ha="center", va="center", fontsize=FONT_SIZE)

    add_x_arrow(fig, axes[1])

    cax = fig.add_axes(COLORBAR_RECT)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                            ticks=COLORBAR_TICKS)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(length=2.5, pad=1.5, labelsize=FONT_SIZE - 2)

    return fig


def main():
    fig = make_figure()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

    for enabled, extension in ((SAVE_PDF, "pdf"), (SAVE_PNG, "png")):
        if not enabled:
            continue
        path = f"{base_path}.{extension}"
        fig.savefig(path, dpi=DPI)
        print(f"Figure '{path}' saved.")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
