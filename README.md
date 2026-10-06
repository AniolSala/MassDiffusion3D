# MassDiffusion3DPlus

Semi-analytical solutions of the steady convection–diffusion equation for a passive scalar (concentration or temperature) in a circular tube, including axial diffusion at finite Péclet number. The numerical kernel is implemented in C++17 (OpenMP) and exposed to Python through pybind11 as the module `MassDiffusion3Dcpp`.

## 1. Scope

The scalar is transported by a fully developed laminar flow in a tube of unit radius, with axial velocity proportional to $1 - r^2$. The position is given in cylindrical coordinates $(x, r, \varphi)$, where $x$ is the axial coordinate ($x = 0$ at the inlet) and the cross-sectional Cartesian coordinates are $y = r\sin\varphi$, $z = r\cos\varphi$. The solution is expanded in eigenfunctions of the associated radial problem, and the inlet condition is projected onto them.

Two problems are available:

| Class | Inlet condition | Wall condition |
|---|---|---|
| `CDStratifiedSolution(zi, ui)` | Piecewise-constant in $z$: layers of value `ui`, separated at the interfaces `zi` | Zero flux (Neumann) |
| `CDGraetzIsothermalSolution(T0, T_wall)` | Uniform value `T0` | Fixed value `T_wall` (Dirichlet) |

Each solution can be set up in two ways:

* `setup_bare_solution()`: axial diffusion is neglected (Péclet number $\to \infty$).
* `setup_fp_solution(peclet)`: axial diffusion is included for the given finite Péclet number.

Once a solution has been set up, it can be evaluated at arbitrary points at negligible cost, for instance on whole cross-sections with `get_solution_at_planes`.

## 2. Requirements

| Component | Purpose |
|---|---|
| C++17 compiler with OpenMP, CMake ≥ 3.10 | Build |
| Python 3 with development headers (`Python.h`) | Build and run |
| Boost (header-only components: `boost/math`) | Build |
| `numpy`, `matplotlib` | Examples |
| A LaTeX installation with `dvipng` | LaTeX fonts in the examples (optional, see §6) |

pybind11 is included in `extern/pybind11`; it does not need to be installed. The code has been built and run on Ubuntu Linux with Python 3.12. On Debian-based systems:

```sh
sudo apt install build-essential cmake python3-dev libboost-dev python3-numpy python3-matplotlib
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super   # optional, for LaTeX fonts
```

## 3. Build

From the repository root:

```sh
sh build_project.sh
```

The script runs `cmake -S . -B build -DPYTHON_EXECUTABLE=$(command -v python3)` followed by `cmake --build build`. It produces the extension module in the repository root:

```
MassDiffusion3Dcpp.cpython-312-x86_64-linux-gnu.so
```

The module is built for the Python interpreter that CMake finds; the `cpython-312` tag in the file name identifies it. It can only be imported by the same minor version of Python. To build for a different interpreter, put that interpreter first in `PATH` (or activate its virtual environment) before running the script, and remove `build/CMakeCache.txt` if an earlier configuration is being reused.

## 4. Making the module importable

Python must be able to find `MassDiffusion3Dcpp...so`. Running a script does not add the repository root to the module search path (it only adds the directory of the script), so one of the following is required.

**(a) Per command.** From the repository root:

```sh
PYTHONPATH=. python3 examples/section_plots.py
```

**(b) Permanently.** Append the repository root to `PYTHONPATH` in the start-up file of your shell, using its **absolute** path:

```sh
# ~/.bashrc  (or ~/.zshrc for zsh)
export PYTHONPATH="/absolute/path/to/repository${PYTHONPATH:+:$PYTHONPATH}"
```

Then open a new terminal, or run `source ~/.bashrc`, and the examples can be run from any directory with `python3 examples/section_plots.py`. The relative path `.` should not be used in this file, since it would then refer to whichever directory the shell happens to be in.

**(c) From inside a script.** Insert `import sys; sys.path.insert(0, "/absolute/path/to/repository")` before the `import MassDiffusion3Dcpp` line.

## 5. Example: cross-sections of the stratified and Graetz solutions

[examples/section_plots.py](examples/section_plots.py) computes both solutions at finite Péclet number and displays one cross-section of each, side by side, with a common colour scale and iso-lines every 0.1. The figure is shown on screen and is not written to disk (use the save button of the plot window to export it, or replace `plt.show()` by `plt.savefig("sections.pdf")`).

```sh
PYTHONPATH=. python3 examples/section_plots.py
```

The run takes on the order of tens of seconds, dominated by the setup of the stratified solution.

The script is organised in the following steps:

1. **Evaluation points.** `make_triangulation()` ([examples/make_triangulation.py](examples/make_triangulation.py)) returns a Delaunay triangulation of the unit disk built from concentric rings of node spacing `h` (default `0.015`). The nodes are converted to $(r, \varphi)$ and the axial position `x0` is fixed.
2. **Stratified solution.** `layers_interfaces` are the $z$ coordinates of the interfaces and `layers_values` the values of the layers, listed from bottom to top; there is one more value than interfaces.
3. **Graetz solution.** `CDGraetzIsothermalSolution(T0, T_wall, verbose_level)`.
4. **Setup.** `setup_fp_solution(peclet)` for both solutions.
5. **Evaluation.** `get_solution_at_planes([x0], r, phi)` returns one list of values for each requested axial position; `[0]` selects the single position used here.
6. **Plot.** `tripcolor` with Gouraud shading and `tricontour` on the triangulation.

The parameters intended to be modified are:

| Parameter | Location | Description |
|---|---|---|
| `x0` | `section_plots.py` | Axial position of the section |
| `peclet` | `section_plots.py` | Péclet number. For large values the result approaches the bare solution; the difference grows as it decreases |
| `layers_interfaces`, `layers_values` | `section_plots.py` | Stratified inlet (one more value than interfaces, bottom to top) |
| `T0`, `T_wall` | `section_plots.py` | Graetz inlet and wall values |
| `h` | `make_triangulation(h=...)` | Mesh spacing; smaller values give a finer figure |

## 6. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `ModuleNotFoundError: No module named 'MassDiffusion3Dcpp'` | The module has not been built (§3), or its directory is not on `PYTHONPATH` (§4), or the Python version differs from the one used for the build. |
| `ImportError` mentioning an undefined symbol or an incompatible ABI | The module was built with a different Python than the one running the script. Rebuild (§3). |
| Errors mentioning `latex`, `dvipng` or `usetex` | Install the LaTeX packages of §2, or set `"text.usetex": False` in the `rcParams` line of `section_plots.py` to use the standard Matplotlib fonts. |
| `Python.h: No such file or directory` during the build | Install the Python development headers (`python3-dev`). |
| `boost/math/...: No such file or directory` during the build | Install Boost (`libboost-dev`). |
| No window appears (no display) | Replace `plt.show()` by `plt.savefig("sections.pdf")`. |

## 7. Repository layout

| Path | Content |
|---|---|
| `src/` | C++ sources: solution classes, eigenvalue and coefficient computations, finite-Péclet solver, pybind11 bindings (`pybind.cpp`) |
| `examples/` | Usage examples |
| `tests/` | C++ unit tests and Python interface tests |
| `extern/pybind11/` | pybind11 |
| `build_project.sh`, `CMakeLists.txt` | Build configuration |
