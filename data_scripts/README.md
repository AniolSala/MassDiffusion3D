# Scripts of the article

This folder holds the scripts that reproduce the figures and tables of the article *Exact Analytical Solution of the Extended Graetz Problem with Axial Diffusion and Step-Wise Boundary Conditions* (A. Sala Pascual and G. Casas González). The data they produce are in [`../data_files`](../data_files), and the figures are written to `../figures`. The figures that compare the solution with finite-element computations come with their data only (see [Finite-element data](#finite-element-data)).

## Requirements

- The `MassDiffusion3Dcpp` module of this repository, built and importable as described in sections 3 and 4 of the [repository README](../README.md). The scripts read the root tables from `src/` of this repository.
- Python 3 with NumPy and Matplotlib. `plot_solution_3D.py` also needs SciPy.
- A LaTeX installation: the labels are typeset with Matplotlib's `text.usetex`. In every script except `nusselt_convergence.py`, setting `USE_TEX` (or `USE_LATEX`) to `False` draws the figure without LaTeX, with slightly different lettering.
- `h5py`, only to read the finite-element data files.

## Running

Each script runs on its own. For example, from the root of the repository (the `PYTHONPATH=.` prefix is option (a) of section 4 of the repository README; with option (b) the scripts can be run from any directory):

```bash
PYTHONPATH=. python3 data_scripts/nusselt_curves.py
```

Every script writes its figure to `figures/` and its numerical data to `data_files/`, both at the root of the repository, whatever the working directory. All paths are built from the location of the script, and all settings are module-level constants at the top of each script; there are no command-line arguments.

## Figures, tables and their sources

| Article | Script | Output in `figures/` | Data in `data_files/` |
|---|---|---|---|
| Fig. 1a | `plot_radial_functions.py` | `Figure_1a.pdf` | `radial_functions.npz` |
| Fig. 1b | `solve_flow_equation.py` | `Figure_1b.pdf`, `.png` | `flow_equation_data.npz` |
| Fig. 2 | `parseval_identity_error_finite_peclet.py` | `Figure_2.pdf`, `.png` | `parseval_identity_error_finite_peclet.npz` |
| Fig. 3 | `nusselt_curves.py` | `Figure_3.pdf` | `nusselt_curves.csv`, `nusselt_developed.csv`, `nusselt_developed.tex`, `nusselt_at_decades.json` |
| Fig. 4 | `nusselt_convergence.py` | `Figure_4.pdf` | `nusselt_convergence_plain_vs_split.csv` |
| Fig. 5 | `plot_solution_3D.py` | `Figure_5.pdf`, `.png` | `plot_solution_3D.npz` |
| Fig. 6 | – | – | `fem/bare_vs_finite_peclet_vs_beta_squared_8b6d347e6c916693.h5` |
| Fig. S1 | – | – | `fem/bare_vs_finite_peclet_vs_x_97012537d5087dad.h5` |
| Figs. S2, S3, Table S1 | – | – | `fem/benchmark_03af62a81f84e4c3/` |
| Table S2 | `nusselt_curves.py` | – | `nusselt_classical_table.csv` |

Figures S1–S3 and Tables S1–S2 are in the Supplementary Material of the article. The data files are described in [data_files/README.md](../data_files/README.md).

## Cached data

Every script stores the data it plots in `data_files/`, and on the next run redraws its figure from that file instead of computing the solution again. To recompute, set `FORCE_RECOMPUTE = True` at the top of the script, or delete its data file. The `.npz` files also store the settings they were computed with, and a script ignores a file whose settings differ from its own. Two scripts work slightly differently:

- `nusselt_curves.py` checks only that the columns of `data_files/nusselt_curves.csv` match its Péclet numbers.
- `nusselt_convergence.py` stores every solve in `data_files/nusselt_convergence_plain_vs_split.csv` as soon as it finishes, and computes only the rows that are missing. Delete the file to recompute everything; the reference at `REFERENCE_BETA = 1600` is the expensive part.

Computing Fig. 2 from scratch takes about five minutes, and Figs. 1a, 1b and 5 less than half a minute each. The full recomputations of Figs. 3 and 4 were not timed.

## Licence

The scripts in this folder are covered by the BSD-3-Clause licence of the repository ([LICENSE](../LICENSE)). The data in `data_files/` are released under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/): they may be shared and adapted for any purpose, provided the article and this repository are credited.

## Finite-element data

Figure 6, Figures S1–S3 and Table S1 compare the series with finite-element solutions computed with Kratos Multiphysics (ConvectionDiffusionApplication). The finite-element computations need Kratos, its case files and the meshes, so their scripts are not included; their results are in `data_files/fem/`.
