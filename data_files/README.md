# Data files

These data are released under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Variables follow the article: `x` is the axial coordinate scaled with the entry length (`x = x_phys / (Pe R)`), `x* = Pe x` is the axial distance in pipe radii, `kappa = Pe^-2`, `beta_K` is the largest retained bare eigenvalue, and `K` is the number of retained terms.

## Analytical solution

These files are written by the scripts in [`../data_scripts`](../data_scripts), which also read them back (see the section *Cached data* of the [scripts README](../data_scripts/README.md)).

Every `.npz` archive holds a scalar array `configuration`, a JSON string with the settings the data were computed with.

**`radial_functions.npz`** (Fig. 1a, `plot_radial_functions.py`). The radii `r` (800 points on [0, 1]) and, for every plotted mode, the normalised radial function `R_nm(r) / N_nm` in arrays named `<wall>_n<n>_m<m>_<curve>`, with `wall` = `Neumann` (zero flux) or `Dirichlet` (isothermal) and `curve` = `bare` (`kappa = 0`) or `kappa<value>` (`kappa = 0.01, 0.1, 1, 10`).

**`parseval_identity_error_finite_peclet.npz`** (Fig. 2, `parseval_identity_error_finite_peclet.py`). Relative squared truncation error of the inlet projection, `||f - f_K||^2 / ||f||^2` in `L^2_omega`. The array `keys` lists the inlets (`stratified2`, `stratified3`, `stratified4`, `graetz`); for each of them, `<key>_beta_max` holds the cutoffs `beta_K`, `<key>_bare` the error of the bare coefficients and `<key>_Pe<value>` that of the finite-Péclet coefficients at `Pe = 10, 100, 1000, 10000`.

**`plot_solution_3D.npz`** (Fig. 5, `plot_solution_3D.py`). The evaluation points on the cross-section, as Cartesian (`y`, `z`) and polar (`r`, `phi`) coordinates, their triangulation `triangles`, the axial positions `x`, and the fields `graetz` and `stratified` (one row per axial position, one column per point), both at `Pe = 5`.

**`flow_equation_data.npz`** (Fig. 1b, `solve_flow_equation.py`). A NumPy archive with one row per mode (70 rows: 7 angular indices `n` times 5 radial indices `m`, for each wall condition):

| Array | Shape | Content |
|---|---|---|
| `configuration` | scalar | JSON string with the settings the data were computed with |
| `problem` | (70,) | `stratified` (zero-flux wall) or `graetz` (isothermal wall) |
| `n`, `m` | (70,) | angular and radial index of the mode |
| `beta`, `bessel_zero` | (70,) | bare eigenvalue `beta_nm` and Bessel zero `q_nm` of the mode |
| `kappa` | (70, 121) | `kappa` grid, including `kappa = 0` |
| `rate` | (70, 121) | decay rate `Lambda_nm(kappa)` on that grid |

The remaining arrays (`index`, `alpha`, `base_roots`, `alpha_row`, `row_size`, `validation_kappas`, `validation_gaps`) are bookkeeping used by the script.

**`nusselt_curves.csv`** (Fig. 3, `nusselt_curves.py`). Local Nusselt number `Nu_K(x)` with `beta_K = 400`. Column `x`, then one column per Péclet number (`Pe_1` … `Pe_1000`), computed with the asymptotic tail of the article (finite Péclet, radial projection), and `bare`, the classical series (`kappa = 0`) summed without the tail.

**`nusselt_developed.csv`** and **`nusselt_developed.tex`** (`nusselt_curves.py`). Fully developed Nusselt number for each Péclet number (`computed`), next to the values of Shah and London (1978) where they exist (`Shah_London_1978`). The `.tex` file holds the same numbers as a LaTeX table.

**`nusselt_classical_table.csv`** (Table S2, `nusselt_curves.py`). Comparison of the classical series (`kappa = 0`, `beta_K = 400`, without the tail) with the values of Shah and London (1978): their abscissa `x_hat` (`x = 2 x_hat`), the abscissa `x`, their value `Shah_London_1978`, the present value `present`, and `1e3_Delta_rel` = 10³ (`Shah_London_1978` − `present`) / `present`.

**`nusselt_at_decades.json`** (`nusselt_curves.py`). `Nu_K(x)` at `x = 1e-4, 1e-3, 1e-2, 1e-1, 1` for each Péclet number, with the settings used (`max_root`, `evaluation`, `projection_space`).

**`nusselt_convergence_plain_vs_split.csv`** (Fig. 4, `nusselt_convergence.py`). One row per solve, with columns `Pe`, `x`, `beta` (`beta_K`), `K`, `Nu_plain` (plain truncation of the series) and `Nu_split` (with the asymptotic tail). The rows with `beta = 1600` are the reference of the figure.

## Finite-element comparison (folder `fem/`)

The finite-element solutions were computed with Kratos Multiphysics (ConvectionDiffusionApplication) on a pipe of radius 1 and length 2, meshed with linear tetrahedra of size `h = 0.1, 0.05, 0.02`, for `Pe = 10, 100, 1000` and both problems (`stratified_layers`, the stratified inlet with a zero-flux wall, and `graetz`, the isothermal Graetz problem). The HDF5 files store their full run configuration as a JSON string in the root attribute `metadata_json`. In that configuration and in the files of the timing benchmark, file paths are reduced to file names and the host name is removed; everything else is as recorded.

**`bare_vs_finite_peclet_vs_beta_squared_8b6d347e6c916693.h5`** (Fig. 6). Error between the truncated series and the finite-element solution on the cross-section `x* = 1`, against the truncation (20 values of `beta_K` from 12 to 400):

- `<problem>/beta_k_squared`: the values of `beta_K^2`.
- `<problem>/peclet_<i>/mesh_<j>/<variant>/l2` and `.../linf`: the errors in the two norms of the article, for `variant` = `bare` (classical series) or `finite_peclet`. The attributes of `peclet_<i>` and `mesh_<j>` give the Péclet number and the mesh size.
- `section_fields`: the cross-section fields of Fig. 6b (at `beta_K = 400` and `h = 0.02`): node coordinates `r` and `phi`, and, under `<problem>/peclet_<i>`, the fields `bare`, `finite_peclet` and `fem`.
- `fem_element_counts`: number of elements of each mesh.

**`bare_vs_finite_peclet_vs_x_97012537d5087dad.h5`** (Fig. S1). The same errors against the axial position, at fixed truncation: dataset `x_physical` (the positions `x*`) and `<problem>/peclet_<i>/mesh_<j>/<variant>/{l2, linf}`.

**`benchmark_03af62a81f84e4c3/`** (Figs. S2 and S3, Table S1). Wall-time and memory benchmark of the finite-element and the series solutions:

- `results.json`: every timed run (setup time `T1` and evaluation time `T2`, per problem, Péclet number, mesh and number of threads), with the per-cell summaries, the thread scaling and the errors of the compared fields.
- `benchmark_memory.json`: peak memory of each run, in MiB, with the definitions of the measured quantities.
- `benchmark_configuration.json`: the configuration fixed before the runs (finite-element settings, truncation of the series, compared scenarios).
- `manifest.json`: the machine, the meshes and the processes of the benchmark.
