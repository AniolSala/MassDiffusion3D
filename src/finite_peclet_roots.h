#ifndef FINITE_PECLET_ROOTS_H
#define FINITE_PECLET_ROOTS_H

// ---------------------------------------------------------------------------
// finite_peclet_roots — decay-rate root finding for the exact finite-Peclet
// solution (the "modify_roots" kernel).
//
// Theory: theory/analytical_solution.tex Sec. 5.2 (characteristic equations
// char_dirichlet_fp / char_neumann_fp), Sec. 5.4 (flow equation, monotonicity),
// Sec. 5.5 (closed-form seed, eq. seed_root).
//
// For each angular index n, the axial decay rates Lam_{nm}(kappa) are the ordered
// positive roots of a SCALAR transcendental equation F^{D/N}_n(Lam) = 0. There is
// no matrix pencil and no coupling across m or n: the finite-Peclet modes are
// genuinely decoupled (Sec. 5.3). The theory further guarantees the invariants
// the ladder solver relies on and validates against:
//   * reality + simplicity of the spectrum      (eq. rate_energy_quadratic),
//   * strict ordering  Lam_{nm}(kappa) < beta_{nm}^2   (eq. rate_ordering),
//   * monotone decrease of every rate in kappa   (eq. flow_equation).
//
// All heavy computation lives here (requirement: not in CDBaseSolution or the
// derived classes). Instantiated for double and long double.
// ---------------------------------------------------------------------------

#include <vector>

#include "series_term_struct.h"

// Wall boundary condition for the radial eigenproblem.
//   Neumann   -> R'(1) = 0   (stratified profile, zero-flux walls)
//   Dirichlet -> R(1)  = 0   (isothermal Graetz, fixed-wall scalar)
enum class WallCondition { Neumann = 0, Dirichlet = 1 };

// Characteristic function F(n, Lam; kappa, wall). Its positive zeros are the axial
// decay rates. Uses the modified radial wall values (psi_at_1_fp / dpsi_dr_at_1_fp).
template <typename Ttype>
Ttype fp_char(const unsigned &n, const Ttype &Lam, const Ttype &kappa, WallCondition wall);

// Closed-form seed for the (m)-th rate (eq. seed_root):
//   Lam_seed = 2 beta^2 / (1 + sqrt(1 + 4 kappa alpha beta^2)).
// base_beta2 = beta_{nm}^2 (the kappa=0 rate). alpha = alpha_{nm} = U^n_mm / N_nm^2
// (eq. seed_root), the exact bare-mode ratio between the diagonal of the unweighted
// pseudo-products matrix and the weighted bare norm — supplied by the caller, not
// guessed here. Warm-starts Newton only; the ladder solver's correctness comes from
// bracketing, so a poor alpha/seed cannot mislead it.
template <typename Ttype>
Ttype fp_seed(const unsigned &n, unsigned m, const Ttype &base_beta2,
              const Ttype &kappa, const Ttype &alpha, WallCondition wall);

// Solve for ONE rate inside a verified bracket [lo, hi] (sign change required):
// safeguarded Newton + bisection, derivative by central differences.
template <typename Ttype>
Ttype fp_solve_bracketed(const unsigned &n, const Ttype &kappa, WallCondition wall,
                         Ttype lo, Ttype hi, const Ttype &rel_tol, unsigned max_iter);

// Overload with iteration tracking: writes the number of iterations taken into final_iters.
// If seed is valid and within (lo, hi), it is used to warm-start Newton.
template <typename Ttype>
Ttype fp_solve_bracketed(const unsigned &n, const Ttype &kappa, WallCondition wall,
                         Ttype lo, Ttype hi, const Ttype &rel_tol, unsigned max_iter,
                         unsigned &final_iters,
                         Ttype seed = static_cast<Ttype>(-1));

// Solve the ordered set of rates for angular index n.
// @param n            angular index.
// @param base_beta2   ascending kappa=0 rates beta^2; used as strict upper bounds
//                     (eq. rate_ordering) and to size the bracketing scan.
// @param kappa        Pe^-2 (>= 0).
// @param wall         Neumann / Dirichlet.
// @param skip_zero_mode  true when Lam = 0 is an exact eigenvalue (Neumann, n = 0):
//                     out[0] is set to 0 and the scan starts above it.
// @param rel_tol      relative tolerance for each bracketed solve.
// @param max_iter     max iterations per bracketed solve.
// @return             ascending Lam_{nm}(kappa), same length as base_beta2.
// Throws std::runtime_error if a produced rate violates ordering/bound checks.
// @param alpha_by_m   alpha_{nm} = U^n_mm / N_nm^2 per retained m (eq. seed_root),
//                     aligned with base_beta2; used only to seed Newton in
//                     fp_solve_bracketed. Empty (default), or an entry missing for
//                     a given m, falls back to a representative constant — still
//                     just a seed, so the ladder remains exact either way.
template <typename Ttype>
std::vector<Ttype> fp_solve_ladder(const unsigned &n,
                                   const std::vector<Ttype> &base_beta2,
                                   const Ttype &kappa,
                                   WallCondition wall,
                                   bool skip_zero_mode,
                                   const Ttype &rel_tol,
                                   unsigned max_iter,
                                   const std::vector<Ttype> &alpha_by_m = std::vector<Ttype>());

// Overload with iteration tracking: writes the number of iterations taken per root into final_iters.
template <typename Ttype>
std::vector<Ttype> fp_solve_ladder(const unsigned &n,
                                   const std::vector<Ttype> &base_beta2,
                                   const Ttype &kappa,
                                   WallCondition wall,
                                   bool skip_zero_mode,
                                   const Ttype &rel_tol,
                                   unsigned max_iter,
                                   std::vector<unsigned> &final_iters,
                                   const std::vector<Ttype> &alpha_by_m = std::vector<Ttype>());

// Orchestrator: replace every retained mode's finite-Péclet eigen-data in place.
// Groups m_series_data by angular index n (contiguous, ascending m, as produced by
// set_series_data), solves each n-ladder with fp_solve_ladder, and writes
// root_fp = sqrt(Λ), btilde_fp = b(1+κΛ), rate_fp = Λ. The zero mode (base rate ≈ 0)
// is detected automatically and kept at Λ = 0. Parallelised over n (see the
// parallelisation report); the m-ladder inside each n stays sequential.
// This keeps all root-finding complexity out of CDBaseSolution.
// @param pseudo_products_matrix  [n][m1][m2] = U^n_{m1,m2} = int_0^1 R_{n,m1} R_{n,m2}
//                     r dr (the tabulated UNWEIGHTED overlap read via the solution's
//                     set_pseudo_products_matrix()). Combined with series_data[k].norm
//                     (the weighted N_nm^2) to form the exact seed ratio alpha_nm =
//                     U^n_mm / N_nm^2 of eq. seed_root — no extra computation, purely
//                     a lookup. A mode outside the tabulated range falls back to a
//                     representative alpha (see fp_solve_ladder); this only affects
//                     the Newton warm-start, never correctness.
template <typename Ttype>
void fp_modify_roots(std::vector<SeriesTermData<Ttype>> &series_data, unsigned max_K,
                     const Ttype &kappa, WallCondition wall,
                     const Ttype &rel_tol, unsigned max_iter,
                     const std::vector<std::vector<std::vector<Ttype>>> &pseudo_products_matrix);

#endif // FINITE_PECLET_ROOTS_H
