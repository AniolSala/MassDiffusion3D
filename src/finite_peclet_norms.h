#ifndef FINITE_PECLET_NORMS_H
#define FINITE_PECLET_NORMS_H

#include <vector>

#include "series_term_struct.h"
#include "finite_peclet_radial_block.h"

// ---------------------------------------------------------------------------
// finite_peclet_norms — generalised norm and radial overlaps for the exact
// finite-Peclet solution (the "compute_norms" kernel).
//
// Theory: theory/analytical_solution.tex Sec. 5.3, eqs. (norm_fp), (U_nm_def).
//
// The rate-dependent orthogonality weight (eq. weight_fp) replaces the bare
// normalisation N^2 = <R,R>_omega by the GENERALISED norm
//     N^2_fp = int_0^1 [ (1 - r^2) + 2 kappa Lam ] R_nm(r)^2 r dr        (norm_fp)
//            = N^2 + 2 kappa Lam U_mm ,
// which is strictly positive for every mode (both weight terms are non-negative).
// The unweighted self-overlap U_mm = int_0^1 r R^2 dr and, more generally,
// U_km = int_0^1 r R_nm R_nk dr (eq. U_nm_def) are also provided; the overlap is
// used for the seed's alpha and for bi-orthogonality CHECKS in the tests, not for
// coefficient assembly (the modes decouple, Sec. 5.3).
//
// Self-contained composite Gauss-Legendre quadrature (finite_peclet_quadrature.h).
// Instantiated for double and long double.
// ---------------------------------------------------------------------------

// Generalised norm N^2_fp of eq. (norm_fp).
// @param n      angular index.
// @param b      = sqrt(Lam) (unshifted radial parameter).
// @param bt     = b*(1+kappa*Lam) (shifted Kummer carrier).
// @param kappa  Pe^-2.
// @param Lam    the mode's decay rate (= b^2).
// @param panels quadrature panels (default 64).
// @return       N^2_fp > 0. Reduces to the bare weighted norm when kappa -> 0.
template <typename Ttype>
Ttype fp_generalized_norm(const unsigned &n, const Ttype &b, const Ttype &bt,
                          const Ttype &kappa, const Ttype &Lam, unsigned panels = 64u);

// Unweighted radial overlap U_km = int_0^1 r R_nm R_nk dr (eq. U_nm_def).
// Pass identical (b, bt) pairs for the diagonal self-overlap U_mm.
template <typename Ttype>
Ttype fp_unweighted_overlap(const unsigned &n,
                            const Ttype &bi, const Ttype &bti,
                            const Ttype &bj, const Ttype &btj,
                            unsigned panels = 64u);

// Orchestrator: fill norm_fp for every retained mode from its finite-Péclet
// eigen-data (root_fp/btilde_fp/rate_fp), computed once per Péclet. Parallelised
// over modes (fully independent). Keeps the norm complexity out of CDBaseSolution.
template <typename Ttype>
void fp_compute_norms(std::vector<SeriesTermData<Ttype>> &series_data, unsigned max_K,
                      const Ttype &kappa, unsigned panels = 64u);

template <typename Ttype>
Ttype fp_generalized_norm_from_row(const FPQuadratureRule<Ttype> &quadrature,
                                   const Ttype *radial_row, const Ttype &kappa,
                                   const Ttype &lambda);

template <typename Ttype>
Ttype fp_unweighted_overlap_from_rows(const FPQuadratureRule<Ttype> &quadrature,
                                      const Ttype *row_i, const Ttype *row_j);

template <typename Ttype>
void fp_compute_norms_from_table(std::vector<SeriesTermData<Ttype>> &series_data,
                                 unsigned max_K, const Ttype &kappa,
                                 const FPRadialTable<Ttype> &table);

#endif // FINITE_PECLET_NORMS_H
