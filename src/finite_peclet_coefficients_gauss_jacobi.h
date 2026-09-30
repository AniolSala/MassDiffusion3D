#ifndef FINITE_PECLET_COEFFICIENTS_GAUSS_JACOBI_H
#define FINITE_PECLET_COEFFICIENTS_GAUSS_JACOBI_H

#include <vector>

#include "series_term_struct.h"
#include "gram_method.h"
#include "rhs_method.h"
#include "finite_peclet_roots.h"
#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_rhs_representer.h"

/** Assemble finite-Peclet stratified coefficients using a fixed Gauss-Jacobi
 * cap rule. cap_quad_margin is forwarded, unmodified, to finite_peclet_rhs.h's
 * fp_rhs_stratified_inlet, which sizes EACH mode's own cap/full-disk
 * quadrature order individually from that mode's own bare root
 * (fp_rhs_required_order = ceil(bare_root) + cap_quad_margin, bucketed) rather
 * than from the whole angular block's mode count: unlike the Gram backend's
 * K-dependent node rule (fp_gram_node_count in finite_peclet_gram_gauss_jacobi.h),
 * which sizes a K-by-K linear system and so must guard against rank
 * deficiency, the cap/full-disk integrals are independent per-mode 1-D
 * integrals -- sizing them by block size would over-resolve every mode less
 * oscillatory than the block's worst one. This function itself no longer
 * computes any order or block-size aggregate; see finite_peclet_rhs.h for
 * where that happens. The resulting per-mode order sizes both the circular-
 * cap rule (Appendix B) for genuine 0<a<1 interfaces and the full-disk
 * projection used for the baseline / center-cap / negative-interface-at-n=0
 * terms. This function never touches the bare-mode radial rule
 * (n_gauss_points) -- that quadrature is reserved for the bare solution only,
 * so the two RHS assemblies cannot mix. gram_method selects the radial Gram
 * backend used for the per-angular-block solve (see gram_method.h); there is no fallback between
 * backends, so an Ultraspherical failure propagates instead of retrying with
 * GaussJacobiQR. wall_condition is forwarded to the selected backend's
 * boundary/normalization convention. When enable_timing is true, prints (to
 * stdout) the time spent on each of the three passes -- Gram matrix assembly,
 * RHS load-vector assembly, and the per-block linear solve -- each summed over
 * every angular block (the per-block loop is OpenMP-parallelized, so this is
 * aggregate work time, not wall-clock, whenever more than one thread is used).
 *
 * The RHS/Gram assembly and evaluation (finite_peclet_rhs.h /
 * finite_peclet_gram_gauss_jacobi.h) are themselves shared, unmodified, with
 * fp_graetz_coefficients_gaussian (finite_peclet_coefficients_gaussian.h): the
 * only difference between the two solvers is the modified root values passed
 * in via `series_data`.
 *
 * When projection_square_norm is non-null it receives the contraction of the
 * solved coefficients with the load vector, sum_n sum_m chat_nm b^n_m. Because
 * the coefficients solve W^n chat^n = b^n, that scalar IS the quadratic form
 * sum_{n,m,k} chat_nm W^n_mk chat_nk = || f_K ||^2 in the projection's own norm,
 * obtained without assembling any Gram matrix: one fused multiply-add per
 * retained mode, against an assembly pass that costs seconds. It belongs to the
 * inlet as passed in, i.e. to the internally scaled profile; see
 * CDBaseSolution::get_inlet_projection_square_norm for the conversion back to the
 * user's scaling. Not written when max_K is zero. */
template <typename Ttype>
void fp_stratified_coefficients_gauss_jacobi(
    const std::vector<Ttype> &interface_positions,
    const std::vector<Ttype> &layer_values, unsigned max_K,
    std::vector<SeriesTermData<Ttype>> &series_data,
    unsigned cap_quad_margin,
    GramMethod gram_method,
    WallCondition wall_condition,
    const FPGramGaussJacobiOptions<Ttype> &gram_options,
    RhsMethod rhs_method,
    unsigned rhs_representer_margin,
    bool enable_timing = false,
    Ttype *projection_square_norm = nullptr);

#endif
