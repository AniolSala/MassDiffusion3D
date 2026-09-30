#ifndef FINITE_PECLET_COEFFICIENTS_GAUSSIAN_H
#define FINITE_PECLET_COEFFICIENTS_GAUSSIAN_H

// Gaussian finite-Peclet inlet coefficients.  This module is independent of
// the legacy composite-quadrature implementation.  The supplied weights
// already represent (1-r^2) r dr, so coefficient assembly never applies that
// factor again.  Each angular family solves its coupled fixed-weight Gram
// system from block-local mode values; no persistent radial table is made.
// Stratified caps use the same Gaussian geometry as the bare solution and
// reflect negative interfaces analytically.  Generalised norms are diagnostic
// data and are deliberately not coefficient normalisers.

#include <vector>

#include "series_term_struct.h"
#include "gram_method.h"
#include "rhs_method.h"
#include "finite_peclet_roots.h"
#include "finite_peclet_gram_gauss_jacobi.h"

/** Compute uniform-inlet Graetz coefficients.
 *  The first max_K entries must contain finite-Peclet spectral data.  Only
 *  coeff_fp is written.  The n>0 coefficients are zero.  Throws on invalid
 *  input or a rank/conditioning failure.  Independent calls are thread-safe
 *  provided the passed vectors are not concurrently mutated.
 *  gram_method selects the radial Gram backend (see gram_method.h); there is
 *  no fallback, so a failure of the selected backend propagates unchanged.
 *  cap_quad_margin is forwarded unmodified to finite_peclet_rhs.h's
 *  fp_rhs_uniform_inlet, which sizes EACH retained mode's own quadrature order
 *  individually from that mode's own bare root (fp_rhs_required_order =
 *  ceil(bare_root) + cap_quad_margin, bucketed) rather than from the n=0
 *  block's mode count (the only block the Graetz RHS -- a full-disk
 *  projection -- ever quadratures; Graetz has no interfaces to cap-project).
 *  This mirrors fp_stratified_coefficients_gauss_jacobi's rule (see
 *  finite_peclet_coefficients_gauss_jacobi.h) and is sized independently of
 *  the bare-mode radial rule, which this function never touches.
 *  When enable_timing is true, prints (to stdout) the time spent assembling
 *  the RHS load vector and the time spent assembling/factoring the Gram
 *  matrix (Graetz has a single n=0 block, so these are plain wall-clock times,
 *  not a parallel sum).
 *
 *  When projection_square_norm is non-null it receives chat^T b over the single
 *  n=0 block, which equals the quadratic form chat^T W chat = || f_K ||^2 of the
 *  internally scaled inlet because the coefficients solve W chat = b; see
 *  fp_stratified_coefficients_gauss_jacobi's own note and
 *  CDBaseSolution::get_inlet_projection_square_norm. Not written when max_K is
 *  zero.
 */
template <typename Ttype>
void fp_graetz_coefficients_gaussian(unsigned max_K,
                                     std::vector<SeriesTermData<Ttype>> &series_data,
                                     GramMethod gram_method,
                                     WallCondition wall_condition,
                                     const FPGramGaussJacobiOptions<Ttype> &gram_options,
                                     unsigned cap_quad_margin,
                                     RhsMethod rhs_method,
                                     unsigned rhs_representer_margin,
                                     bool enable_timing = false,
                                     Ttype *projection_square_norm = nullptr);

/** Compute piecewise-constant stratified-inlet coefficients using Gaussian caps.
 *  interface_positions are strictly increasing in (-1,1), layer_values has
 *  exactly one extra entry, and active modes are the first max_K terms.  Only
 *  coeff_fp is written, including the Neumann zero mode.  Throws on invalid
 *  geometry, spectral data, or Gram factorisation.  Blocks are independent and
 *  may run concurrently; complexity is O(sum K_n^2 N + A K N + sum K_n^3).
 */
template <typename Ttype>
void fp_stratified_coefficients_gaussian(const std::vector<Ttype> &interface_positions,
                                         const std::vector<Ttype> &layer_values,
                                         unsigned max_K,
                                         std::vector<SeriesTermData<Ttype>> &series_data,
                                         const std::vector<Ttype> &gauss_weights,
                                         const std::vector<Ttype> &gauss_nodes);

#endif
