#ifndef FINITE_PECLET_COEFFICIENTS_RADIAL_H
#define FINITE_PECLET_COEFFICIENTS_RADIAL_H

#include <vector>

#include "series_term_struct.h"
#include "rhs_method.h"
#include "finite_peclet_roots.h"
#include "finite_peclet_gram_radial.h"
#include "finite_peclet_rhs_radial.h"
#include "finite_peclet_rhs_representer_radial.h"

// ---------------------------------------------------------------------------
// Finite-Peclet inlet coefficients by Galerkin projection in L2_r, the
// alternative to finite_peclet_coefficients_gauss_jacobi.h's projection in
// L2_omega. Both solve one dense system per angular block,
//
//     sum_k A^n_{mk} chat_{nk} = b^n_m ,
//
// and differ ONLY in the weight that defines A and b:
//
//   projection | Gram entry                              | load vector
//   -----------+-----------------------------------------+---------------------------------
//   "w"        | W^n_{mk} = int_0^1 omega R_nm R_nk r dr | b^n_m = int_Omega omega f R_nm Phi_n
//   (default)  |                                         |
//   "r" (here) | U^n_{mk} = int_0^1       R_nm R_nk r dr | b^n_m = int_Omega       f R_nm Phi_n
//
// with omega(r) = 1 - r^2.
//
// THE s = r^2 REDUCTION. Writing R_nm(r) = r^n G_nm(s) with s = r^2 (so that
// fp_radial_factor IS G_nm, and psinm_r_fp(n,b,bt,r) == r^n fp_radial_factor(n,b,bt,r*r)):
//
//     W^n_{mk} = 1/2 int_0^1 (1-s) s^n G_m G_k ds ,
//     U^n_{mk} = 1/2 int_0^1       s^n G_m G_k ds ,
//
// so the entire difference between the two Gram matrices is the single factor
// (1-s) -- i.e. the Jacobi parameter alpha of the shifted-Jacobi quadrature
// measure, which moves from (alpha, beta) = (1, n), total mass
// 1/((n+1)(n+2)), to (0, n), total mass 1/(n+1). Everything else in the two
// backends is identical.
//
// WHAT EACH PROJECTION BUYS. L2_omega is the norm in which the BARE modes are
// orthogonal, and in which the inlet residual is classically measured -- it is
// the natural, and the default, choice. But the MODIFIED (finite-Peclet) modes
// are a Riesz basis of L2_r, not of L2_omega: because omega vanishes at the
// wall, multiplication by omega is positive but not boundedly invertible, so
// lambda_min of the normalised W^n decays like 1/K (measured ~1.2/K) while
// that of the normalised U^n is flat in K (measured 0.747 at Pe = 5, 0.548 at
// 31.6, 0.514 at 200). The r-projection is therefore the one whose Gram
// conditioning does not degrade with truncation.
//
// Neither is "more correct": for the same inlet the two return DIFFERENT
// coefficient vectors, each the best approximation from the same span in its
// own norm. Comparing them entry by entry is meaningless; compare invariants
// (see tests/test_fp_radial_projection.cpp).
//
// BOTH RHS BACKENDS ARE AVAILABLE HERE. RhsMethod::Representer was originally
// absent on this path, which cost it a factor of four in setup at production
// truncations -- the entire gap between the two projections was that one
// missing backend, not the projection: measured at max_root = 600, the RHS pass
// takes 0.06 s by representer against 45 s by direct quadrature, while the
// radial Gram assembly is marginally FASTER than the weighted one (10.89 s
// against 11.25 s). finite_peclet_rhs_representer_radial.h now supplies it; the
// construction is a near-copy of the weighted one because, by the lemma quoted
// there, both projections expand the same exact representer and differ only in
// the measure of that expansion.
//
// KNOWN LIMITATION of the L2_r path (documented, not a bug; rejected loudly at
// setup time rather than silently falling back -- see
// CDBaseSolution::setup_fp_solution): GramMethod::Ultraspherical does not carry
// over, because its factor is built for the (1, n) measure. Hence these
// functions take an RhsMethod but no GramMethod.
// ---------------------------------------------------------------------------

/** Assemble finite-Peclet stratified coefficients by L2_r projection.
 *
 * Structured exactly like fp_stratified_coefficients_gauss_jacobi: Pass 1 Gram,
 * Pass 2 RHS, Pass 3 solve, each an OpenMP parallel loop over angular blocks.
 * Gram-first ordering is deliberate -- it makes the n=0 block's exact
 * constant-mode column U^0_{:,0} available to the RHS pass, which turns it into
 * the exact disk-average term at zero extra Kummer evaluations.
 *
 * cap_quad_margin is forwarded unmodified to finite_peclet_rhs_radial.h, which
 * sizes EACH mode's own cap/full-disk quadrature order individually from that
 * mode's own bare root (fp_rhs_required_order, shared with the weighted path)
 * rather than from the angular block's mode count. This function never touches
 * the bare-mode radial rule (n_gauss_points), whose stored weights already bake
 * in omega and would silently reintroduce it here.
 *
 * wall_condition is accepted for signature symmetry with the weighted driver;
 * the radial Gram backend, like the Gauss-Jacobi one, is boundary-blind and
 * infers the exact constant mode from rate_fp == 0 alone.
 *
 * rhs_method selects the load-vector backend (see rhs_method.h). Representer
 * needs the Gram assembly's raw samples, so this function turns on
 * retain_samples for that method only, and releases them as soon as each
 * block's load vector has been formed. There is no fallback between backends.
 *
 * When enable_timing is true, prints the time spent on each of the three
 * passes, summed over every angular block (the loops are parallel, so this is
 * aggregate work time, not wall clock). */
template <typename Ttype>
void fp_stratified_coefficients_radial(const std::vector<Ttype> &interface_positions,
                                       const std::vector<Ttype> &layer_values, unsigned max_K,
                                       std::vector<SeriesTermData<Ttype>> &series_data,
                                       unsigned cap_quad_margin, WallCondition wall_condition,
                                       const FPGramRadialOptions<Ttype> &gram_options,
                                       RhsMethod rhs_method,
                                       unsigned rhs_representer_margin,
                                       bool enable_timing = false,
                                       Ttype *projection_square_norm = nullptr);

/** When projection_square_norm is non-null, both functions above and below write
 *  chat^T b into it. That scalar is the quadratic form chat^T U chat of the Gram
 *  matrix this path actually solves with, i.e. || f_K ||^2 in L2_r and NOT in
 *  L2_omega -- the two are different numbers for the same inlet.
 *  CDBaseSolution::get_inlet_projection_square_norm converts this radial norm
 *  to the user's inlet scaling. Not written when max_K is zero. */

/** Assemble uniform-inlet (Graetz) finite-Peclet coefficients by L2_r
 *  projection. Only coeff_fp is written; the n > 0 coefficients are zero
 *  (the uniform inlet has no angular content). Graetz has a single n = 0 block
 *  and no parallel loop over blocks, so the timings are plain wall clock. */
template <typename Ttype>
void fp_graetz_coefficients_radial(unsigned max_K,
                                   std::vector<SeriesTermData<Ttype>> &series_data,
                                   WallCondition wall_condition,
                                   const FPGramRadialOptions<Ttype> &gram_options,
                                   unsigned cap_quad_margin,
                                   RhsMethod rhs_method,
                                   unsigned rhs_representer_margin,
                                   bool enable_timing = false,
                                   Ttype *projection_square_norm = nullptr);

#endif // FINITE_PECLET_COEFFICIENTS_RADIAL_H
