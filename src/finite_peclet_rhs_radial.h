#ifndef FINITE_PECLET_RHS_RADIAL_H
#define FINITE_PECLET_RHS_RADIAL_H

#include <vector>
#include "series_term_struct.h"
#include "finite_peclet_roots.h"   // WallCondition
#include "finite_peclet_rhs.h"     // FPGaussJacobiRule / FPCapKernel / FPCapRuleCache
                                   // + make_fp_gauss_jacobi_rule / fp_rhs_required_order
                                   // + fp_rhs_build_rule_cache (the (0,1/2) cap rule)

// ---------------------------------------------------------------------------
// Inlet/boundary-condition-specific load vectors ("RHS") for the L2_r
// finite-Peclet Galerkin system  U^n c^n = b^n, the r-weighted counterpart of
// finite_peclet_rhs.h's L2_omega system  W^n c^n = b^n.
//
//     L2_omega (finite_peclet_rhs.h)  : b^n_m = int_Omega omega f R_nm Phi_n dOmega
//     L2_r     (this module)          : b^n_m = int_Omega       f R_nm Phi_n dOmega
//
// This module owns everything that depends on the inlet profile; the radial
// Gram module (finite_peclet_gram_radial.h) is inlet-blind and must never be
// edited to accommodate a new inlet -- add a new builder here instead.
//
// WHERE omega ENTERS, AND WHERE IT DOES NOT. Re-deriving Appendix B's cap
// reduction with weight r dr instead of omega r dr: with s = r^2,
// s = a^2 + Tm*eta and Tm = 1 - a^2, the interface-cap integral over the cap
// z > a reads
//
//   n == 0 :  b_{0m} = (1/sqrt(2pi)) int_{a^2}^1 [omega] theta(s) G(s) ds
//   n >= 1 :  b_{nm} = (1/(n sqrt(pi))) int_{a^2}^1 [omega] sqrt(Tm*eta)
//                                        q_{n-1}(a,s) G(s) ds
//
// with theta(s) = atan2(sqrt(s-a^2), a) the cap half-angle and q_{n-1} the
// (a,s)-polynomial form of s^{(n-1)/2} U_{n-1}(a/sqrt(s)). Dropping the
// bracketed omega = Tm(1-eta) removes exactly ONE power of Tm from the
// prefactor and the factor (1-eta) from the kernel, and nothing else:
//
//   n == 0 :  prefactor  Tm^2 / sqrt(2pi)          ->  Tm / sqrt(2pi)
//   n >= 1 :  prefactor  Tm^2 sqrt(Tm) /(n sqrt(pi)) -> Tm sqrt(Tm) /(n sqrt(pi))
//   both   :  kernel     w_i (1-eta) * [...]       ->  w_i * [...]
//
// The eta^{1/2} cap measure ((alpha, beta) = (0, 1/2)) is UNCHANGED: it
// absorbs the 1/sqrt(eta) geometric singularity of the cap parameterisation,
// which has nothing to do with omega. The cap rule and its cache
// (make_fp_gauss_jacobi_rule / fp_rhs_build_rule_cache) are therefore reused
// from finite_peclet_rhs.h verbatim.
//
// The FULL-DISK rule does change: int_0^1 R_{0m} r dr = 1/2 int_0^1 G(s) ds is
// a shifted-LEGENDRE integral, (alpha, beta) = (0, 0), not the weighted path's
// (1, 0). fp_rhs_radial_build_full_disk_rule_cache is the only new cache
// builder needed, and the weighted path's fp_rhs_build_full_disk_rule_cache
// must NEVER be passed to a function in this header (nor vice versa).
//
// KNOWN LIMITATIONS (do not file as bugs):
//  * There is no ultraspherical Gram backend for L2_r: that factor is built for
//    the (1, n) measure. The combination is rejected loudly at setup time (see
//    CDBaseSolution::setup_fp_solution) rather than silently falling back.
//    (The representer RHS backend, which was also absent here originally, now
//    exists as finite_peclet_rhs_representer_radial.*: the L2_omega and L2_r
//    cap functionals turn out to share the SAME exact representer, so only the
//    polynomial family and two constants change.)
//  * The bare-mode Gaussian rule (m_gauss_weights / m_gauss_points) already
//    represents (1-r^2) r dr, so it must NEVER be reused on this path. Nothing
//    in this module touches it; the n=0, a=0 "center" cap is instead half the
//    r-weighted full-disk projection, exactly as on the weighted path it is
//    half the omega-weighted one.
// ---------------------------------------------------------------------------

// Builds the r-weighted full-disk rule ((alpha,beta)=(0,0), shifted Legendre)
// cache for fp_rhs_radial_full_disk_projection / fp_rhs_radial_uniform_inlet.
// NOT interchangeable with finite_peclet_rhs.h's fp_rhs_build_full_disk_rule_cache
// ((alpha,beta)=(1,0)) nor with its cap-rule cache.
template <typename Ttype>
FPCapRuleCache<Ttype> fp_rhs_radial_build_full_disk_rule_cache(const std::vector<SeriesTermData<Ttype>> &modes,
                                                               unsigned cap_quad_margin);

/** One interface cap's r-weighted quadrature kernel:
 *      L+_{nm}(a) = prefactor * sum_i kernel_i * G_nm(s_i),
 *  the L2_r counterpart of fp_rhs_build_cap_kernel. `empty` (a >= 1),
 *  `center` (n = 0, a = 0) and `zero` (n > 0 even, a = 0) are the degenerate
 *  cases; when any is set, `s`/`kernel` are unused. The degenerate branches are
 *  weight-independent (they are geometric statements about the cap), so they
 *  match the weighted builder's exactly. */
template <typename Ttype>
FPCapKernel<Ttype> fp_rhs_radial_build_cap_kernel(unsigned angular_index, const Ttype &absolute_interface,
                                                  const FPGaussJacobiRule<Ttype> &rule);

// Uniform inlet (Graetz): b^0_m = sqrt(2*pi) * <1, R_{0m}>_r, and b^n = 0 for n > 0.
//   modes                : the retained modes of the block (all with the same n)
//   angular_index        : n
//   cap_quad_margin      : see fp_rhs_required_order (finite_peclet_rhs.h);
//                          each mode's own quadrature order is derived from its
//                          own bare root, not shared uniformly across the block.
//   full_disk_rule_cache : optional, see fp_rhs_radial_build_full_disk_rule_cache.
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_uniform_inlet(unsigned angular_index,
                                               const std::vector<SeriesTermData<Ttype>> &modes,
                                               unsigned cap_quad_margin,
                                               const FPCapRuleCache<Ttype> *full_disk_rule_cache = nullptr);

// Stratified inlet: piecewise-constant layers separated by interfaces z_i.
//   cap_quad_margin  : see fp_rhs_required_order.
//   full_disk_column : when non-empty, used as the n=0 block's exact full-disk
//                       term (see fp_rhs_radial_full_disk_from_gram_column)
//                       instead of recomputing it by quadrature. Ignored for
//                       angular_index > 0. Pass an empty vector (the default)
//                       to always fall back to the independent quadrature.
//   cap_rule_cache   : optional; built by finite_peclet_rhs.h's
//                       fp_rhs_build_rule_cache -- the (0,1/2) cap rule is
//                       shared with the weighted path unchanged.
//   full_disk_rule_cache : optional, and specific to THIS module; only
//                       consulted for angular_index == 0 when full_disk_column
//                       is empty.
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_stratified_inlet(unsigned angular_index,
                                                  const std::vector<SeriesTermData<Ttype>> &modes,
                                                  const std::vector<Ttype> &interface_positions,
                                                  const std::vector<Ttype> &layer_values,
                                                  unsigned cap_quad_margin,
                                                  const std::vector<Ttype> &full_disk_column = std::vector<Ttype>(),
                                                  const FPCapRuleCache<Ttype> *cap_rule_cache = nullptr,
                                                  const FPCapRuleCache<Ttype> *full_disk_rule_cache = nullptr);

// sqrt(2*pi) * int_0^1 R_{0m}(r) r dr for an n = 0 block.
// See finite_peclet_gram_radial.h: when the block contains the exact constant
// mode (Neumann, n = 0, Lambda = 0), prefer
// fp_rhs_radial_full_disk_from_gram_column instead -- it is exact and reuses
// samples already computed by the Gram assembly, at zero extra Kummer
// evaluations.
//   cap_quad_margin: see fp_rhs_required_order.
//   rule_cache: optional, see fp_rhs_radial_build_full_disk_rule_cache.
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_full_disk_projection(const std::vector<SeriesTermData<Ttype>> &modes,
                                                      unsigned cap_quad_margin,
                                                      const FPCapRuleCache<Ttype> *rule_cache = nullptr);

// Exact alternative: R_{0,0} == 1, so <1, R_{0m}>_r == U^0_{m,0}.
// `gram_zero_column[i]` must be U^0_{i, zero_index} from the SAME assembly that
// produced the block's Gram factor (FPGramRadialFactor::constant_mode_gram_column).
// Returns sqrt(2*pi) * gram_zero_column.
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_full_disk_from_gram_column(const std::vector<Ttype> &gram_zero_column);

#endif // FINITE_PECLET_RHS_RADIAL_H
