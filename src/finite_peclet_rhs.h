#ifndef FINITE_PECLET_RHS_H
#define FINITE_PECLET_RHS_H

#include <vector>
#include "series_term_struct.h"
#include "finite_peclet_roots.h"   // WallCondition

// ---------------------------------------------------------------------------
// Inlet/boundary-condition-specific load vectors ("RHS") for the finite-Peclet
// Galerkin system  W^n c^n = b^n  of eq:coefficients_fp.
//
// This module owns everything that depends on the inlet profile. The Gram
// modules (finite_peclet_gram_*.h) are inlet-blind and must never be edited to
// accommodate a new inlet: add a new builder here instead.
//
// The cap quadrature used here is the Jacobi rule with (alpha, beta) = (0, 1/2)
// for the weight eta^{-1/2}-regularised circular-cap integral of Appendix B.
// It is NOT the Gram rule (alpha, beta) = (1, n). Do not merge the two.
// ---------------------------------------------------------------------------

/** Fixed Gauss-Jacobi rule on [0,1] for eta^(1/2) d eta -- the interface-cap
 *  rule of Appendix B. Distinct from the Gram backend's own (alpha=1, n) rule. */
template <typename Ttype>
struct FPGaussJacobiRule
{
    std::vector<Ttype> nodes;
    std::vector<Ttype> weights;
};

/** Construct a positive, fixed-order Gauss-Jacobi cap rule on [0,1]. */
template <typename Ttype>
FPGaussJacobiRule<Ttype> make_fp_gauss_jacobi_rule(unsigned number_of_points);

// Required Gauss-Jacobi quadrature order for ONE mode's own cap/full-disk
// integral, sized off that mode's own BARE (kappa-independent) root -- the
// parameter that actually governs how oscillatory its radial factor is over
// [0,1]. This replaced an earlier scheme that sized the whole angular block
// by its mode COUNT (K_n): since the cap/full-disk integrals are independent
// per-mode 1-D integrals (unlike the Gram matrix, which couples all K modes
// of a block into one K-by-K system), sizing by block size over-resolved
// every low-root mode sharing a block with a more oscillatory one.
//
// Result = ceil(kCapOrderRootCoefficient * bare_root) + cap_quad_margin,
// rounded UP to the nearest power of two times a fixed base unit (20; see
// finite_peclet_rhs.cpp) -- GEOMETRIC, not linear.
//
// The coefficient (0.6) is empirical, not 1: a direct convergence study
// (compute_layer_nm_quadrature against a genuine interface, threshold 1e-10)
// across a grid of (n, m) pairs found the SMALLEST converging order stays a
// roughly CONSTANT, bounded fraction of the bare root (worst measured ratio
// ~0.48) -- using the raw root itself (coefficient=1) over-resolves by
// roughly 2x everywhere. See finite_peclet_rhs.cpp's kCapOrderRootCoefficient
// comment for the measured (n, m, root, order_needed) table. This does NOT
// mean the radial index m alone predicts the order (an earlier, wrong
// hypothesis): the bare root ~ 4m + 2n + 2 already mixes both indices, and
// remains the right single predictor once correctly scaled.
//
// Bucketing at all is deliberate: building a Gauss-Jacobi rule costs
// O(order^2) (Golub-Welsch), while evaluating it costs only O(order) per
// mode, so building a fresh rule for every individual mode would trade a
// cheap per-mode evaluation for an expensive per-mode construction -- a net
// regression. The bucket spacing must itself be geometric, not linear: a
// single angular block's modes can span a WIDE range of bare roots (root
// ~2 to ~400 within one n=0 block at max_root=600), and a linear step would
// then produce O(range/step) distinct buckets -- and rule constructions --
// in that one block alone, which measured SLOWER than the single shared rule
// this scheme replaced. Geometric spacing bounds the number of distinct
// buckets anywhere in the whole solve to O(log2(max_root)), independent of
// how many modes populate any one block.
template <typename Ttype>
unsigned fp_rhs_required_order(const Ttype &bare_root, unsigned cap_quad_margin);

// ---------------------------------------------------------------------------
// Rule caches. A quadrature rule of a given order costs O(order^2) to BUILD
// (Golub-Welsch) but only O(order) to EVALUATE per mode. fp_rhs_required_order
// bounds the number of DISTINCT orders needed by any one block to O(log2),
// but the stratified driver calls fp_rhs_stratified_inlet once PER ANGULAR
// BLOCK (up to ~100 blocks for a max_root=600 solve) -- and different blocks
// routinely need the SAME order (e.g. every block's lowest-root mode buckets
// to 20). Without sharing, every block independently rebuilds every order it
// needs, multiplying the O(log2) bound by the block count -- measured to cost
// MORE than the single-shared-rule scheme this module originally replaced.
//
// Build a cache ONCE from the WHOLE retained mode set (every block, not just
// one) before looping over blocks, and pass it into every per-block call
// below via the trailing pointer parameter. The cache is read-only after
// construction, so sharing it (by const pointer) across an OpenMP parallel-
// for over blocks is safe. Every function below also has a cache-less form
// (pointer defaults to nullptr) that builds its own private, call-scoped
// cache -- self-contained and correct, just without the cross-block sharing,
// for tests and other simple call sites.
//
// Two SEPARATE cache instances are required, never interchangeable: the cap
// rule ((alpha, beta) = (0, 1/2), Appendix B, used by fp_rhs_stratified_inlet's
// interface caps) and the full-disk rule ((alpha, beta) = (1, n=0), used by
// fp_rhs_full_disk_projection) are different quadrature families that happen
// to share the same (nodes, weights) storage shape -- do not pass a cache
// built by one of fp_rhs_build_rule_cache / fp_rhs_build_full_disk_rule_cache
// to a function expecting the other.
template <typename Ttype>
using FPCapRuleCache = std::vector<std::pair<unsigned, FPGaussJacobiRule<Ttype>>>;

// Builds the cap rule ((alpha,beta)=(0,1/2)) cache for fp_rhs_stratified_inlet.
template <typename Ttype>
FPCapRuleCache<Ttype> fp_rhs_build_rule_cache(const std::vector<SeriesTermData<Ttype>> &modes, unsigned cap_quad_margin);

// Builds the full-disk rule ((alpha,beta)=(1,0)) cache for
// fp_rhs_full_disk_projection / fp_rhs_uniform_inlet.
template <typename Ttype>
FPCapRuleCache<Ttype> fp_rhs_build_full_disk_rule_cache(const std::vector<SeriesTermData<Ttype>> &modes, unsigned cap_quad_margin);

// Uniform inlet (Graetz): b^0_m = sqrt(2*pi) * <1, R_{0m}>_omega, and b^n = 0 for n > 0.
//   modes            : the retained modes of the block (all with the same n)
//   angular_index    : n
//   cap_quad_margin  : see fp_rhs_required_order; each mode's own quadrature
//                      order is derived from its own bare root, not shared
//                      uniformly across the block.
//   full_disk_rule_cache : optional, see fp_rhs_build_full_disk_rule_cache.
template <typename Ttype>
std::vector<Ttype> fp_rhs_uniform_inlet(unsigned angular_index,
                                        const std::vector<SeriesTermData<Ttype>> &modes,
                                        unsigned cap_quad_margin,
                                        const FPCapRuleCache<Ttype> *full_disk_rule_cache = nullptr);

// Stratified inlet: piecewise-constant layers separated by interfaces z_i.
//   cap_quad_margin  : see fp_rhs_required_order.
//   full_disk_column : when non-empty, used as the n=0 block's exact full-disk
//                       term (see fp_rhs_full_disk_from_gram_column) instead of
//                       recomputing it via fp_rhs_full_disk_projection. Ignored
//                       for angular_index > 0. Pass an empty vector (the
//                       default) to always fall back to the independent
//                       quadrature -- e.g. no Gram factor available yet, or the
//                       block has no exact constant mode (Dirichlet wall).
//   cap_rule_cache, full_disk_rule_cache : optional, see fp_rhs_build_rule_cache
//                       / fp_rhs_build_full_disk_rule_cache. full_disk_rule_cache
//                       is only consulted for angular_index == 0 when
//                       full_disk_column is empty.
template <typename Ttype>
std::vector<Ttype> fp_rhs_stratified_inlet(unsigned angular_index,
                                           const std::vector<SeriesTermData<Ttype>> &modes,
                                           const std::vector<Ttype> &interface_positions,
                                           const std::vector<Ttype> &layer_values,
                                           unsigned cap_quad_margin,
                                           const std::vector<Ttype> &full_disk_column = std::vector<Ttype>(),
                                           const FPCapRuleCache<Ttype> *cap_rule_cache = nullptr,
                                           const FPCapRuleCache<Ttype> *full_disk_rule_cache = nullptr);

// sqrt(2*pi) * int_0^1 (1 - r^2) R_{0m}(r) r dr for an n = 0 block.
// See finite_peclet_gram_gauss_jacobi.h: when the block contains the exact
// constant mode (Neumann, n = 0, Lambda = 0), prefer
// fp_rhs_full_disk_from_gram_column instead -- it is exact and reuses samples
// already computed by the Gram assembly, at zero extra Kummer evaluations.
//   cap_quad_margin: see fp_rhs_required_order.
//   rule_cache: optional, see fp_rhs_build_full_disk_rule_cache.
template <typename Ttype>
std::vector<Ttype> fp_rhs_full_disk_projection(const std::vector<SeriesTermData<Ttype>> &modes,
                                               unsigned cap_quad_margin,
                                               const FPCapRuleCache<Ttype> *rule_cache = nullptr);

// Exact alternative: R_{0,0} == 1, so <1, R_{0m}>_omega == W^0_{m,0}.
// `gram_zero_column[i]` must be W^0_{i, zero_index} from the SAME assembly that
// produced the block's Gram factor (FPGramGaussJacobiFactor::constant_mode_gram_column).
// Returns sqrt(2*pi) * gram_zero_column.
template <typename Ttype>
std::vector<Ttype> fp_rhs_full_disk_from_gram_column(const std::vector<Ttype> &gram_zero_column);

/** One interface cap's quadrature kernel:
 *      L+_{nm}(a) = prefactor * sum_i kernel_i * G_nm(s_i).
 *  `empty` (a >= 1), `center` (n = 0, a = 0) and `zero` (n > 0 even, a = 0) are
 *  the degenerate cases; when any is set, `s`/`kernel` are unused. This is the
 *  same object make_cap has always built, promoted to the public API so that
 *  finite_peclet_rhs_representer.h can integrate the SAME kernel against
 *  shifted Jacobi polynomials instead of radial factors. */
template <typename Ttype>
struct FPCapKernel
{
    unsigned angular_index = 0;
    std::vector<Ttype> s, kernel;
    Ttype prefactor = 0;
    bool empty = false, center = false, zero = false;
};

template <typename Ttype>
FPCapKernel<Ttype> fp_rhs_build_cap_kernel(unsigned angular_index, const Ttype &absolute_interface,
                                           const FPGaussJacobiRule<Ttype> &rule);

// Retained for the existing tests; unchanged semantics. Evaluates the positive
// circular-cap projection from the Appendix B reduction for one mode at one
// absolute interface `a`; `radial_gauss_weights`/`radial_gauss_nodes` are the
// bare-mode radial Gaussian rule, consumed only for the n=0,a=0 "center" case.
template <typename Ttype>
Ttype compute_layer_nm_quadrature(const SeriesTermData<Ttype> &mode, const Ttype &absolute_interface,
                                  const FPGaussJacobiRule<Ttype> &rule,
                                  const std::vector<Ttype> &radial_gauss_weights,
                                  const std::vector<Ttype> &radial_gauss_nodes);

#endif
