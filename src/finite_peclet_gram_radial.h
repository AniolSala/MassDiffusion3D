#ifndef FINITE_PECLET_GRAM_RADIAL_H
#define FINITE_PECLET_GRAM_RADIAL_H

// ---------------------------------------------------------------------------
// finite_peclet_gram_radial.{h,cpp} are the L2_r counterpart of
// finite_peclet_gram_gauss_jacobi.{h,cpp}: they assemble
//
//     U^n_{mk} = int_0^1 R_{nm} R_{nk} r dr          (weight r dr)
//
// instead of that module's
//
//     W^n_{mk} = int_0^1 omega R_{nm} R_{nk} r dr,   omega = 1 - r^2 .
//
// With s = r^2 and R_{nm}(r) = r^n G_{nm}(s) (fp_radial_factor IS G_{nm}), the
// two reduce to
//
//     W^n_{mk} = 1/2 int_0^1 (1-s) s^n G_m G_k ds        (alpha, beta) = (1, n)
//     U^n_{mk} = 1/2 int_0^1       s^n G_m G_k ds        (alpha, beta) = (0, n)
//
// so the ENTIRE difference between the two backends is the Jacobi parameter
// alpha of the shifted-Jacobi quadrature measure -- i.e. the three-term
// recurrence in rule(), and nothing else.  Everything downstream (column
// equilibration, pivoted QR, the under-resolution gate, the constant-mode
// column capture) is identical by construction.
//
// This module obeys the SAME inlet-blindness contract as
// finite_peclet_gram_gauss_jacobi.h: it depends on nothing but the boundary
// value problem's spectral data (the angular index n, the retained modes'
// root_fp/btilde_fp, and the node-rule options).  It must never include a
// boundary-condition-specific header, never reference profile geometry,
// stratum positions or values, jump positions, projected load vectors, or a
// disk-average baseline term, and never be specialized per solver subclass
// (stratified vs. Graetz).  Everything profile/boundary-specific lives in
// finite_peclet_rhs_radial.{h,cpp}.
//
// KNOWN LIMITATION (do not file as a bug): as kappa -> 0 the modified modes
// tend to the bare ones, which are orthogonal in L2_omega but NOT in L2_r.
// W^n therefore becomes diagonal in that limit while U^n tends to the
// non-diagonal unweighted overlap matrix (the pseudo-products matrix,
// pseudo_products.h).  Any test asserting a diagonal collapse of U^n is wrong;
// the correct kappa -> 0 cross-check is the pencil identity
// W^n + kappa (Lambda U^n + U^n Lambda) = diag, tested in
// tests/test_fp_radial_projection.cpp.
// ---------------------------------------------------------------------------

#include <cmath>
#include <stdexcept>
#include <vector>
#include "series_term_struct.h"
#include "diagnostics/gram_diagnostics.h"

template <typename Ttype> struct FPGramRadialOptions {
    // Node rule:  N_n = oversampling_factor·K_n + oversampling_margin,
    // then clamped from below by minimum_factor·K_n -- see the header
    // comment on fp_gram_radial_node_count (the only place this product is
    // actually computed) for the measured justification.
    unsigned oversampling_factor = 2u;    // VALIDATED DEFAULT -- do not lower
    unsigned oversampling_margin = 20u;   // VALIDATED DEFAULT -- do not lower
    bool     enable_order_check  = false;

    // Hard safety floor applied after the affine rule. N < K makes B^T B exactly
    // singular (rank(B^T B) <= N); N only slightly above K destroys the small
    // singular values that govern the solve. Set to 0 to disable the floor
    // ONLY in a deliberate study; production must leave it at 2.
    unsigned minimum_factor = 2u;         // floor: N >= minimum_factor * K

    // When true (the default), the node rule adds its b_max-driven term (see
    // fp_gram_radial_node_count). Set false ONLY to prescribe an exact node
    // count -- a dense reference, or a deliberately starved rule in a test.
    bool scale_aware_nodes = true;

    // When true, fp_gram_factor_radial keeps the raw sampled radial factors,
    // and its quadrature rule, in the returned factor. Costs ~K*N stored
    // scalars per block; leave false when they are not needed.
    //
    // Consumed by finite_peclet_rhs_representer_radial.h, which contracts them
    // against the L2_r representer of each cap functional at ZERO additional
    // confluent-hypergeometric evaluations -- exactly as the weighted backend's
    // samples serve finite_peclet_rhs_representer.h.
    bool retain_samples = false;
};

// Number of Gauss-Jacobi (alpha=0, beta=n) nodes used to sample an angular block
// of K modes.
//
//     N = max(factor*K + margin, minimum_factor*K)
//
// MEASURED JUSTIFICATION for THIS backend. The rule was re-measured rather
// than inherited from fp_gram_node_count: the L2_r integrand differs from the
// L2_omega one by a polynomial factor, so nothing about the old measurement
// transfers automatically.
//
// Protocol: real blocks from the bare-root catalogue via fp_solve_ladder at
// kappa = 1e-2, the real stratified load vector (zi = {-0.4, 0.82},
// ui = {1, 0.3, 0}), criterion = worst RELATIVE COEFFICIENT ERROR against the
// same block solved at a dense N = 600, measured against that block's own
// coefficient scale, threshold 1e-8.
//
// The criterion matters. Do NOT use the solve residual
// ||A c - b|| / max(1, max|b|) that tests/test_fp_gram_node_rule.cpp uses: its
// max(1, .) normalisation is VACUOUS for high-n blocks, whose load vectors run
// to ~1e-28, so it reports round-off-level "success" for blocks whose
// coefficients are wrong by a factor of hundreds. An earlier version of this
// table was measured that way and was wrong.
//
//     n   K    radial   weighted   2K+20   2K
//     84  8      32        32        36     16   <- 2K insufficient
//     83  9      35        31        38     18   <- 2K insufficient
//     90  6      28        28        32     12   <- 2K insufficient
//     99  8      34        34        36     16   <- 2K insufficient
//     60  10     34        34        40     20   <- 2K insufficient
//     60  20     54        54        60     40   <- 2K insufficient
//     20  60    112       112       140    120
//     0   60    104       104       140    120
//     99  60    130       130       140    120   <- 2K insufficient
//
// Two conclusions. First, the two backends need essentially IDENTICAL node
// counts, so the weighted backend's validated (2, 20) defaults carry over
// unchanged -- the tightest margins in the grid are 38 vs 35 and 140 vs 130.
//
// Second, and more important: the requirement does NOT scale with K. At small
// K it is governed by n and the modes' own oscillation, so the MULTIPLICATIVE
// part of the rule collapses there and the ADDITIVE oversampling_margin is the
// only thing carrying the block -- at n = 84, K = 8 the need is 32 nodes while
// 2K is 16. oversampling_margin is therefore not decoration to be tuned down
// for speed: lowering it to 1 (as one benchmark config did) leaves high-n,
// small-K blocks under-resolved, which is silent because the R-diagonal gate
// below does not fire. That configuration was measured producing coefficients
// wrong by a relative factor of 435 (radial) and 286 (weighted) at n ~ 84,
// K ~ 8, and a visibly wrong assembled field. Both backends are affected;
// radial somewhat more.
//
// The grid above deliberately includes SMALL K at HIGH n. An earlier version of
// this study swept only (0,60), (20,60), (60,20), (60,40), (60,60) and so
// missed the only corner where the default rule is genuinely load-bearing.
//
// WARNING: entry-wise agreement of U is NOT a valid acceptance criterion
// either. As for the weighted backend, a badly under-sampled rule can reproduce
// every entry of U to ~1e-12 while the solve is wrong by many orders of
// magnitude, because rank(B^T B) <= N. Any future retuning of these parameters
// MUST be validated on the coefficients, and must keep the run-time gate.
//
// THIRD TERM (b_max), added with the weighted backend's: the measurement above
// was taken at kappa = 1e-2 (Pe = 10) only, where every rate -- and with it
// b_max = max sqrt(Lam) -- stays small. The mode carries s^n e^{-b s/2} on
// BOTH projections, so the mass sits at s* = n/b on both, and a b_max-blind
// node count starves the same high-n blocks here as there. Measured on the
// weighted backend (see fp_gram_node_count); the radial requirement tracks it,
// as the table above already shows for the K-driven terms.
//
//     N = max(factor*K + margin, minimum_factor*K, ceil(sqrt(K*b_max)) + margin)
//
// radial_scale = 0 disables the third term, reproducing the historical rule.
template <typename Ttype>
unsigned fp_gram_radial_node_count(unsigned mode_count, const FPGramRadialOptions<Ttype> &options,
                                   Ttype radial_scale = static_cast<Ttype>(0))
{
    if (!mode_count)
        throw std::invalid_argument("fp_gram_radial_node_count: empty block");
    if (!options.oversampling_factor)
        throw std::invalid_argument("fp_gram_radial_node_count: oversampling factor must be positive");
    const unsigned affine = options.oversampling_factor * mode_count + options.oversampling_margin;
    const unsigned floor_ = options.minimum_factor * mode_count;
    unsigned N = affine > floor_ ? affine : floor_;
    if (radial_scale > static_cast<Ttype>(0) && std::isfinite(static_cast<long double>(radial_scale)))
    {
        const long double scaled =
            std::ceil(std::sqrt(static_cast<long double>(mode_count) * static_cast<long double>(radial_scale)))
            + static_cast<long double>(options.oversampling_margin);
        if (scaled > static_cast<long double>(N))
            N = static_cast<unsigned>(scaled);
    }
    if (N < mode_count)
        throw std::runtime_error("fp_gram_radial_node_count: node count below mode count");
    return N;
}

template <typename Ttype> struct FPGramRadialFactor {
    unsigned angular_index = 0, mode_count = 0, node_count = 0;
    std::vector<Ttype> upper_triangular_R, column_scaling_D;
    std::vector<unsigned> pivot_permutation;
    GramBlockDiagnostics<Ttype> diagnostics;

    // Column U^0_{:,j0} of the assembled (unfactored) Gram block, where j0 is the
    // local index of the exact constant mode (Neumann, n = 0, rate_fp == 0).
    // Empty when the block contains no such mode. Filled during assembly at zero
    // extra radial evaluations: the sampled values are already in memory. The
    // fp_rhs_radial_ module turns this column into the exact n=0 disk-average
    // load-vector term (its name deliberately not spelled out here -- see
    // finite_peclet_rhs_radial.h's own design-contract comment for why).
    std::vector<Ttype> constant_mode_gram_column;
    bool     has_constant_mode = false;
    unsigned constant_mode_local_index = 0;

    // Present only when options.retain_samples was set.
    // radial_samples[q*mode_count + j] = G_{n,j}(quadrature_nodes[q])
    // UNSCALED: neither the sqrt(w_q/2) sampling factor nor the column scaling
    // D is applied. Storing a scaled copy is a silent correctness bug.
    std::vector<Ttype> radial_samples;
    std::vector<Ttype> quadrature_nodes;    // length node_count
    std::vector<Ttype> quadrature_weights;  // length node_count
    bool samples_retained = false;
};
template <typename Ttype> FPGramRadialFactor<Ttype> fp_gram_factor_radial(unsigned, const std::vector<SeriesTermData<Ttype>> &, const FPGramRadialOptions<Ttype> &);
template <typename Ttype> void fp_gram_solve_radial(const FPGramRadialFactor<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &);
template <typename Ttype> void fp_gram_reconstruct_radial(const FPGramRadialFactor<Ttype> &, std::vector<std::vector<Ttype>> &);

#endif
