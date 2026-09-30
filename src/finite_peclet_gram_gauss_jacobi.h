#ifndef FINITE_PECLET_GRAM_GAUSS_JACOBI_H
#define FINITE_PECLET_GRAM_GAUSS_JACOBI_H

// ---------------------------------------------------------------------------
// finite_peclet_gram_gauss_jacobi.{h,cpp} depend on nothing but the boundary
// value problem's spectral data: the angular index n, the retained modes'
// root_fp/btilde_fp, and the node-rule options. They must never include a
// boundary-condition-specific header, never reference profile geometry,
// stratum positions or values, jump positions, projected load vectors, or a
// disk-average baseline term, and never be specialized per solver subclass
// (stratified vs. Graetz). Everything profile/boundary-specific instead lives
// in finite_peclet_rhs.{h,cpp}.
// ---------------------------------------------------------------------------

#include <cmath>
#include <stdexcept>
#include <vector>
#include "series_term_struct.h"
#include "diagnostics/gram_diagnostics.h"

template <typename Ttype> struct FPGramGaussJacobiOptions {
    // Node rule:  N_n = oversampling_factor·K_n + oversampling_margin,
    // then clamped from below by minimum_factor·K_n -- see the header
    // comment on fp_gram_node_count (the only place this product is actually
    // computed) for the measured justification.
    unsigned oversampling_factor = 2u;    // VALIDATED DEFAULT -- do not lower
    unsigned oversampling_margin = 20u;   // VALIDATED DEFAULT -- do not lower
    bool     enable_order_check  = false;

    // Hard safety floor applied after the affine rule. N < K makes B^T B exactly
    // singular (rank(B^T B) <= N); N only slightly above K destroys the small
    // singular values that govern the solve. Set to 0 to disable the floor
    // ONLY in a deliberate study; production must leave it at 2.
    unsigned minimum_factor = 2u;         // floor: N >= minimum_factor * K

    // When true (the default), the node rule adds its b_max-driven term (see
    // fp_gram_node_count). Set false ONLY to prescribe an exact node count --
    // a dense reference, or a deliberately starved rule in a test. Production
    // code must leave it true: without it, high-n blocks at high Peclet are
    // sampled nowhere near where their modes live.
    bool scale_aware_nodes = true;

    // When true, fp_gram_factor_gauss_jacobi keeps the raw sampled radial
    // factors, and its quadrature rule, in the returned factor, so that the
    // representer RHS backend can contract against them at ZERO additional
    // confluent-hypergeometric evaluations. Costs ~K*N stored scalars per
    // block; leave false when they are not needed.
    //
    // This module's design contract (see the header comment above) is
    // unweakened by retaining them: they are spectral samples of the block's
    // own modes at the block's own quadrature nodes, nothing more --
    // boundary-condition-specific code still belongs only in finite_peclet_rhs.h.
    bool retain_samples = false;
};

// Number of Gauss-Jacobi (alpha=1, beta=n) nodes used to sample an angular block
// of K modes.
//
//     N = max(factor*K + margin, minimum_factor*K, ceil(sqrt(K*b_max)) + margin)
//
// where b_max = max_j sqrt(Lam_j) over the block's retained modes (root_fp).
//
// MEASURED JUSTIFICATION of the first two terms (kappa = 1e-2; criterion =
// relative error of the SOLVE W c = b, not entry-wise agreement of W):
//
//     n   K    minimum N for solve err < 1e-8     factor*K + margin (2, 20)
//     0   60           108                                140
//     20  60           120                                140
//     60  20/40/60     60 / 100 / 140                     60 / 100 / 140
//
// At high n the default is EXACTLY the minimum: there is no headroom.
//
// WHY THE THIRD TERM (b_max) EXISTS. The measurement above was taken at
// kappa = 1e-2 (Pe = 10) ONLY, where the axial-diffusion deformation shrinks
// every rate so much that b_max stays small. It does not survive to high Pe.
// The mode carries s^n e^{-b s/2}, so |R_nm|^2 against the rule's own measure
// peaks at s* = n/b: for b <~ n the mass sits at the wall, exactly where the
// Gauss-Jacobi (1,n) nodes cluster, and a K-driven node count is fine. For
// b >> n the mass moves INWARD to s* = n/b while the nodes stay at the wall,
// and the rule stops sampling the region where the modes differ. Since
// b_max ~ max_root for EVERY angular block at high Pe while K collapses like
// (max_root - 2n)/4, the high-n blocks are the ones starved of nodes.
//
// Measured (stratified, zi {-0.55, 0.45}, criterion = R-diagonal ratio < 2,
// which is ~1 for a healthy block; minimum N found by scan):
//
//   Pe    max_root  n    K    b_max   minimum N   factor*K+margin (2,20)
//   1000  400       0    100  358     184         220
//   1000  400       100  50   354     126         120   <- rule insufficient
//   1000  400       140  30   354     96          80    <- rule insufficient
//   1000  400       163  19   355     79          58    <- rule insufficient
//   1000  400       190  5    354     47          30    <- rule insufficient
//   1000  300       120  15   277     63          50    <- rule insufficient
//   100   400       163  19   155     49          58
//
// At Pe = 1000, max_root = 400 the (2, 20) rule leaves every block above
// n ~ 100 under-sampled, and blocks n ~ 150-175 reach R-diagonal ratios of
// 1e6-2.6e6, i.e. past the gate below. The same blocks at Pe = 100 are healthy
// untouched, because b_max there is 155 rather than 355.
//
// The sqrt(K*b_max) form fits that table within ~10%: with K ~ (b-2n)/4 it is
// 0.5*sqrt(b(b-2n)), which reproduces the measured minimum-N/b_max ratios
// (0.51 at n=0, 0.35 at n=100, 0.22 at n=163) essentially exactly. The
// additive margin then supplies the small-K headroom that the multiplicative
// part loses, exactly as it does for the affine term.
//
// WARNING: entry-wise agreement of W is NOT a valid acceptance criterion. At
// n = 20, K = 60, N = 15 the entries agree to 7e-12 while the solve is wrong by
// 6e+15, because rank(B^T B) <= N. Any future retuning of these parameters MUST
// be validated on the solve, and must keep the run-time gate in
// fp_gram_factor_gauss_jacobi (finite_peclet_gram_gauss_jacobi.cpp).
//
// radial_scale = 0 disables the third term, reproducing the historical rule.
template <typename Ttype>
unsigned fp_gram_node_count(unsigned mode_count, const FPGramGaussJacobiOptions<Ttype> &options,
                            Ttype radial_scale = static_cast<Ttype>(0))
{
    if (!mode_count)
        throw std::invalid_argument("fp_gram_node_count: empty block");
    if (!options.oversampling_factor)
        throw std::invalid_argument("fp_gram_node_count: oversampling factor must be positive");
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
        throw std::runtime_error("fp_gram_node_count: node count below mode count");
    return N;
}

template <typename Ttype> struct FPGramGaussJacobiFactor {
    unsigned angular_index = 0, mode_count = 0, node_count = 0;
    std::vector<Ttype> upper_triangular_R, column_scaling_D;
    std::vector<unsigned> pivot_permutation;
    GramBlockDiagnostics<Ttype> diagnostics;

    // Column W^0_{:,j0} of the assembled (unfactored) Gram block, where j0 is the
    // local index of the exact constant mode (Neumann, n = 0, rate_fp == 0).
    // Empty when the block contains no such mode. Filled during assembly at zero
    // extra radial evaluations: the sampled values are already in memory. The
    // fp_rhs_ module turns this column into the exact n=0 disk-average load-
    // vector term (its name deliberately not spelled out here -- see
    // finite_peclet_rhs.h's own design-contract comment for why).
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
template <typename Ttype> FPGramGaussJacobiFactor<Ttype> fp_gram_factor_gauss_jacobi(unsigned, const std::vector<SeriesTermData<Ttype>> &, const FPGramGaussJacobiOptions<Ttype> &);
template <typename Ttype> void fp_gram_solve_gauss_jacobi(const FPGramGaussJacobiFactor<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &);
template <typename Ttype> void fp_gram_reconstruct_gauss_jacobi(const FPGramGaussJacobiFactor<Ttype> &, std::vector<std::vector<Ttype>> &);

#endif
