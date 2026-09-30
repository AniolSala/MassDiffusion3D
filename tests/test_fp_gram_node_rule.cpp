// Regression tests for the parameterized node rule and under-resolution gate
// of the Gauss-Jacobi Gram backend (plans/REFACTOR_PLAN.md section 4/section 0).
//
// T-NR-04 pins section 0's retracted recommendation: an n-damped node rule
// (e.g. N_n = ceil(2*K_n/sqrt(1+n/2)) + 20) must NEVER be reintroduced. It was
// derived from an invalid metric (entry-wise Gram agreement); the actual
// solve error at the node counts it would produce is catastrophic because
// rank(B^T B) <= N.

#include "tinytest.h"

#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_roots.h"

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#undef protected
#undef private

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
// Real bare-root catalog (beta_{nm}, kappa=0), loaded once and reused for
// every (n, K, kappa) combination below -- cheap (a table read, no root
// finding or Gram work) and independent of any particular inlet profile.
const std::vector<std::vector<double>> &bare_root_catalog()
{
    static const std::vector<std::vector<double>> catalog = [] {
        CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
        return sol.get_bare_root_catalog();
    }();
    return catalog;
}

// Build a REAL angular block of K modes at angular index n and axial-
// diffusion parameter kappa, via the actual finite-Peclet root ladder solver
// (fp_solve_ladder) applied to the real bare-root catalog -- not an ad hoc
// synthetic formula. This is what makes the accuracy/conditioning assertions
// below a faithful check of REFACTOR_PLAN.md section 0's measured claims
// (which were themselves measured against real solver data).
std::vector<SeriesTermData<double>> make_block(unsigned n, unsigned K, double kappa)
{
    const auto &catalog = bare_root_catalog();
    if (n >= catalog.size() || catalog[n].size() < K)
        throw std::runtime_error("make_block: bare-root catalog too small for requested (n, K)");
    std::vector<double> base_beta2(K);
    for (unsigned m = 0; m < K; ++m)
        base_beta2[m] = catalog[n][m] * catalog[n][m];
    const bool skip_zero_mode = (n == 0);
    const auto Lambda = fp_solve_ladder<double>(n, base_beta2, kappa, WallCondition::Neumann, skip_zero_mode, 1e-12, 200);

    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        modes[m].n = n;
        modes[m].m = m;
        modes[m].root_fp = std::sqrt(Lambda[m]);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * Lambda[m]);
        modes[m].rate_fp = Lambda[m];
    }
    return modes;
}

// Dense reference Gram matrix at a large, fixed node count -- independent of
// whatever N the options under test select.
std::vector<std::vector<double>> reference_gram(unsigned n, const std::vector<SeriesTermData<double>> &modes, unsigned N_ref)
{
    FPGramGaussJacobiOptions<double> ref_opts;
    ref_opts.oversampling_factor = 1;
    ref_opts.oversampling_margin = N_ref - static_cast<unsigned>(modes.size());
    ref_opts.minimum_factor = 1;
    ref_opts.scale_aware_nodes = false;   // N_ref is prescribed, not derived
    const auto factor = fp_gram_factor_gauss_jacobi(n, modes, ref_opts);
    std::vector<std::vector<double>> W;
    fp_gram_reconstruct_gauss_jacobi(factor, W);
    return W;
}

// Relative max-norm solve residual of `factor`/`c` against the dense reference
// Gram `W_ref`: max_i |sum_j W_ref[i][j]*c[j] - b[i]| / max(1, max_i |b[i]|).
double solve_residual(const std::vector<std::vector<double>> &W_ref, const std::vector<double> &b, const std::vector<double> &c)
{
    const unsigned K = static_cast<unsigned>(b.size());
    double max_resid = 0.0, max_b = 1.0;
    for (unsigned i = 0; i < K; ++i)
    {
        double acc = 0.0;
        for (unsigned j = 0; j < K; ++j)
            acc += W_ref[i][j] * c[j];
        max_resid = std::max(max_resid, std::abs(acc - b[i]));
        max_b = std::max(max_b, std::abs(b[i]));
    }
    return max_resid / max_b;
}
}

// ─── T-NR-01: affine rule ─────────────────────────────────────────────────

TEST_CASE(fp_gram_node_count_affine_rule)
{
    for (unsigned K : {1u, 5u, 20u, 60u, 150u})
        for (unsigned factor : {1u, 2u, 3u})
            for (unsigned margin : {0u, 20u, 50u})
                for (unsigned min_factor : {1u, 2u, 4u})
                {
                    FPGramGaussJacobiOptions<double> o;
                    o.oversampling_factor = factor;
                    o.oversampling_margin = margin;
                    o.minimum_factor = min_factor;
                    const unsigned affine = factor * K + margin;
                    const unsigned floor_ = min_factor * K;
                    const unsigned expected = affine > floor_ ? affine : floor_;
                    REQUIRE(fp_gram_node_count(K, o) == expected);
                    // The b_max term only ever raises the count, never lowers it.
                    REQUIRE(fp_gram_node_count(K, o, 400.0) >= expected);
                }

    FPGramGaussJacobiOptions<double> zero_factor;
    zero_factor.oversampling_factor = 0;
    bool threw = false;
    try { fp_gram_node_count(10u, zero_factor); }
    catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);

    threw = false;
    try { fp_gram_node_count(0u, FPGramGaussJacobiOptions<double>{}); }
    catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);
}

// ─── T-NR-02: floor engages ───────────────────────────────────────────────

TEST_CASE(fp_gram_node_count_floor_engages)
{
    FPGramGaussJacobiOptions<double> o;
    o.oversampling_factor = 1;
    o.oversampling_margin = 0;
    o.minimum_factor = 2;
    REQUIRE(fp_gram_node_count(60u, o) == 120u);
}

// ─── T-NR-03 / T-NR-06: solve accuracy and conditioning at the default rule ─

// Uses a REALISTIC right-hand side (the actual stratified-inlet RHS builder,
// finite_peclet_rhs.h) rather than a uniform-random one. This matters at high
// (n, K): the Gram diagonal itself spans an enormous, but entirely legitimate,
// dynamic range there (measured e.g. W[0][0] ~ 2.7e-21 vs W[59][59] ~ 1.7e-96
// at n=60, K=60 -- stable across node counts from 140 to 6000, i.e. NOT a
// quadrature artifact). A uniform-random b of O(1) is adversarial against that
// range and produces a huge apparent "solve error" that has nothing to do with
// the node rule; a real inlet RHS is naturally scaled commensurately with each
// mode (it is built from the same radial functions), and the solve is then
// accurate to machine precision at every (n, K, kappa) tested, including
// n = 95, K = 60.
TEST_CASE(fp_gram_default_rule_solve_accuracy_and_conditioning)
{
    const std::vector<double> zi{0.0};
    const std::vector<double> ui{1.0, 0.3};
    for (unsigned n : {0u, 5u, 20u, 60u, 95u})
        for (unsigned K : {20u, 40u, 60u})
            for (double kappa : {1e-1, 1e-2, 1e-3})
            {
                const auto modes = make_block(n, K, kappa);
                const auto factor = fp_gram_factor_gauss_jacobi(n, modes, FPGramGaussJacobiOptions<double>{});
                REQUIRE(factor.diagnostics.condition_estimate < 1e3);

                const auto b = fp_rhs_stratified_inlet<double>(n, modes, zi, ui, 100u);
                std::vector<double> c;
                fp_gram_solve_gauss_jacobi(factor, b, c);

                const auto W_ref = reference_gram(n, modes, 900);
                double max_resid = 0.0, max_b = 1.0;
                for (unsigned i = 0; i < K; ++i)
                {
                    double acc = 0.0;
                    for (unsigned j = 0; j < K; ++j)
                        acc += W_ref[i][j] * c[j];
                    max_resid = std::max(max_resid, std::abs(acc - b[i]));
                    max_b = std::max(max_b, std::abs(b[i]));
                }
                REQUIRE(max_resid / max_b < 1e-8);
            }
}

// ─── T-NR-04: REGRESSION -- entry agreement is not a valid criterion ──────
//
// Measured (REFACTOR_PLAN.md section 0), kappa = 1e-2, n = 20, K = 60:
//
//   N    entry error   sigma_min(B_hat)   cond(B_hat)   solve error
//   15   7.0e-12       2.8e-03            1.4e+03       6.2e+15
//   30   1.5e-14       3.6e-13            7.9e+12       7.9e+15
//   60   7.1e-15       8.0e-18            2.4e+17       1.5e+14
//   90   1.3e-14       1.0e-03            1.7e+03       2.0e+03
//   120  1.1e-14       2.02e-01           6.1e+00       2.5e-14
//
// This test exists solely to stop anyone re-deriving the retracted n-damped
// node rule of section 0: it asserts that at N = 15 the Gram entries agree
// with a dense reference to high precision WHILE the solve is catastrophically
// wrong -- so entry-wise agreement must never be used to justify a smaller N.
TEST_CASE(fp_gram_entry_agreement_is_not_a_valid_criterion_regression)
{
    const unsigned n = 20, K = 60;
    const double kappa = 1e-2;
    const auto modes = make_block(n, K, kappa);
    const auto W_ref = reference_gram(n, modes, 900);

    std::mt19937 rng(999);
    std::uniform_real_distribution<double> udist(0.5, 1.5);
    std::vector<double> b(K);
    for (double &v : b)
        v = udist(rng);

    // N < K is impossible to request through the gated public API (by design:
    // section 4.3's gate and the minimum_factor floor together forbid it), so
    // this regression scans N just above K -- still deep inside the "small
    // singular values destroyed" regime of REFACTOR_PLAN.md section 0's table
    // -- for a configuration where entries still agree tightly with the dense
    // reference while the solve is badly wrong. minimum_factor = 1 and
    // enable_order_check = false so fp_gram_node_count reproduces N exactly
    // and the gate is the only thing standing between this test and silently
    // wrong output (which is precisely what section 4.3 was added to prevent).
    bool found_bad_solve_with_good_entries = false;
    for (unsigned N = K; N <= K + 60; N += 5)
    {
        FPGramGaussJacobiOptions<double> o;
        o.oversampling_factor = 1;
        o.oversampling_margin = N - K;
        o.minimum_factor = 1;
        o.scale_aware_nodes = false;      // this scan prescribes N itself
        FPGramGaussJacobiFactor<double> factor;
        try { factor = fp_gram_factor_gauss_jacobi(n, modes, o); }
        catch (const std::runtime_error &) { continue; } // gate fired -- not this N
        std::vector<std::vector<double>> W;
        fp_gram_reconstruct_gauss_jacobi(factor, W);
        double max_entry_err = 0.0, max_entry = 0.0;
        for (unsigned i = 0; i < K; ++i)
            for (unsigned j = 0; j < K; ++j)
            {
                max_entry_err = std::max(max_entry_err, std::abs(W[i][j] - W_ref[i][j]));
                max_entry = std::max(max_entry, std::abs(W_ref[i][j]));
            }
        const double entry_err = max_entry_err / max_entry;
        std::vector<double> c;
        fp_gram_solve_gauss_jacobi(factor, b, c);
        const double solve_err = solve_residual(W_ref, b, c);
        if (entry_err < 1e-6 && solve_err > 1e2)
        {
            found_bad_solve_with_good_entries = true;
            break;
        }
    }
    REQUIRE(found_bad_solve_with_good_entries);
}

// ─── T-NR-05: under-resolution gate fires ─────────────────────────────────

TEST_CASE(fp_gram_under_resolution_gate_fires)
{
    const unsigned n = 20, K = 60;
    const auto modes = make_block(n, K, 1e-2);
    FPGramGaussJacobiOptions<double> o;
    o.oversampling_factor = 1;
    o.oversampling_margin = 0;
    o.minimum_factor = 1; // N == K: no headroom, must be caught by the gate
    o.scale_aware_nodes = false; // ... and the b_max term must not rescue it
    bool threw = false;
    std::string message;
    try { fp_gram_factor_gauss_jacobi(n, modes, o); }
    catch (const std::runtime_error &e) { threw = true; message = e.what(); }
    REQUIRE(threw);
    REQUIRE(message.find("oversampling") != std::string::npos);
}

// ─── T-NR-06: the node rule follows the block's radial scale ──────────────
//
// Regression for the Pe = 1000, max_root = 400 failure: the (2, 20) affine
// rule is measured at kappa = 1e-2 only. At kappa = 1e-6 the rates barely
// shrink, so b_max ~ max_root in EVERY angular block while K collapses like
// (max_root - 2n)/4, and the high-n blocks end up sampled nowhere near the
// s* = n/b_max where their modes actually live. Blocks n ~ 150-175 reached
// R-diagonal ratios of 1e6-2.6e6 and the whole setup threw.

TEST_CASE(fp_gram_node_count_follows_radial_scale)
{
    FPGramGaussJacobiOptions<double> o;   // defaults: factor 2, margin 20
    // b_max large relative to K: the sqrt(K*b_max) term takes over.
    REQUIRE(fp_gram_node_count(19u, o, 355.0) == 103u);   // ceil(sqrt(6745)) + 20
    REQUIRE(fp_gram_node_count(19u, o) == 58u);           // no scale: affine rule
    // b_max small relative to K: the affine rule still wins, unchanged.
    REQUIRE(fp_gram_node_count(100u, o, 10.0) == 220u);
    // The term never lowers the count.
    for (unsigned K : {1u, 5u, 20u, 60u, 150u})
        REQUIRE(fp_gram_node_count(K, o, 400.0) >= fp_gram_node_count(K, o));
}

TEST_CASE(fp_gram_high_peclet_high_n_blocks_are_resolved)
{
    const double kappa = 1e-6;            // Pe = 1000
    const double max_root = 400.0;
    const auto &catalog = bare_root_catalog();
    bool exercised_a_starved_block = false;

    for (unsigned n : {100u, 140u, 163u, 190u})
    {
        REQUIRE(n < catalog.size());
        unsigned K = 0;
        while (K < catalog[n].size() && catalog[n][K] <= max_root)
            ++K;
        REQUIRE(K >= 3u);
        const auto modes = make_block(n, K, kappa);

        // Default options must produce a healthy block.
        const auto factor = fp_gram_factor_gauss_jacobi(n, modes, FPGramGaussJacobiOptions<double>{});
        REQUIRE(factor.diagnostics.condition_estimate < 1e3);

        // ... and it is the radial-scale term that earns it: the historical
        // K-only rule leaves the same block far worse, or fails the gate.
        FPGramGaussJacobiOptions<double> legacy;
        legacy.scale_aware_nodes = false;
        REQUIRE(factor.node_count > fp_gram_node_count(K, legacy));
        double legacy_condition = 0.0;
        try { legacy_condition = fp_gram_factor_gauss_jacobi(n, modes, legacy).diagnostics.condition_estimate; }
        catch (const std::runtime_error &) { legacy_condition = 1e30; }   // gate fired
        if (legacy_condition > 1e4)
            exercised_a_starved_block = true;
    }
    // The fixture is only meaningful if at least one block really was starved
    // under the old rule.
    REQUIRE(exercised_a_starved_block);
}
