// Focused regression tests for the dual Gram-matrix backend selector
// (plans/IMPLEMENTATION_PLAN_GJQR_ultraspherical.md).
//
// Scope (deliberately reduced from the full plan for this pass): GaussJacobiQR
// is the production default and is verified against an independent brute-force
// quadrature reference; GramMethod::Ultraspherical is not yet implemented and
// must fail loudly with no silent fallback to the other backend.

#include "tinytest.h"

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#undef protected
#undef private

#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_gram_ultraspherical.h"
#include "finite_peclet_radial.h"
#include "gram_method.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
// Independent reference for W^n_mk = 1/2 int_0^1 (1-s) s^n G_m(s) G_k(s) ds,
// deliberately not sharing any code with either Gram backend.
double reference_gram_entry(unsigned n, const SeriesTermData<double> &a, const SeriesTermData<double> &b)
{
    const int N = 20000; // even, composite Simpson
    const double h = 1.0 / N;
    double sum = 0.0;
    for (int q = 0; q <= N; ++q)
    {
        const double s = q * h;
        const double weight = (q == 0 || q == N) ? 1.0 : (q % 2 == 0 ? 2.0 : 4.0);
        const double gi = fp_radial_factor(n, a.root_fp, a.btilde_fp, s);
        const double gj = fp_radial_factor(n, b.root_fp, b.btilde_fp, s);
        sum += weight * (1.0 - s) * std::pow(s, static_cast<double>(n)) * gi * gj;
    }
    return 0.5 * sum * h / 3.0;
}

// Synthetic n=0, K=60 block for the T-GB-NEW cost regression below --
// independent of any solver/root-finding machinery, matching the block size
// used in REFACTOR_PLAN.md section 5.1's defect-1 measurement (K=60, N=140).
std::vector<SeriesTermData<double>> synthetic_block(unsigned n, unsigned K, double kappa)
{
    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        const double beta = 4.0 * (m + 1) + 2.0 * n + 2.0;
        const double beta2 = beta * beta;
        const double Lambda = (n == 0 && m == 0) ? 0.0 : 2.0 * beta2 / (1.0 + std::sqrt(1.0 + 4.0 * kappa * beta2));
        modes[m].n = n;
        modes[m].m = m;
        modes[m].root_fp = std::sqrt(Lambda);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * Lambda);
        modes[m].rate_fp = Lambda;
    }
    return modes;
}
} // namespace

// ─── T-CV-05: default backend is GaussJacobiQR ───────────────────────────────

TEST_CASE(fp_gram_default_method_is_gauss_jacobi_qr)
{
    CDStratifiedSolution<double> strat({0.0}, {1.0, 0.0}, 0);
    REQUIRE(strat.get_gram_method() == GramMethod::GaussJacobiQR);

    CDGraetzIsothermalSolution<double> graetz(1.0, 0.0, 0);
    REQUIRE(graetz.get_gram_method() == GramMethod::GaussJacobiQR);
}

// ─── T-CV-04: no-fallback contract ────────────────────────────────────────────
// GramMethod::Ultraspherical is not implemented yet (finite_peclet_gram_ultraspherical.cpp);
// selecting it must fail loudly and must never silently reuse GaussJacobiQR's numbers.

TEST_CASE(fp_gram_ultraspherical_selection_throws_and_leaves_uninitialized_stratified)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    sol.set_max_root(10.0);
    sol.set_gram_method(GramMethod::Ultraspherical);
    REQUIRE(sol.get_gram_method() == GramMethod::Ultraspherical);

    bool threw = false;
    try { sol.setup_fp_solution(5.0); }
    catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    // Switching back to the default backend must still work: the failed
    // Ultraspherical attempt must not have corrupted any cached state.
    sol.set_gram_method(GramMethod::GaussJacobiQR);
    sol.setup_fp_solution(5.0);
    REQUIRE(sol.get_solution_method() == SolutionMethod::ModifiedRoots);
}

TEST_CASE(fp_gram_ultraspherical_selection_throws_and_leaves_uninitialized_graetz)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(10.0);
    sol.set_gram_method(GramMethod::Ultraspherical);

    bool threw = false;
    try { sol.setup_fp_solution(5.0); }
    catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    sol.set_gram_method(GramMethod::GaussJacobiQR);
    sol.setup_fp_solution(5.0);
    REQUIRE(sol.get_solution_method() == SolutionMethod::ModifiedRoots);
}

TEST_CASE(fp_gram_ultraspherical_factor_throws_directly)
{
    // Direct low-level check: the backend itself throws, independent of the
    // solver-level dispatch tested above.
    std::vector<SeriesTermData<double>> modes(1);
    modes[0].n = 0; modes[0].m = 0;
    modes[0].root_fp = 1.0; modes[0].btilde_fp = 1.0; modes[0].rate_fp = 1.0;
    bool threw = false;
    try { fp_gram_factor_ultraspherical<double>(0u, modes, WallCondition::Neumann, FPGramUltrasphericalOptions<double>{}); }
    catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
}

// ─── Backend A correctness, verified against an independent reference ────────

TEST_CASE(fp_gram_gauss_jacobi_reconstructs_reference_gram)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_max_root(20.0);
    sol.setup_fp_solution(5.0);

    std::vector<SeriesTermData<double>> modes;
    for (const auto &term : sol.m_series_data)
        if (term.n == 0)
            modes.push_back(term);
    REQUIRE(modes.size() >= 3u);

    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, FPGramGaussJacobiOptions<double>{});
    std::vector<std::vector<double>> W;
    fp_gram_reconstruct_gauss_jacobi(factor, W);

    for (std::size_t i = 0; i < modes.size(); ++i)
        for (std::size_t j = 0; j < modes.size(); ++j)
        {
            const double ref = reference_gram_entry(0u, modes[i], modes[j]);
            const double rel = std::abs(W[i][j] - ref) / std::max(1.0, std::abs(ref));
            REQUIRE(rel < 1e-8);
        }
}

TEST_CASE(fp_gram_gauss_jacobi_solve_matches_reconstructed_system)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_max_root(20.0);
    sol.setup_fp_solution(5.0);

    std::vector<SeriesTermData<double>> modes;
    for (const auto &term : sol.m_series_data)
        if (term.n == 0)
            modes.push_back(term);
    REQUIRE(modes.size() >= 3u);

    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, FPGramGaussJacobiOptions<double>{});
    std::vector<std::vector<double>> W;
    fp_gram_reconstruct_gauss_jacobi(factor, W);

    std::vector<double> b(modes.size());
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = 1.0 + 0.3 * static_cast<double>(i);

    std::vector<double> c;
    fp_gram_solve_gauss_jacobi(factor, b, c);
    REQUIRE(c.size() == modes.size());

    for (std::size_t i = 0; i < modes.size(); ++i)
    {
        double acc = 0.0;
        for (std::size_t j = 0; j < modes.size(); ++j)
            acc += W[i][j] * c[j];
        REQUIRE_APPROX(acc, b[i], 1e-8, 1e-8);
    }
}

TEST_CASE(fp_gram_gauss_jacobi_kappa_to_zero_is_diagonal)
{
    // Very large Peclet -> kappa = 1/Pe^2 tiny -> the modified radial modes
    // approach the bare (orthogonal) basis, so off-diagonal Gram entries must
    // vanish relative to the diagonal.
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_max_root(15.0);
    sol.setup_fp_solution(1.0e6);

    std::vector<SeriesTermData<double>> modes;
    for (const auto &term : sol.m_series_data)
        if (term.n == 0)
            modes.push_back(term);
    REQUIRE(modes.size() >= 3u);

    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, FPGramGaussJacobiOptions<double>{});
    std::vector<std::vector<double>> W;
    fp_gram_reconstruct_gauss_jacobi(factor, W);

    for (std::size_t i = 0; i < modes.size(); ++i)
        for (std::size_t j = 0; j < modes.size(); ++j)
        {
            if (i == j)
                continue;
            const double ratio = std::abs(W[i][j]) / std::sqrt(std::abs(W[i][i] * W[j][j]));
            REQUIRE(ratio < 1e-6);
        }
}

// ─── T-GB-NEW: order-check cost regression (REFACTOR_PLAN.md section 5.1) ───
//
// raw_gram must evaluate fp_radial_factor N*K times, not N*K^2/2: the previous
// version evaluated the second mode's radial factor inside the inner (j <= i)
// loop, making enable_order_check's two raw_gram calls (N and N+40) roughly
// 60x slower for a K=60 block. No instrumentation hook exists to count calls
// directly, so this pins the O(N*K) fix via wall-clock: with the quadratic
// bug, N=140, K=60 costs ~256,200 radial evaluations per raw_gram call
// instead of ~8,400 -- a ~30x difference that the 5x wall-clock margin below
// comfortably distinguishes from measurement noise.
TEST_CASE(fp_gram_gauss_jacobi_order_check_cost_is_linear_not_quadratic)
{
    const auto modes = synthetic_block(0u, 60u, 1e-2);

    FPGramGaussJacobiOptions<double> without_check;
    without_check.enable_order_check = false;
    const auto t0 = std::chrono::steady_clock::now();
    static_cast<void>(fp_gram_factor_gauss_jacobi(0u, modes, without_check));
    const auto t1 = std::chrono::steady_clock::now();
    const double baseline_s = std::chrono::duration<double>(t1 - t0).count();

    FPGramGaussJacobiOptions<double> with_check;
    with_check.enable_order_check = true;
    const auto t2 = std::chrono::steady_clock::now();
    static_cast<void>(fp_gram_factor_gauss_jacobi(0u, modes, with_check));
    const auto t3 = std::chrono::steady_clock::now();
    const double checked_s = std::chrono::duration<double>(t3 - t2).count();

    // Guard against a near-zero baseline making the ratio noisy on a very fast
    // machine: the order check does two extra full raw_gram builds (N and
    // N+40), so its floor is a small positive constant even when factoring
    // itself is too fast to time reliably.
    REQUIRE(checked_s < 5.0 * std::max(baseline_s, 1e-4));
}
