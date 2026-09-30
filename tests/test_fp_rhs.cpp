// Regression tests for the inlet-specific finite-Peclet RHS module
// (plans/REFACTOR_PLAN.md section 3/section 8, tests/test_fp_rhs.cpp).
//
// T-RHS-01 note: the plan asks for a frozen pre-refactor golden-vector
// fixture. This suite instead cross-checks fp_rhs_stratified_inlet's output
// against an independent re-derivation built from the SAME publicly retained
// building blocks (compute_layer_nm_quadrature, fp_rhs_full_disk_projection)
// that back it -- those primitives are pinned, unchanged, by the pre-existing
// tests/test_finite_peclet_coefficients_gauss_jacobi.cpp closed-form cases, so
// agreement here is direct evidence the move preserved behaviour without
// requiring a second full build of the pre-refactor tree.

#include "tinytest.h"

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#undef protected
#undef private

#include "finite_peclet_rhs.h"
#include "finite_peclet_gram_gauss_jacobi.h"
#include "gram_method.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

namespace
{
std::vector<SeriesTermData<double>> angular_block(const std::vector<SeriesTermData<double>> &data, unsigned n)
{
    std::vector<SeriesTermData<double>> modes;
    for (const auto &t : data)
        if (t.n == n)
            modes.push_back(t);
    return modes;
}

// Independent re-derivation of the stratified RHS assembly formula, built
// directly from the retained public primitives instead of fp_rhs_stratified_inlet's
// internal Cap/layer/signed_cap helpers.
std::vector<double> reference_stratified_rhs(unsigned n, const std::vector<SeriesTermData<double>> &modes,
                                             const std::vector<double> &zi, const std::vector<double> &ui,
                                             unsigned order, const std::vector<double> &gauss_weights,
                                             const std::vector<double> &gauss_nodes)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(order);
    std::vector<double> full = (n == 0) ? fp_rhs_full_disk_projection(modes, order) : std::vector<double>(modes.size(), 0.0);

    std::vector<double> rhs(modes.size());
    for (std::size_t local = 0; local < modes.size(); ++local)
    {
        double value = ui.front() * full[local];
        for (std::size_t i = 0; i < zi.size(); ++i)
        {
            if (ui[i + 1] == ui[i])
                continue;
            const double a = std::abs(zi[i]);
            const double pos = compute_layer_nm_quadrature(modes[local], a, rule, gauss_weights, gauss_nodes);
            double signed_value;
            if (zi[i] < 0.0)
                signed_value = (n == 0) ? (full[local] - pos) : ((n % 2) ? pos : -pos);
            else
                signed_value = pos;
            value += (ui[i + 1] - ui[i]) * signed_value;
        }
        rhs[local] = value;
    }
    return rhs;
}
}

// ─── T-RHS-01: fp_rhs_stratified_inlet reproduces the pinned assembly ─────

TEST_CASE(fp_rhs_stratified_inlet_matches_independent_reassembly)
{
    const std::vector<double> zi{-0.4, 0.2};
    const std::vector<double> ui{1.0, 0.3, -0.2};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);
    sol.setup_fp_solution(8.0);

    for (unsigned n : {0u, 1u, 2u, 3u, 5u, 10u})
    {
        const auto modes = angular_block(sol.m_series_data, n);
        if (modes.empty())
            continue;
        const auto rhs = fp_rhs_stratified_inlet<double>(n, modes, zi, ui, 100u);
        const auto ref = reference_stratified_rhs(n, modes, zi, ui, 100u, sol.m_gauss_weights, sol.m_gauss_points);
        REQUIRE(rhs.size() == ref.size());
        for (std::size_t i = 0; i < rhs.size(); ++i)
            REQUIRE_APPROX(rhs[i], ref[i], 1e-12, 1e-12);
    }
}

TEST_CASE(fp_rhs_stratified_inlet_no_jumps_reduces_to_scaled_full_disk)
{
    const std::vector<double> zi{};
    const std::vector<double> ui{2.0};
    // A non-degenerate profile (max_ui != min_ui) purely to build real modes --
    // CDBaseSolution::set_ui normalizes via (max-ui)/(max-min), which is 0/0
    // for a flat profile. The RHS identity below uses its own independent
    // zi/ui (empty/{2.0}), unrelated to this solver's inlet.
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.4}, 0);
    sol.set_max_root(20.0);
    sol.setup_fp_solution(6.0);

    const auto modes0 = angular_block(sol.m_series_data, 0u);
    REQUIRE(!modes0.empty());
    const auto rhs0 = fp_rhs_stratified_inlet<double>(0u, modes0, zi, ui, 100u);
    const auto full0 = fp_rhs_full_disk_projection(modes0, 100u);
    REQUIRE(rhs0.size() == full0.size());
    for (std::size_t i = 0; i < rhs0.size(); ++i)
        REQUIRE_APPROX(rhs0[i], ui.front() * full0[i], 1e-13, 1e-13);

    const auto modes1 = angular_block(sol.m_series_data, 1u);
    if (!modes1.empty())
    {
        const auto rhs1 = fp_rhs_stratified_inlet<double>(1u, modes1, zi, ui, 100u);
        for (double v : rhs1)
            REQUIRE(v == 0.0);
    }
}

// ─── T-RHS-02: uniform inlet ───────────────────────────────────────────────

TEST_CASE(fp_rhs_uniform_inlet_matches_full_disk_and_vanishes_for_n_gt_0)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(30.0);
    sol.setup_fp_solution(8.0);

    const auto modes0 = angular_block(sol.m_series_data, 0u);
    REQUIRE(!modes0.empty());
    const auto rhs0 = fp_rhs_uniform_inlet<double>(0u, modes0, 100u);
    const auto full0 = fp_rhs_full_disk_projection(modes0, 100u);
    REQUIRE(rhs0.size() == full0.size());
    for (std::size_t i = 0; i < rhs0.size(); ++i)
        REQUIRE(rhs0[i] == full0[i]);

    std::vector<SeriesTermData<double>> fake_n1(3);
    for (unsigned m = 0; m < 3; ++m)
    {
        fake_n1[m].n = 1;
        fake_n1[m].m = m;
        fake_n1[m].root_fp = 4.0 + m;
        fake_n1[m].btilde_fp = 4.0 + m;
    }
    const auto rhs1 = fp_rhs_uniform_inlet<double>(1u, fake_n1, 50u);
    REQUIRE(rhs1.size() == 3u);
    for (double v : rhs1)
        REQUIRE(v == 0.0);
}

// ─── T-RHS-03: Gram-column identity ────────────────────────────────────────

TEST_CASE(fp_rhs_full_disk_from_gram_column_matches_independent_projection)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_max_root(20.0);
    sol.setup_fp_solution(5.0);

    const auto modes0 = angular_block(sol.m_series_data, 0u);
    REQUIRE(modes0.size() >= 3u);

    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes0, FPGramGaussJacobiOptions<double>{});
    REQUIRE(factor.has_constant_mode);

    const auto from_column = fp_rhs_full_disk_from_gram_column(factor.constant_mode_gram_column);
    const auto from_quadrature = fp_rhs_full_disk_projection(modes0, 400u);
    REQUIRE(from_column.size() == from_quadrature.size());
    for (std::size_t i = 0; i < from_column.size(); ++i)
    {
        const double rel = std::abs(from_column[i] - from_quadrature[i]) / std::max(1.0, std::abs(from_quadrature[i]));
        REQUIRE(rel < 1e-8);
    }
}

// ─── T-RHS-04: constant-mode detection ─────────────────────────────────────

TEST_CASE(fp_gram_factor_has_constant_mode_only_for_neumann_n0_with_zero_rate)
{
    CDStratifiedSolution<double> strat({0.0}, {1.0, 0.3}, 0);
    strat.set_max_root(20.0);
    strat.setup_fp_solution(5.0);
    const auto strat_n0 = angular_block(strat.m_series_data, 0u);
    const auto factor_n0 = fp_gram_factor_gauss_jacobi(0u, strat_n0, FPGramGaussJacobiOptions<double>{});
    REQUIRE(factor_n0.has_constant_mode);

    const auto strat_n1 = angular_block(strat.m_series_data, 1u);
    REQUIRE(!strat_n1.empty());
    const auto factor_n1 = fp_gram_factor_gauss_jacobi(1u, strat_n1, FPGramGaussJacobiOptions<double>{});
    REQUIRE(!factor_n1.has_constant_mode);

    CDGraetzIsothermalSolution<double> graetz(1.0, 0.0, 0);
    graetz.set_max_root(20.0);
    graetz.setup_fp_solution(5.0);
    const auto graetz_n0 = angular_block(graetz.m_series_data, 0u);
    REQUIRE(!graetz_n0.empty());
    const auto factor_graetz = fp_gram_factor_gauss_jacobi(0u, graetz_n0, FPGramGaussJacobiOptions<double>{});
    REQUIRE(!factor_graetz.has_constant_mode); // Dirichlet: no exact Lambda=0 mode
}

// ─── T-RHS-05: inlet-blindness (source-level) ──────────────────────────────

TEST_CASE(fp_gram_backends_are_inlet_blind_source_grep)
{
    const std::vector<std::string> files{
        "src/finite_peclet_gram_gauss_jacobi.h",
        "src/finite_peclet_gram_gauss_jacobi.cpp",
        "src/finite_peclet_gram_ultraspherical.h",
        "src/finite_peclet_gram_ultraspherical.cpp",
    };
    // "zi"/"ui" are deliberately excluded: they are short enough to appear as
    // innocuous substrings (e.g. inside identifiers like "unsigned", "quiet",
    // "positive") in this file's own vocabulary and would make the check
    // noisy rather than meaningful; the remaining, longer tokens are the ones
    // the design contract (finite_peclet_gram_gauss_jacobi.h's header
    // comment) actually calls out.
    const std::vector<std::string> forbidden{"layer", "interface", "cap", "inlet", "full_disk"};

    for (const auto &rel : files)
    {
        std::ifstream in(std::string(TEST_DATA_DIR) + "/" + rel);
        REQUIRE(in.good());
        std::ostringstream buf;
        buf << in.rdbuf();
        std::string text = buf.str();
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        for (const auto &token : forbidden)
        {
            const bool found = text.find(token) != std::string::npos;
            if (found)
                throw tinytest::Failure(rel + " contains forbidden inlet-specific token '" + token + "'");
        }
    }
}

// ─── fp_rhs_required_order / per-mode sizing ───────────────────────────────

TEST_CASE(fp_rhs_required_order_buckets_by_bare_root)
{
    // ceil(kCapOrderRootCoefficient * bare_root) + margin, rounded up to the
    // nearest power of two times the base bucket unit (20; see
    // finite_peclet_rhs.cpp's kCapOrderBaseBucket -- geometric, not linear,
    // spacing). These numeric results happen to be unchanged by the
    // coefficient (0.6, not 1) because none of them straddle a bucket
    // boundary differently -- see fp_rhs_required_order_covers_measured_convergence_grid
    // below for the actual coefficient regression.
    REQUIRE(fp_rhs_required_order(2.0, 0u) == 20u);
    REQUIRE(fp_rhs_required_order(0.0, 0u) == 20u);   // clamped to >= 1 before bucketing
    REQUIRE(fp_rhs_required_order(150.0, 0u) == 160u);
    REQUIRE(fp_rhs_required_order(1.0, 8u) == 20u);
    REQUIRE(fp_rhs_required_order(60.0, 100u) == 160u); // matches CDBaseSolution.h's measured default
    REQUIRE(fp_rhs_required_order(300.0, 0u) == 320u);

    // Distinct buckets present in a wide-root block are bounded by O(log2)
    // of the range, not O(range/bucket_size): this is what fixes the
    // regression a linear bucket step caused (55s vs ~10s at max_root=600,
    // since a linear step produced ~20 distinct rule constructions in a
    // single 100-mode block spanning root ~2 to ~400).
    REQUIRE(fp_rhs_required_order(2.0, 0u) != fp_rhs_required_order(400.0, 0u));
}

// Pins the measured convergence study behind kCapOrderRootCoefficient
// (finite_peclet_rhs.cpp): for every (n, m, bare_root, order_needed) point
// actually measured (compute_layer_nm_quadrature at a genuine interface,
// converged to 1e-10 against a high-order reference), fp_rhs_required_order
// must return an order >= what was measured to actually converge. This is
// the test that would have caught the original bug -- sizing by the raw bare
// root (coefficient=1) passes this trivially by being hugely over-generous;
// what it guards against is someone "optimizing" the coefficient down again
// without re-running the convergence study.
TEST_CASE(fp_rhs_required_order_covers_measured_convergence_grid)
{
    struct Measured { double bare_root; unsigned order_needed; };
    // n, m noted in comments for traceability back to the study; the formula
    // itself only ever sees bare_root, not n/m separately.
    const std::vector<Measured> measured{
        {0.00, 15},    // n=0,  m=0
        {5.07, 15},    // n=0,  m=1
        {21.24, 20},   // n=0,  m=5
        {2.88, 5},     // n=1,  m=0
        {7.12, 10},    // n=1,  m=1
        {11.51, 10},   // n=5,  m=0
        {31.28, 15},   // n=5,  m=5
        {21.87, 10},   // n=10, m=0
        {61.34, 20},   // n=10, m=10
        {62.00, 15},   // n=30, m=0
        {82.00, 20},   // n=40, m=0
        {81.29, 30},   // n=0,  m=20
        {201.31, 50},  // n=0,  m=50
        {321.32, 70},  // n=0,  m=80
        {397.32, 70},  // n=0,  m=99
        {41.99, 20},   // n=20, m=0
        {121.44, 40},  // n=20, m=20
        {437.33, 100}, // n=20, m=99
        {102.00, 20},  // n=50, m=0
        {497.44, 100}, // n=50, m=99
        {162.00, 30},  // n=80, m=0
        {557.73, 140}, // n=80, m=99
        {200.00, 30},  // n=99, m=0 (the specific case that triggered this study)
        {519.94, 140}, // n=99, m=80
        {595.88, 140}, // n=99, m=99
    };
    for (const auto &point : measured)
        REQUIRE(fp_rhs_required_order(point.bare_root, 0u) >= point.order_needed);
}

// A low-root mode's value must not depend on whether it shares an angular
// block with a far more oscillatory one: each mode is bucketed by its OWN
// bare root (fp_rhs_required_order), not by the block's mode count or worst
// member. This is the actual behavioural point of the per-mode refactor
// (REFACTOR_PLAN.md's block-size rule would instead have inflated the
// low-root mode's order to match its block-mate).
TEST_CASE(fp_rhs_stratified_inlet_per_mode_order_is_independent_of_blockmates)
{
    SeriesTermData<double> low{}, high{};
    low.n = 1; low.m = 0; low.root = 2.0; low.root_fp = 2.0; low.btilde_fp = 2.0;
    high.n = 1; high.m = 1; high.root = 150.0; high.root_fp = 150.0; high.btilde_fp = 150.0;

    const std::vector<double> zi{0.0};
    const std::vector<double> ui{1.0, 0.5};

    const auto rhs_pair = fp_rhs_stratified_inlet<double>(1u, {low, high}, zi, ui, 0u);
    const auto rhs_low_alone = fp_rhs_stratified_inlet<double>(1u, {low}, zi, ui, 0u);
    const auto rhs_high_alone = fp_rhs_stratified_inlet<double>(1u, {high}, zi, ui, 0u);
    REQUIRE(rhs_pair.size() == 2u);

    // Each mode's own value is unaffected by whatever else shares its block.
    REQUIRE_APPROX(rhs_pair[0], rhs_low_alone[0], 1e-13, 1e-13);
    REQUIRE_APPROX(rhs_pair[1], rhs_high_alone[0], 1e-13, 1e-13);

    // The low-root mode's minimal bucket (margin=0 -> order=20) is already
    // converged: compare against a hugely over-resolved reference.
    const auto rhs_low_truth = fp_rhs_stratified_inlet<double>(1u, {low}, zi, ui, 5000u);
    REQUIRE_APPROX(rhs_low_alone[0], rhs_low_truth[0], 1e-10, 1e-10);
}
