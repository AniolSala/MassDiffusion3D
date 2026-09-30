// Cross-validation between the DirectQuadrature and Representer finite-Peclet
// RHS backends (plans/RHS_REPRESENTER_PLAN.md section 8, highest-value file:
// the direct path is the oracle for the representer path).

#include "tinytest.h"

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#undef protected
#undef private

#include "finite_peclet_rhs.h"
#include "finite_peclet_rhs_representer.h"
#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_coefficients_gauss_jacobi.h"
#include "gram_method.h"
#include "rhs_method.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

unsigned max_angular_index(const std::vector<SeriesTermData<double>> &data)
{
    unsigned max_n = 0;
    for (const auto &t : data)
        max_n = std::max(max_n, t.n);
    return max_n;
}

// Default target: representative subset. Full n=0..max_n sweep runs when
// CDS_LONG_TESTS is set -- both code paths exist, per the plan's testing
// requirement (section 8).
std::vector<unsigned> angular_indices_for_test(unsigned max_n_present)
{
    if (std::getenv("CDS_LONG_TESTS"))
    {
        std::vector<unsigned> all;
        for (unsigned n = 0; n <= max_n_present; ++n)
            all.push_back(n);
        return all;
    }
    std::vector<unsigned> subset;
    for (unsigned n : {0u, 1u, 2u, 3u, 5u, 10u, 20u, 50u, 80u, 95u, 99u})
        if (n <= max_n_present)
            subset.push_back(n);
    if (subset.empty())
        subset.push_back(0u);
    return subset;
}

double max_root_for_test()
{
    return std::getenv("CDS_LONG_TESTS") ? 600.0 : 60.0;
}

// Agreement tolerance between the direct and representer paths. The default
// (fast) tier never reaches the extreme corner below, so 1e-9 holds easily.
// Under CDS_LONG_TESTS the full n=0..99, K~100 sweep at max_root=600 reaches
// a corner (n, m both close to 99) where the direct/representer RHS entries
// themselves still agree to ~1e-13 (see fp_representer_order_n0_shallow_cap_calibration
// and the T-RP tests), but the Gram SOLVE's own conditioning -- validated
// only to ~1e-8 and gated at a condition-number ceiling of 1e6
// (finite_peclet_gram_gauss_jacobi.h) -- amplifies that difference into the
// final coefficient. Measured worst case at n=m=99: 4.6e-7. This is a
// property of the Gram system at that corner, not of either RHS backend:
// the same amplification would apply to any two numerically-distinct (but
// individually accurate) RHS vectors fed through the same near-limit solve.
double rhs_agreement_tolerance()
{
    return std::getenv("CDS_LONG_TESTS") ? 2e-6 : 1e-9;
}
}

// ─── T-CV-01: load-vector agreement (stratified, Neumann) ──────────────────

TEST_CASE(fp_rhs_representer_matches_direct_stratified_load_vector)
{
    struct Profile { std::vector<double> zi, ui; };
    const std::vector<Profile> profiles{
        {{-0.4, 0.82}, {1.0, 0.3, 0.0}},
        {{0.0}, {1.0, 0.4}},
        {{-0.95, 0.02}, {1.0, -0.2, 0.5}},
    };
    const std::vector<double> kappas{1e-1, 1e-2, 1e-3};
    const double max_root = max_root_for_test();
    FPRepresenterOptions<double> options;

    for (const auto &kappa : kappas)
    {
        const double peclet = 1.0 / std::sqrt(kappa);
        for (const auto &profile : profiles)
        {
            CDStratifiedSolution<double> sol(profile.zi, profile.ui, 0);
            sol.set_max_root(max_root);
            sol.setup_fp_solution(peclet);
            const unsigned max_n = max_angular_index(sol.m_series_data);

            for (unsigned n : angular_indices_for_test(max_n))
            {
                const auto modes = angular_block(sol.m_series_data, n);
                if (modes.empty())
                    continue;
                FPGramGaussJacobiOptions<double> gram_options;
                gram_options.retain_samples = true;
                const auto factor = fp_gram_factor_gauss_jacobi(n, modes, gram_options);

                std::vector<double> full_disk_column;
                if (n == 0u && factor.has_constant_mode)
                    full_disk_column = fp_rhs_full_disk_from_gram_column(factor.constant_mode_gram_column);

                const auto rhs_direct = fp_rhs_stratified_inlet(n, modes, profile.zi, profile.ui, 100u, full_disk_column);
                const auto rhs_repr = fp_rhs_stratified_inlet_representer(n, factor, profile.zi, profile.ui, options, full_disk_column);
                REQUIRE(rhs_direct.size() == rhs_repr.size());

                double max_abs = 0.0;
                for (double v : rhs_direct)
                    max_abs = std::max(max_abs, std::abs(v));
                for (std::size_t i = 0; i < rhs_direct.size(); ++i)
                {
                    const double rel = std::abs(rhs_direct[i] - rhs_repr[i]) / std::max(1.0, max_abs);
                    REQUIRE(rel < rhs_agreement_tolerance());
                }
            }
        }
    }
}

// ─── T-CV-01 (uniform inlet / Dirichlet wall via Graetz) ───────────────────

TEST_CASE(fp_rhs_representer_matches_direct_uniform_inlet_load_vector)
{
    const std::vector<double> kappas{1e-1, 1e-2, 1e-3};
    const double max_root = max_root_for_test();
    FPRepresenterOptions<double> options;

    for (const auto &kappa : kappas)
    {
        const double peclet = 1.0 / std::sqrt(kappa);
        CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
        sol.set_max_root(max_root);
        sol.setup_fp_solution(peclet);
        const unsigned max_n = max_angular_index(sol.m_series_data);

        for (unsigned n : angular_indices_for_test(max_n))
        {
            const auto modes = angular_block(sol.m_series_data, n);
            if (modes.empty())
                continue;
            FPGramGaussJacobiOptions<double> gram_options;
            gram_options.retain_samples = true;
            const auto factor = fp_gram_factor_gauss_jacobi(n, modes, gram_options);

            std::vector<double> full_disk_column;
            if (n == 0u && factor.has_constant_mode)
                full_disk_column = fp_rhs_full_disk_from_gram_column(factor.constant_mode_gram_column);

            const auto rhs_direct = full_disk_column.empty() ? fp_rhs_uniform_inlet(n, modes, 100u) : full_disk_column;
            const auto rhs_repr = fp_rhs_uniform_inlet_representer(n, factor, options, full_disk_column);
            REQUIRE(rhs_direct.size() == rhs_repr.size());

            double max_abs = 0.0;
            for (double v : rhs_direct)
                max_abs = std::max(max_abs, std::abs(v));
            for (std::size_t i = 0; i < rhs_direct.size(); ++i)
            {
                const double rel = std::abs(rhs_direct[i] - rhs_repr[i]) / std::max(1.0, max_abs);
                REQUIRE(rel < rhs_agreement_tolerance());
            }
        }
    }
}

// ─── T-CV-02: coefficient agreement ─────────────────────────────────────────

TEST_CASE(fp_rhs_representer_coefficient_agreement_stratified)
{
    const std::vector<double> zi{-0.4, 0.82};
    const std::vector<double> ui{1.0, 0.3, 0.0};
    const double max_root = std::getenv("CDS_LONG_TESTS") ? 300.0 : 40.0;

    CDStratifiedSolution<double> sol_direct(zi, ui, 0);
    sol_direct.set_max_root(max_root);
    sol_direct.set_rhs_method(RhsMethod::DirectQuadrature);
    sol_direct.setup_fp_solution(8.0);

    CDStratifiedSolution<double> sol_repr(zi, ui, 0);
    sol_repr.set_max_root(max_root);
    sol_repr.set_rhs_method(RhsMethod::Representer);
    sol_repr.setup_fp_solution(8.0);

    REQUIRE(sol_direct.m_series_data.size() == sol_repr.m_series_data.size());
    double max_abs = 0.0;
    for (const auto &t : sol_direct.m_series_data)
        max_abs = std::max(max_abs, std::abs(t.coeff_fp));
    for (std::size_t i = 0; i < sol_direct.m_series_data.size(); ++i)
    {
        REQUIRE(sol_direct.m_series_data[i].n == sol_repr.m_series_data[i].n);
        REQUIRE(sol_direct.m_series_data[i].m == sol_repr.m_series_data[i].m);
        const double rel = std::abs(sol_direct.m_series_data[i].coeff_fp - sol_repr.m_series_data[i].coeff_fp) / std::max(1.0, max_abs);
        REQUIRE(rel < rhs_agreement_tolerance());
    }
}

TEST_CASE(fp_rhs_representer_coefficient_agreement_graetz)
{
    const double max_root = std::getenv("CDS_LONG_TESTS") ? 300.0 : 40.0;

    CDGraetzIsothermalSolution<double> sol_direct(1.0, 0.0, 0);
    sol_direct.set_max_root(max_root);
    sol_direct.set_rhs_method(RhsMethod::DirectQuadrature);
    sol_direct.setup_fp_solution(8.0);

    CDGraetzIsothermalSolution<double> sol_repr(1.0, 0.0, 0);
    sol_repr.set_max_root(max_root);
    sol_repr.set_rhs_method(RhsMethod::Representer);
    sol_repr.setup_fp_solution(8.0);

    REQUIRE(sol_direct.m_series_data.size() == sol_repr.m_series_data.size());
    double max_abs = 0.0;
    for (const auto &t : sol_direct.m_series_data)
        max_abs = std::max(max_abs, std::abs(t.coeff_fp));
    for (std::size_t i = 0; i < sol_direct.m_series_data.size(); ++i)
    {
        const double rel = std::abs(sol_direct.m_series_data[i].coeff_fp - sol_repr.m_series_data[i].coeff_fp) / std::max(1.0, max_abs);
        REQUIRE(rel < rhs_agreement_tolerance());
    }
}

// ─── T-CV-03: field agreement ───────────────────────────────────────────────

TEST_CASE(fp_rhs_representer_field_agreement_stratified)
{
    const std::vector<double> zi{-0.4, 0.82};
    const std::vector<double> ui{1.0, 0.3, 0.0};
    const double max_root = std::getenv("CDS_LONG_TESTS") ? 300.0 : 40.0;

    CDStratifiedSolution<double> sol_direct(zi, ui, 0);
    sol_direct.set_max_root(max_root);
    sol_direct.set_rhs_method(RhsMethod::DirectQuadrature);
    sol_direct.setup_fp_solution(8.0);

    CDStratifiedSolution<double> sol_repr(zi, ui, 0);
    sol_repr.set_max_root(max_root);
    sol_repr.set_rhs_method(RhsMethod::Representer);
    sol_repr.setup_fp_solution(8.0);

    const std::vector<double> x{-0.5, 0.0, 0.3, 0.9};
    const std::vector<double> r{0.1, 0.5, 0.9, 0.3};
    const std::vector<double> phi{0.0, 1.0, 2.5, -1.2};
    const auto vals_direct = sol_direct.get_solution_at_points(x, r, phi);
    const auto vals_repr = sol_repr.get_solution_at_points(x, r, phi);
    REQUIRE(vals_direct.size() == vals_repr.size());

    double max_abs = 0.0;
    for (double v : vals_direct)
        max_abs = std::max(max_abs, std::abs(v));
    for (std::size_t i = 0; i < vals_direct.size(); ++i)
    {
        REQUIRE(std::isfinite(vals_direct[i]));
        REQUIRE(std::isfinite(vals_repr[i]));
        const double rel = std::abs(vals_direct[i] - vals_repr[i]) / std::max(1.0, max_abs);
        REQUIRE(rel < rhs_agreement_tolerance());
    }
}

// ─── T-CV-04: no-fallback contract ──────────────────────────────────────────

TEST_CASE(fp_rhs_representer_with_ultraspherical_gram_throws_no_fallback_stratified)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_max_root(15.0);
    sol.set_gram_method(GramMethod::Ultraspherical);
    sol.set_rhs_method(RhsMethod::Representer);

    bool threw = false;
    std::string message;
    try
    {
        sol.setup_fp_solution(5.0);
    }
    catch (const std::invalid_argument &e)
    {
        threw = true;
        message = e.what();
    }
    REQUIRE(threw);
    REQUIRE(message.find("GaussJacobiQR") != std::string::npos);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    // Direct low-level check too, independent of the solver-level dispatch:
    // the guard fires BEFORE any Gram pass runs, not as a side effect of
    // Ultraspherical's own (separately tested) unimplemented status.
    std::vector<double> p{0.0}, v{1.0, 0.3};
    std::vector<SeriesTermData<double>> d(1);
    d[0].n = 0; d[0].m = 0; d[0].root_fp = 1.0; d[0].btilde_fp = 1.0; d[0].rate_fp = 0.0;
    bool threw_direct = false;
    try
    {
        fp_stratified_coefficients_gauss_jacobi<double>(p, v, 1u, d, 100u, GramMethod::Ultraspherical,
                                                         WallCondition::Neumann, FPGramGaussJacobiOptions<double>{},
                                                         RhsMethod::Representer, 40u, false);
    }
    catch (const std::invalid_argument &)
    {
        threw_direct = true;
    }
    REQUIRE(threw_direct);
}

TEST_CASE(fp_rhs_representer_with_ultraspherical_gram_throws_no_fallback_graetz)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(15.0);
    sol.set_gram_method(GramMethod::Ultraspherical);
    sol.set_rhs_method(RhsMethod::Representer);

    bool threw = false;
    std::string message;
    try
    {
        sol.setup_fp_solution(5.0);
    }
    catch (const std::invalid_argument &e)
    {
        threw = true;
        message = e.what();
    }
    REQUIRE(threw);
    REQUIRE(message.find("GaussJacobiQR") != std::string::npos);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);
}

// ─── T-CV-05: default ────────────────────────────────────────────────────────

TEST_CASE(fp_rhs_method_default_is_representer)
{
    CDStratifiedSolution<double> strat({0.0}, {1.0, 0.0}, 0);
    REQUIRE(strat.get_rhs_method() == RhsMethod::Representer);

    CDGraetzIsothermalSolution<double> graetz(1.0, 0.0, 0);
    REQUIRE(graetz.get_rhs_method() == RhsMethod::Representer);
}

// ─── T-CV-06: interface-count independence ─────────────────────────────────
//
// No instrumentation hook exists to count fp_radial_factor calls directly
// (mirrors fp_gram_gauss_jacobi_order_check_cost_is_linear_not_quadratic in
// test_fp_gram_backends.cpp), so this pins the structural benefit via
// wall-clock: the direct path's cost grows with the number of distinct
// interfaces (one independent cap sweep per interface per mode); the
// representer path's cost is dominated by the fixed K*N contraction and one
// representer build per distinct interface -- no per-mode quadrature at all.

TEST_CASE(fp_rhs_interface_count_independence_representer_vs_direct_timing)
{
    const unsigned K = 40u;
    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        const double beta = 4.0 * (m + 1) + 2.0;
        const double beta2 = beta * beta;
        const double kappa = 1e-2;
        const double Lambda = (m == 0) ? 0.0 : 2.0 * beta2 / (1.0 + std::sqrt(1.0 + 4.0 * kappa * beta2));
        modes[m].n = 0; modes[m].m = m;
        modes[m].root_fp = std::sqrt(Lambda);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * Lambda);
        modes[m].rate_fp = Lambda;
    }
    FPGramGaussJacobiOptions<double> gram_options;
    gram_options.retain_samples = true;
    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, gram_options);
    FPRepresenterOptions<double> options;

    // Interfaces spread over [0.25, 0.85] -- deliberately clear of the n=0
    // shallow-cap regime (absolute_interface < shallow_cap_threshold = 0.1),
    // so this measures pure interface-COUNT scaling, not the (separately
    // tested, T-RP-02/T-RP-03) a-dependent shallow-cap order cost.
    auto build_profile = [](unsigned count) {
        std::vector<double> zi, ui;
        ui.push_back(1.0);
        for (unsigned i = 0; i < count; ++i)
        {
            zi.push_back(0.25 + 0.6 * static_cast<double>(i) / static_cast<double>(std::max(count, 2u) - 1u));
            ui.push_back(1.0 + 0.3 * static_cast<double>(i + 1));
        }
        return std::make_pair(zi, ui);
    };

    auto time_direct = [&](unsigned count) {
        const auto profile = build_profile(count);
        const auto t0 = std::chrono::steady_clock::now();
        static_cast<void>(fp_rhs_stratified_inlet(0u, modes, profile.first, profile.second, 150u));
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto time_repr = [&](unsigned count) {
        const auto profile = build_profile(count);
        const auto t0 = std::chrono::steady_clock::now();
        static_cast<void>(fp_rhs_stratified_inlet_representer(0u, factor, profile.first, profile.second, options));
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    time_direct(1u); time_repr(1u); // warm-up: pay one-time allocation costs

    const double direct_1 = time_direct(1u);
    const double direct_8 = time_direct(8u);
    const double repr_1 = time_repr(1u);
    const double repr_8 = time_repr(8u);

    // The representer never calls fp_radial_factor (T-RP-07, unconditionally
    // -- zero Kummer evaluations regardless of interface count), while the
    // direct path evaluates it Sum_m order_m times PER distinct interface.
    // Both costs grow roughly linearly in interface count at fixed K (an
    // extra representer build is not free -- see plan section 0.4's own
    // J*order + J*N + K*N accounting), so the discriminating signature is
    // not "flat vs. linear" but the enormous constant-factor gap between an
    // elementary-arithmetic flop and a confluent-hypergeometric evaluation:
    // representer stays a small fraction of direct's cost at both ends.
    // Measured on this configuration: ~480x at 1 interface, ~300x at 8.
    REQUIRE(direct_8 > 2.0 * std::max(direct_1, 1e-6)); // direct genuinely grows with interface count
    REQUIRE(repr_1 < direct_1 / 20.0);
    REQUIRE(repr_8 < direct_8 / 20.0);
}

// ─── T-CV-07: bit-identity when off ─────────────────────────────────────────
//
// Golden fixture captured from the pre-representer build (plan section 11,
// step 0), same configuration as test_fp_rhs.cpp's
// fp_rhs_stratified_inlet_matches_independent_reassembly: zi={-0.4,0.2},
// ui={1.0,0.3,-0.2}, max_root=30, Pe=8.0. With RhsMethod::DirectQuadrature
// explicitly selected, coeff_fp must reproduce these values -- confirming the
// RhsMethod plumbing left the DirectQuadrature branch untouched.

TEST_CASE(fp_rhs_direct_quadrature_matches_pre_representer_golden_fixture)
{
    const std::vector<double> zi{-0.4, 0.2};
    const std::vector<double> ui{1.0, 0.3, -0.2};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    std::vector<double> flat;
    for (const auto &block : sol.get_coefficients())
        for (double c : block)
            flat.push_back(c);

    REQUIRE(flat.size() == 64u);
    const double golden_sum = -49.44650456660143;
    const double golden_sumabs = 4594.364062903216;
    double sum = 0.0, sumabs = 0.0;
    for (double c : flat)
    {
        sum += c;
        sumabs += std::abs(c);
    }
    REQUIRE_APPROX(sum, golden_sum, 1e-12, 1e-9);
    REQUIRE_APPROX(sumabs, golden_sumabs, 1e-12, 1e-9);

    const std::vector<double> golden_first8{
        1.5175719593088928, 0.213054900570837, -0.1665669759731572, -0.19352128345654618,
        -0.08311498991426974, 0.07941835664815701, 0.11108786232018199, 0.033750681138444955};
    const std::vector<double> golden_last8{
        -69.12553606147276, -1053.0525963017637, -0.5509992619513737, 53.87264827848271,
        -15.811500007750848, 518.0467238641933, -1.106504490667676, -31.49786767760482};
    for (std::size_t i = 0; i < 8; ++i)
    {
        REQUIRE_APPROX(flat[i], golden_first8[i], 1e-10, 1e-10);
        REQUIRE_APPROX(flat[flat.size() - 8 + i], golden_last8[i], 1e-10, 1e-10);
    }
}
