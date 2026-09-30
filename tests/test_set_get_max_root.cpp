// Tests for set_max_root() / get_max_root().
//
// m_max_root is a method-independent cutoff on the BARE reference roots
// (see plans/finite_peclet_solution_architecture_refactor_plan.md §2.5, §6):
//  a) default value is exactly 400;
//  b) set_max_root stores the given value unchanged -- there is no snapping
//     to the nearest tabulated eigenvalue;
//  c) nonpositive and nonfinite cutoffs throw std::invalid_argument;
//  d) the retained mode set contains every bare root <= cutoff and none
//     above it (a root exactly equal to the cutoff is included);
//  e) changing the cutoff invalidates any active setup; setting the same
//     cutoff again does not;
//  f) rerunning a setup method after a cutoff change restores readiness
//     with the new mode count.

#include "tinytest.h"

#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#include "../src/CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// ─── default value ──────────────────────────────────────────────────────────

TEST_CASE(test_get_max_root_default_is_400)
{
    CDGraetzIsothermalSolution<double> sol_g(1.0, 0.0, 0);
    REQUIRE_APPROX(sol_g.get_max_root(), 400.0, 0.0, 1e-12);

    CDStratifiedSolution<double> sol_s({0.0}, {1.0, 0.0}, 0);
    REQUIRE_APPROX(sol_s.get_max_root(), 400.0, 0.0, 1e-12);
}

// ─── no snapping ─────────────────────────────────────────────────────────────

TEST_CASE(test_set_max_root_stores_value_unchanged_no_snap)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    for (double v : {3.5, 5.5, 9.0, 1.0, 123.456})
    {
        sol.set_max_root(v);
        REQUIRE_APPROX(sol.get_max_root(), v, 0.0, 1e-12);
    }
}

// ─── argument validation ─────────────────────────────────────────────────────

TEST_CASE(test_set_max_root_rejects_nonpositive_and_nonfinite)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);

    auto expect_throw = [&](double v) {
        bool threw = false;
        try { sol.set_max_root(v); }
        catch (const std::invalid_argument &) { threw = true; }
        REQUIRE(threw);
    };

    expect_throw(0.0);
    expect_throw(-5.0);
    expect_throw(std::numeric_limits<double>::infinity());
    expect_throw(-std::numeric_limits<double>::infinity());
    expect_throw(std::numeric_limits<double>::quiet_NaN());

    // Rejected calls must not perturb the cutoff.
    REQUIRE_APPROX(sol.get_max_root(), 400.0, 0.0, 1e-12);
}

// ─── retained mode set ────────────────────────────────────────────────────────

TEST_CASE(test_set_max_root_selects_modes_by_bare_root_cutoff_graetz)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(7.0);
    sol.setup_bare_solution();

    auto roots = sol.get_roots();
    REQUIRE(roots.size() == 1u);
    for (double r : roots[0])
        REQUIRE(r <= 7.0 + 1e-12);

    // Known Belhocine & Omar (2017) eigenvalues: 2.7044, 6.6790 <= 7.0 < 10.6733.
    REQUIRE(roots[0].size() == 2u);
}

TEST_CASE(test_set_max_root_stratified_mode_count_matches_catalog)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    const double cutoff = 15.0;
    sol.set_max_root(cutoff);
    sol.setup_bare_solution();

    auto roots = sol.get_roots();
    for (const auto &row : roots)
        for (double r : row)
            REQUIRE(r <= cutoff + 1e-9);

    auto catalog = sol.get_bare_root_catalog();
    unsigned expected = 0;
    for (const auto &row : catalog)
        for (double r : row)
            if (r <= cutoff)
                ++expected;

    REQUIRE(sol.get_number_of_coefficients() == expected);
}

// A root exactly equal to the cutoff must be retained (inclusive comparison).
TEST_CASE(test_set_max_root_boundary_root_is_included)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    auto catalog = sol.get_bare_root_catalog();
    REQUIRE(catalog.size() >= 1u);
    REQUIRE(catalog[0].size() >= 2u);
    const double boundary_root = catalog[0][1];

    sol.set_max_root(boundary_root);
    sol.setup_bare_solution();

    auto roots = sol.get_roots();
    REQUIRE(roots[0].size() == 2u);
    REQUIRE_APPROX(roots[0][1], boundary_root, 0.0, 1e-12);
}

// ─── invalidation semantics ──────────────────────────────────────────────────

TEST_CASE(test_set_max_root_same_value_does_not_invalidate)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    sol.set_max_root(30.0);
    sol.setup_bare_solution();
    REQUIRE(sol.get_solution_method() == SolutionMethod::Bare);

    sol.set_max_root(30.0); // bitwise-identical value: no invalidation
    REQUIRE(sol.get_solution_method() == SolutionMethod::Bare);
}

TEST_CASE(test_set_max_root_changed_value_invalidates)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    sol.set_max_root(30.0);
    sol.setup_bare_solution();
    REQUIRE(sol.get_solution_method() == SolutionMethod::Bare);

    sol.set_max_root(35.0);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    bool threw = false;
    try { sol.get_roots(); }
    catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
}

TEST_CASE(test_set_max_root_resetup_restores_readiness_with_new_mode_count)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    sol.set_max_root(15.0);
    sol.setup_bare_solution();
    const unsigned count_small = sol.get_number_of_coefficients();

    sol.set_max_root(60.0);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    sol.setup_bare_solution();
    REQUIRE(sol.get_solution_method() == SolutionMethod::Bare);
    const unsigned count_large = sol.get_number_of_coefficients();

    REQUIRE(count_large > count_small);
}
