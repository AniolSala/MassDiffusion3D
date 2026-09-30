// Pre-check (plan §0): verify that m_pseudo_products_matrix[n][m][m] is the
// UNWEIGHTED integral ∫ R_{nm}² r dr, NOT the weighted norm N² = ∫ R² ω r dr.
//
// The eigenvalue-shift formula uses α = P_kk / N², so getting the measure wrong
// (P_kk == N²  →  α == 1) would silently produce a wrong correction.
//
// Strategy: compute ∫ R_{0m}² r dr by independent Gauss-Legendre quadrature and
// assert it equals m_pseudo_products_matrix[0][m][m]. Also assert it differs from
// term.norm (the weighted squared norm).

#include "tinytest.h"

#include <cmath>
#include <limits>
#include <vector>

#define private public
#define protected public
#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#include "../src/CDStratifiedSolution/CDStratifiedSolution.h"

// Independent unweighted integral ∫₀¹ R_{nm}²(r) r dr via Gauss-Legendre.
// Uses the same psinm_r function available through the solver headers.
static double unweighted_norm_sq(unsigned n, double root, int n_pt = 400)
{
    // Composite midpoint rule on [0,1]; 400 points gives ~4 digits, enough for 2% tolerance.
    double h = 1.0 / n_pt;
    double s = 0.0;
    for (int i = 0; i < n_pt; i++)
    {
        double r = (i + 0.5) * h;
        double psi = psinm_r(n, root, r);
        s += psi * psi * r * h;
    }
    return s;
}

// Weighted integral ∫₀¹ R_{nm}²(r) (1-r²) r dr via the same midpoint rule.
static double weighted_norm_sq(unsigned n, double root, int n_pt = 400)
{
    double h = 1.0 / n_pt;
    double s = 0.0;
    for (int i = 0; i < n_pt; i++)
    {
        double r = (i + 0.5) * h;
        double psi = psinm_r(n, root, r);
        double omega = 1.0 - r * r;
        s += psi * psi * omega * r * h;
    }
    return s;
}

TEST_CASE(test_pseudo_product_diagonal_equals_unweighted_graetz)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.compute_and_fix_coefficients();

    // Load pseudo-products and check diagonal for first few modes.
    sol.set_pseudo_products_matrix(sol.m_pseudo_products_matrix);

    for (unsigned k = 0; k < sol.m_max_K && k < 5; k++)
    {
        const auto &td = sol.m_series_data[k];
        const unsigned n = td.n, m = td.m;

        if (n >= sol.m_pseudo_products_matrix.size()) continue;
        if (m >= sol.m_pseudo_products_matrix[n].size()) continue;
        if (m >= sol.m_pseudo_products_matrix[n][m].size()) continue;

        const double P_kk     = sol.m_pseudo_products_matrix[n][m][m];
        const double quad_uw  = unweighted_norm_sq(n, td.root);
        const double quad_w   = weighted_norm_sq(n, td.root);

        // P_kk should match the unweighted integral to ~1% (midpoint rule).
        double rel_err_uw = std::abs(P_kk - quad_uw) / (std::abs(quad_uw) + 1e-300);
        REQUIRE(rel_err_uw < 0.02);

        // P_kk must NOT equal the weighted norm (td.norm) — the ratio must differ
        // from 1 by more than rounding when omega is not constant.
        double ratio = P_kk / td.norm;
        REQUIRE(std::abs(ratio - 1.0) > 0.01);  // alpha != 1

        // The weighted quad should match td.norm.
        double rel_err_w = std::abs(td.norm - quad_w) / (std::abs(quad_w) + 1e-300);
        REQUIRE(rel_err_w < 0.02);
    }
}

TEST_CASE(test_pseudo_product_diagonal_equals_unweighted_stratified)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.0}, 0);
    sol.set_max_root(30.0);  // diagonal check needs only a representative mode set; keeps it fast
    sol.compute_and_fix_coefficients();

    sol.set_pseudo_products_matrix(sol.m_pseudo_products_matrix);

    for (unsigned k = 0; k < sol.m_max_K && k < 5; k++)
    {
        const auto &td = sol.m_series_data[k];
        const unsigned n = td.n, m = td.m;

        if (sol.is_zero_eigenvalue_mode(n, m)) continue;  // skip β=0 mode
        if (n >= sol.m_pseudo_products_matrix.size()) continue;
        if (m >= sol.m_pseudo_products_matrix[n].size()) continue;
        if (m >= sol.m_pseudo_products_matrix[n][m].size()) continue;

        const double P_kk    = sol.m_pseudo_products_matrix[n][m][m];
        const double quad_uw = unweighted_norm_sq(n, td.root);

        double rel_err = std::abs(P_kk - quad_uw) / (std::abs(quad_uw) + 1e-300);
        REQUIRE(rel_err < 0.02);

        // alpha = P_kk / N² must differ from 1 (omega != 1 means measures differ)
        double ratio = P_kk / td.norm;
        REQUIRE(std::abs(ratio - 1.0) > 0.01);
    }
}
