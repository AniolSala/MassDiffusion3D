#include "tinytest.h"

#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"

#include <cmath>
#include <vector>

// Reference center temperatures θ(ζ, r=0) from Belhocine & Omar (2017), Table 2.
// T0=1, T_wall=0 → physical T = θ.
TEST_CASE(test_graetz_center_temperature_match_belhocine)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution(); // small x → keeps all modes

    // Belhocine Table 2 center temperatures at r=0
    const std::vector<double> x_vals     = {0.1,         0.25,        0.5        };
    const std::vector<double> theta_ref  = {0.70123412,  0.23720134,  0.03811139 };

    for (std::size_t i = 0; i < x_vals.size(); ++i)
    {
        std::vector<double> x_in   = {x_vals[i]};
        std::vector<double> r_in   = {0.0};
        std::vector<double> phi_in = {0.0};
        auto result = sol.get_solution(x_in, r_in, phi_in);

        REQUIRE(result.size() == 1);
        // Tolerance 5e-3: Belhocine Table 2 gives 8 significant figures; series
        // convergence and quadrature errors keep residuals well below 1e-3.
        REQUIRE_APPROX(result[0], theta_ref[i], 5e-3, 5e-3);
    }
}

// Dirichlet BC: T(x, r=1) = T_wall = 0 for all x > 0.
TEST_CASE(test_graetz_wall_bc)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution();

    const std::vector<double> x_vals = {0.05, 0.1, 0.25, 0.5, 1.0};
    for (double x : x_vals)
    {
        auto result = sol.get_solution({x}, {1.0}, {0.0});
        REQUIRE(result.size() == 1);
        REQUIRE_APPROX(result[0], 0.0, 1e-10, 1e-10);
    }
}

// Near-inlet condition: θ(x→0, r=0) → 1 (uniform inlet).
TEST_CASE(test_graetz_inlet_condition)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution();

    auto result = sol.get_solution({1e-5}, {0.0}, {0.0});
    REQUIRE(result.size() == 1);
    // With 100 modes and a very small x, the series sums to ≈1 at r=0.
    REQUIRE_APPROX(result[0], 1.0, 1e-2, 1e-2);
}
