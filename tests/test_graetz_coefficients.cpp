#include "tinytest.h"

#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"

#include <cmath>
#include <vector>

// Reference values: |C_m_code| = |C_m_Belhocine * G_m(0)| from Table 1 of
// Belhocine & Omar (2017). Our eigenfunctions satisfy psinm_r(0, beta, 0) = 1,
// while Belhocine's G_m(r) uses a different normalization with G_m(0) != 1,
// so the stored coefficient absorbs the G_m(0) factor:
//   code coefficient / sqrt(2pi) = C_m_Belhocine * G_m(0)
//
// m=0: |0.9774 * 1.5106| = 1.4764
// m=1: |0.3858 * 2.0895| = 0.8061
// m=2: |0.2351 * 2.5045| = 0.5888
// m=3: |0.1674 * 2.8426| = 0.4759
// m=4: |0.1292 * 3.1338| = 0.4049
TEST_CASE(test_graetz_coefficients_magnitude_match_belhocine)
{
    const std::vector<double> expected_abs = {1.4764, 0.8061, 0.5888, 0.4759, 0.4049};

    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution(); // small x -> keeps many modes

    const double sqrt2pi = std::sqrt(2.0 * M_PI);
    auto coefficients = sol.get_coefficients(); // coefficients[n][m] = sqrt(2pi) * C_m_code

    REQUIRE(coefficients.size() >= 1);
    REQUIRE(coefficients[0].size() >= expected_abs.size());

    for (std::size_t m = 0; m < expected_abs.size(); ++m)
    {
        const double cm_abs = std::fabs(coefficients[0][m] / sqrt2pi);
        REQUIRE_APPROX(cm_abs, expected_abs[m], 5e-3, 5e-3);
    }
}

// n>0 modes must be absent (Graetz is axisymmetric).
// get_coefficients() matches the shape of get_roots(), which has exactly 1 row
// for Graetz (only n=0 modes exist), so the matrix size itself proves axisymmetry.
TEST_CASE(test_graetz_coefficients_axisymmetric)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution();

    auto coefficients = sol.get_coefficients();
    REQUIRE(coefficients.size() == 1);
}
