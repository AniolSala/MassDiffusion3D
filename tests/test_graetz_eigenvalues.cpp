#include "tinytest.h"

#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"

#include <cmath>
#include <vector>

// Reference eigenvalues from Belhocine & Omar (2017), Table 1.
// β_m are the roots of psinm_r(0, β_m, 1) = 0  (Dirichlet BC at r=1).
TEST_CASE(test_graetz_eigenvalues_match_belhocine)
{
    // Belhocine Table 1 values (4 decimal places)
    const std::vector<double> expected = {2.7044, 6.6790, 10.6733, 14.6710, 18.6698};

    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution(); // small x → loads many modes

    auto roots = sol.get_roots(); // roots[0][m] for Graetz (only n=0)
    REQUIRE(roots.size() >= 1);
    REQUIRE(roots[0].size() >= expected.size());

    for (std::size_t m = 0; m < expected.size(); ++m)
    {
        // Tolerance: match Belhocine's 4-digit table to within 1e-3
        REQUIRE_APPROX(roots[0][m], expected[m], 1e-3, 1e-3);
    }
}
