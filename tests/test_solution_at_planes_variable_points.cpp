#include "tinytest.h"

#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <sstream>
#include <vector>

TEST_CASE(test_solution_at_planes_variable_points_matches_points) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    const std::vector<double> x_points = {0.01, 0.02, 0.03};

    const std::vector<std::vector<double>> r_points = {
        {0.0, 0.2, 0.5, 0.7, 1.0},
        {0.1, 0.3, 0.6, 0.9},
        {0.0, 0.25, 0.75, 1.0}
    };

    const std::vector<std::vector<double>> phi_points = {
        {0.0, M_PI / 6.0, M_PI / 3.0, M_PI / 2.0, M_PI},
        {M_PI / 8.0, M_PI / 4.0, 3.0 * M_PI / 4.0, M_PI},
        {0.0, M_PI / 5.0, 2.0 * M_PI / 3.0, M_PI}
    };

    const std::vector<std::vector<double>> planes = sol.get_solution_at_planes(x_points, r_points, phi_points);

    REQUIRE(planes.size() == x_points.size());

    const double rel_tol = 1e-9;
    const double abs_tol = 1e-12;

    for (size_t p = 0; p < x_points.size(); p++) {
        REQUIRE(planes[p].size() == r_points[p].size());

        const std::vector<double> expected = sol.get_solution_at_points(x_points[p], r_points[p], phi_points[p]);
        for (size_t i = 0; i < expected.size(); i++) {
            if (!tinytest::approx_equal(planes[p][i], expected[i], rel_tol, abs_tol)) {
                std::ostringstream os;
                os << "Mismatch at plane " << p
                   << " (x=" << x_points[p] << ") index=" << i
                   << " r=" << r_points[p][i] << " phi=" << phi_points[p][i]
                   << " got=" << planes[p][i] << " expected=" << expected[i]
                   << " rel_tol=" << rel_tol << " abs_tol=" << abs_tol;
                throw tinytest::Failure(os.str());
            }
        }
    }
}
