#include "tinytest.h"

#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <sstream>
#include <vector>

TEST_CASE(test_solution_at_planes_matches_points) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    const std::vector<double> x_points = {0.01, 0.02, 0.03};
    const std::vector<double> r_points = {0.0, 0.2, 0.5, 0.7, 1.0};
    const std::vector<double> phi_points = {0.0, M_PI / 6.0, M_PI / 3.0, M_PI / 2.0, M_PI};

    const std::vector<double> sol_x0 = sol.get_solution_at_points(x_points[0], r_points, phi_points);
    const std::vector<double> sol_x1 = sol.get_solution_at_points(x_points[1], r_points, phi_points);
    const std::vector<double> sol_x2 = sol.get_solution_at_points(x_points[2], r_points, phi_points);

    const std::vector<std::vector<double>> planes = sol.get_solution_at_planes(x_points, r_points, phi_points);

    REQUIRE(planes.size() == x_points.size());
    REQUIRE(planes[0].size() == r_points.size());
    REQUIRE(planes[1].size() == r_points.size());
    REQUIRE(planes[2].size() == r_points.size());

    const double rel_tol = 1e-9;
    const double abs_tol = 1e-12;

    for (size_t i = 0; i < r_points.size(); i++) {
        const double r_val = r_points[i];
        const double phi_val = phi_points[i];

        auto check_plane = [&](size_t plane_idx, double x_val, double got, double expected) {
            if (!tinytest::approx_equal(got, expected, rel_tol, abs_tol)) {
                std::ostringstream os;
                os << "Mismatch at plane " << plane_idx
                   << " (x=" << x_val << ") index=" << i
                   << " r=" << r_val << " phi=" << phi_val
                   << " got=" << got << " expected=" << expected
                   << " rel_tol=" << rel_tol << " abs_tol=" << abs_tol;
                throw tinytest::Failure(os.str());
            }
        };

        check_plane(0, x_points[0], planes[0][i], sol_x0[i]);
        check_plane(1, x_points[1], planes[1][i], sol_x1[i]);
        check_plane(2, x_points[2], planes[2][i], sol_x2[i]);
    }
}
