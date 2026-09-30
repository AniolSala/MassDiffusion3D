#include "tinytest.h"

#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <sstream>
#include <vector>

TEST_CASE(test_get_solution_cloud_of_points_basic) {
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // Create a cloud of points with repeated x values
    const std::vector<double> x_points = {0.012234, 0.012234, 0.025542, 0.0321, 0.012234};
    const std::vector<double> r_points = {0.1, 0.3, 0.5, 0.7, 0.9};
    const std::vector<double> phi_points = {0.0, M_PI / 4.0, M_PI / 2.0, 3.0 * M_PI / 4.0, M_PI};

    // Get solution using get_solution_cloud_of_points
    std::vector<double> cloud_result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);

    // Check that result has correct size
    REQUIRE(cloud_result.size() == x_points.size());

    // Compute solution independently for each point
    std::vector<double> individual_results;
    for (size_t i = 0; i < x_points.size(); i++)
    {
        double result;
        sol.get_solution_at_point(x_points[i], r_points[i], phi_points[i], result);
        individual_results.push_back(result);
    }

    // Check that cloud result matches individual computations
    const double rel_tol = 1e-5;
    const double abs_tol = 1e-5;

    for (size_t i = 0; i < x_points.size(); i++)
    {
        if (!tinytest::approx_equal(cloud_result[i], individual_results[i], rel_tol, abs_tol))
        {
            std::ostringstream os;
            os << "Mismatch at point " << i
               << " (x=" << x_points[i] << ", r=" << r_points[i] << ", phi=" << phi_points[i] << ")"
               << " cloud_result=" << cloud_result[i]
               << " individual=" << individual_results[i]
               << " rel_tol=" << rel_tol << " abs_tol=" << abs_tol;
            throw tinytest::Failure(os.str());
        }
    }
}

TEST_CASE(test_get_solution_cloud_of_points_empty) {
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // Empty cloud of points
    const std::vector<double> x_points;
    const std::vector<double> r_points;
    const std::vector<double> phi_points;

    std::vector<double> result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);

    // Should return empty result
    REQUIRE(result.size() == 0);
}

TEST_CASE(test_get_solution_cloud_of_points_single_point) {
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // Single point
    const std::vector<double> x_points = {0.015};
    const std::vector<double> r_points = {0.5};
    const std::vector<double> phi_points = {M_PI / 3.0};

    std::vector<double> cloud_result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);

    // Check that result has one element
    REQUIRE(cloud_result.size() == 1);

    // Compute individually
    double individual_result;
    sol.get_solution_at_point(x_points[0], r_points[0], phi_points[0], individual_result);

    // Compare
    const double rel_tol = 1e-7;
    const double abs_tol = 1e-10;

    if (!tinytest::approx_equal(cloud_result[0], individual_result, rel_tol, abs_tol))
    {
        std::ostringstream os;
        os << "Single point mismatch: cloud=" << cloud_result[0]
           << " individual=" << individual_result;
        throw tinytest::Failure(os.str());
    }
}

TEST_CASE(test_planes_equal_interleaved_input) {
    // Two x-planes share the same (r, phi) grid, but points are interleaved in the input
    // (x alternates 0.02, 0.04, 0.02, 0.04). Before the sorting fix this triggered the
    // slow per-plane path and produced wrong results when planes_are_equal was false.
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // Grid: r=[0.2, 0.6], phi=[0.0, M_PI/3]  (2 r-values x 2 phi-values = 4 points per plane)
    // Interleaved across x=0.02 and x=0.04:
    //   input index 0: x=0.02, r=0.2, phi=0.0
    //   input index 1: x=0.04, r=0.2, phi=0.0
    //   input index 2: x=0.02, r=0.6, phi=M_PI/3
    //   input index 3: x=0.04, r=0.6, phi=M_PI/3
    const std::vector<double> x_points   = {0.02, 0.04, 0.02, 0.04};
    const std::vector<double> r_points   = {0.2, 0.2, 0.6, 0.6};
    const std::vector<double> phi_points = {0.0, 0.0, M_PI / 3.0, M_PI / 3.0};

    std::vector<double> cloud_result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);
    REQUIRE(cloud_result.size() == x_points.size());

    const double rel_tol = 1e-5, abs_tol = 1e-5;
    for (size_t i = 0; i < x_points.size(); i++)
    {
        double ref;
        sol.get_solution_at_point(x_points[i], r_points[i], phi_points[i], ref);
        if (!tinytest::approx_equal(cloud_result[i], ref, rel_tol, abs_tol))
        {
            std::ostringstream os;
            os << "Mismatch at interleaved point " << i
               << " (x=" << x_points[i] << ", r=" << r_points[i] << ", phi=" << phi_points[i] << ")"
               << " cloud=" << cloud_result[i] << " ref=" << ref;
            throw tinytest::Failure(os.str());
        }
    }
}

TEST_CASE(test_planes_equal_phi_pm_pi) {
    // Two x-planes share the same (r, phi) grid except one plane uses phi=+pi and the
    // other uses phi=-pi for the same physical point. Before the tolerance fix, planes_are_equal
    // was falsely set to false for that point because +pi != -pi exactly.
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // Plane 0 (x=0.02): r=0.5, phi=+pi
    // Plane 1 (x=0.04): r=0.5, phi=-pi  (same point physically)
    const std::vector<double> x_points   = {0.02, 0.04};
    const std::vector<double> r_points   = {0.5,  0.5};
    const std::vector<double> phi_points = {M_PI, -M_PI};

    std::vector<double> cloud_result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);
    REQUIRE(cloud_result.size() == 2);

    const double rel_tol = 1e-5, abs_tol = 1e-5;
    for (size_t i = 0; i < x_points.size(); i++)
    {
        double ref;
        sol.get_solution_at_point(x_points[i], r_points[i], phi_points[i], ref);
        if (!tinytest::approx_equal(cloud_result[i], ref, rel_tol, abs_tol))
        {
            std::ostringstream os;
            os << "Mismatch at phi boundary point " << i
               << " (x=" << x_points[i] << ", phi=" << phi_points[i] << ")"
               << " cloud=" << cloud_result[i] << " ref=" << ref;
            throw tinytest::Failure(os.str());
        }
    }
}

TEST_CASE(test_get_solution_cloud_of_points_all_same_x) {
    const std::vector<double> zi = {-0.23423, 0.33452};
    const std::vector<double> ui = {1., 0.3, 0.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(30.0);  // fixed truncation keeps the suite fast; self-consistency holds for any cutoff
    sol.setup_bare_solution();

    // All points have the same x value
    const std::vector<double> x_points = {0.02, 0.02, 0.02, 0.02, 0.02};
    const std::vector<double> r_points = {0.1, 0.3, 0.5, 0.7, 0.9};
    const std::vector<double> phi_points = {0.0, M_PI / 4.0, M_PI / 2.0, 3.0 * M_PI / 4.0, M_PI};

    std::vector<double> cloud_result = sol.get_solution_cloud_of_points(x_points, r_points, phi_points);

    // Check that result has 5 points
    REQUIRE(cloud_result.size() == 5);

    // Compute independently for each point
    std::vector<double> individual_results;
    for (size_t i = 0; i < x_points.size(); i++)
    {
        double result;
        sol.get_solution_at_point(x_points[i], r_points[i], phi_points[i], result);
        individual_results.push_back(result);
    }

    // Compare
    const double rel_tol = 1e-7;
    const double abs_tol = 1e-10;

    for (size_t i = 0; i < x_points.size(); i++)
    {
        if (!tinytest::approx_equal(cloud_result[i], individual_results[i], rel_tol, abs_tol))
        {
            std::ostringstream os;
            os << "Mismatch at point " << i << " (all same x): "
               << "cloud=" << cloud_result[i]
               << " individual=" << individual_results[i];
            throw tinytest::Failure(os.str());
        }
    }
}
