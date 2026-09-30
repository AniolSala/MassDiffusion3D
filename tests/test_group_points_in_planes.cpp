#include "tinytest.h"

#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <sstream>
#include <vector>

TEST_CASE(test_group_points_in_planes_basic) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);

    // Example from the docstring:
    // Input: x=[0.01, 0.02, 0.01], r=[0.1, 0.2, 0.3], phi=[0.5, 1.0, 1.5]
    // Output: x_planes=[[0.01, 0.01], [0.02]], r_planes=[[0.1, 0.3], [0.2]], phi_planes=[[0.5, 1.5], [1.0]]
    const std::vector<double> x_points = {0.01, 0.02, 0.01};
    const std::vector<double> r_points = {0.1, 0.2, 0.3};
    const std::vector<double> phi_points = {0.5, 1.0, 1.5};

    std::vector<double> x_planes;
    std::vector<std::vector<double>> r_planes;
    std::vector<std::vector<double>> phi_planes;
    std::vector<std::pair<size_t, size_t>> ordering_map;

    sol.group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map);

    // Check number of planes
    REQUIRE(x_planes.size() == 2);
    REQUIRE(r_planes.size() == 2);
    REQUIRE(phi_planes.size() == 2);

    // Check ordering_map size
    REQUIRE(ordering_map.size() == 3);

    // Verify ordering_map:
    // Original index 0 (x=0.01, r=0.1, phi=0.5) -> plane 0, position 0
    // Original index 1 (x=0.02, r=0.2, phi=1.0) -> plane 1, position 0
    // Original index 2 (x=0.01, r=0.3, phi=1.5) -> plane 0, position 1
    if (ordering_map[0] != std::make_pair(size_t(0), size_t(0))) {
        throw tinytest::Failure("ordering_map[0] should be (0, 0)");
    }
    if (ordering_map[1] != std::make_pair(size_t(1), size_t(0))) {
        throw tinytest::Failure("ordering_map[1] should be (1, 0)");
    }
    if (ordering_map[2] != std::make_pair(size_t(0), size_t(1))) {
        throw tinytest::Failure("ordering_map[2] should be (0, 1)");
    }

    // Check first plane (x=0.01)
    REQUIRE(r_planes[0].size() == 2);
    REQUIRE(phi_planes[0].size() == 2);

    if (x_planes[0] != 0.01) {
        throw tinytest::Failure("First plane x value should be 0.01");
    }

    if (r_planes[0][0] != 0.1 || r_planes[0][1] != 0.3) {
        throw tinytest::Failure("First plane r values should be [0.1, 0.3]");
    }

    if (phi_planes[0][0] != 0.5 || phi_planes[0][1] != 1.5) {
        throw tinytest::Failure("First plane phi values should be [0.5, 1.5]");
    }

    // Check second plane (x=0.02)
    REQUIRE(r_planes[1].size() == 1);
    REQUIRE(phi_planes[1].size() == 1);

    if (x_planes[1] != 0.02) {
        throw tinytest::Failure("Second plane x value should be 0.02");
    }

    if (r_planes[1][0] != 0.2) {
        throw tinytest::Failure("Second plane r value should be 0.2");
    }

    if (phi_planes[1][0] != 1.0) {
        throw tinytest::Failure("Second plane phi value should be 1.0");
    }
}

TEST_CASE(test_group_points_in_planes_empty) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);

    const std::vector<double> x_points;
    const std::vector<double> r_points;
    const std::vector<double> phi_points;

    std::vector<double> x_planes;
    std::vector<std::vector<double>> r_planes;
    std::vector<std::vector<double>> phi_planes;
    std::vector<std::pair<size_t, size_t>> ordering_map;

    sol.group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map);

    // Check that output is empty
    REQUIRE(x_planes.size() == 0);
    REQUIRE(r_planes.size() == 0);
    REQUIRE(phi_planes.size() == 0);
    REQUIRE(ordering_map.size() == 0);
}

TEST_CASE(test_group_points_in_planes_single_plane) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);

    // All points have the same x value
    const std::vector<double> x_points = {0.01, 0.01, 0.01};
    const std::vector<double> r_points = {0.1, 0.2, 0.3};
    const std::vector<double> phi_points = {0.5, 1.0, 1.5};

    std::vector<double> x_planes;
    std::vector<std::vector<double>> r_planes;
    std::vector<std::vector<double>> phi_planes;
    std::vector<std::pair<size_t, size_t>> ordering_map;

    sol.group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map);

    // Check that we have only one plane
    REQUIRE(x_planes.size() == 1);
    REQUIRE(r_planes.size() == 1);
    REQUIRE(phi_planes.size() == 1);
    REQUIRE(ordering_map.size() == 3);

    // Check plane contents
    REQUIRE(r_planes[0].size() == 3);
    REQUIRE(phi_planes[0].size() == 3);

    if (x_planes[0] != 0.01) {
        throw tinytest::Failure("The x value in single plane should be 0.01");
    }

    std::vector<double> expected_r = {0.1, 0.2, 0.3};
    std::vector<double> expected_phi = {0.5, 1.0, 1.5};

    for (size_t i = 0; i < 3; i++) {
        if (r_planes[0][i] != expected_r[i] || phi_planes[0][i] != expected_phi[i]) {
            throw tinytest::Failure("Single plane r and phi values do not match");
        }
        // All points should map to plane 0 with position i
        if (ordering_map[i] != std::make_pair(size_t(0), i)) {
            throw tinytest::Failure("ordering_map[" + std::to_string(i) + "] should be (0, " + std::to_string(i) + ")");
        }
    }
}

TEST_CASE(test_group_points_in_planes_sorting) {
    // Points within a plane must come out sorted by (r, phi) regardless of input order.
    // Here all points share the same x, but are given in descending r order.
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};
    CDStratifiedSolution<double> sol(zi, ui, 0);

    const std::vector<double> x_points   = {0.01, 0.01, 0.01};
    const std::vector<double> r_points   = {0.9,  0.5,  0.1};
    const std::vector<double> phi_points = {1.0,  0.5,  0.0};

    std::vector<double> x_planes;
    std::vector<std::vector<double>> r_planes, phi_planes;
    std::vector<std::pair<size_t, size_t>> ordering_map;

    sol.group_points_in_planes(x_points, r_points, phi_points,
                               x_planes, r_planes, phi_planes, ordering_map);

    REQUIRE(x_planes.size() == 1);
    REQUIRE(r_planes[0].size() == 3);

    // Output must be in ascending r order
    for (size_t i = 0; i + 1 < r_planes[0].size(); i++) {
        if (r_planes[0][i] > r_planes[0][i + 1]) {
            throw tinytest::Failure("r_planes[0] is not sorted in ascending order");
        }
    }

    // ordering_map must be consistent: r_planes[0][map.second] == r_points[original]
    for (size_t orig = 0; orig < x_points.size(); orig++) {
        const auto [plane_idx, pos] = ordering_map[orig];
        if (plane_idx != 0) {
            throw tinytest::Failure("All points should map to plane 0");
        }
        if (r_planes[0][pos] != r_points[orig]) {
            throw tinytest::Failure("ordering_map does not correctly reflect sorted position for r");
        }
        if (phi_planes[0][pos] != phi_points[orig]) {
            throw tinytest::Failure("ordering_map does not correctly reflect sorted position for phi");
        }
    }
}

TEST_CASE(test_group_points_in_planes_with_tolerance) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);

    // Points with x values close to each other: 0.01, 0.01001, 0.02
    // With default tolerance (1e-8), they should be in separate planes
    // With tolerance 1e-3, 0.01 and 0.01001 should be grouped together
    const std::vector<double> x_points = {0.01, 0.01001, 0.02};
    const std::vector<double> r_points = {0.1, 0.2, 0.3};
    const std::vector<double> phi_points = {0.5, 1.0, 1.5};

    // Test with default tolerance (1e-8)
    {
        std::vector<double> x_planes;
        std::vector<std::vector<double>> r_planes;
        std::vector<std::vector<double>> phi_planes;
        std::vector<std::pair<size_t, size_t>> ordering_map;

        sol.group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map);

        // Should have 3 planes since differences are larger than 1e-8
        REQUIRE(x_planes.size() == 3);
        REQUIRE(r_planes.size() == 3);
        REQUIRE(phi_planes.size() == 3);
        REQUIRE(ordering_map.size() == 3);

        // Each point should map to its own plane
        for (size_t i = 0; i < 3; i++) {
            if (ordering_map[i].first != i || ordering_map[i].second != 0) {
                throw tinytest::Failure("With default tolerance, each point should be in its own plane");
            }
        }
    }

    // Test with larger tolerance (1e-3)
    {
        std::vector<double> x_planes;
        std::vector<std::vector<double>> r_planes;
        std::vector<std::vector<double>> phi_planes;
        std::vector<std::pair<size_t, size_t>> ordering_map;

        sol.group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map, 1e-3);

        // Should have 2 planes: 0.01 and 0.01001 grouped together, 0.02 separate
        REQUIRE(x_planes.size() == 2);
        REQUIRE(r_planes.size() == 2);
        REQUIRE(phi_planes.size() == 2);
        REQUIRE(ordering_map.size() == 3);

        // Check first plane has 2 points (0.01 and 0.01001)
        REQUIRE(r_planes[0].size() == 2);
        REQUIRE(phi_planes[0].size() == 2);

        // Check second plane has 1 point (0.02)
        REQUIRE(r_planes[1].size() == 1);
        REQUIRE(phi_planes[1].size() == 1);

        if (x_planes[1] != 0.02) {
            throw tinytest::Failure("Second plane x value should be 0.02");
        }

        // Verify ordering map for grouped points
        // Point 0 (x=0.01) should be at (plane 0, position 0)
        // Point 1 (x=0.01001) should be at (plane 0, position 1)
        // Point 2 (x=0.02) should be at (plane 1, position 0)
        if (ordering_map[0] != std::make_pair(size_t(0), size_t(0))) {
            throw tinytest::Failure("ordering_map[0] should be (0, 0)");
        }
        if (ordering_map[1] != std::make_pair(size_t(0), size_t(1))) {
            throw tinytest::Failure("ordering_map[1] should be (0, 1)");
        }
        if (ordering_map[2] != std::make_pair(size_t(1), size_t(0))) {
            throw tinytest::Failure("ordering_map[2] should be (1, 0)");
        }
    }
}

