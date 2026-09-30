#include "tinytest.h"

#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <sstream>
#include <vector>

// cutoff_modes(tol) skips, at evaluation time only, the modes whose amplitude
// |C_k exp(-Lam_k x)| * max_u |R_k(r_u)| falls below tol times the largest
// amplitude of the plane. These tests pin the contract: the cut must not move
// the solution, must leave the mode set and the coefficients alone, and must be
// exactly a no-op at tol = 0.

namespace {

// A transverse sample with repeated radii, so the radial table is built and the
// cut is active (it needs max_u |R_k|, which only the table provides).
void sample_points(std::vector<double> &r_points, std::vector<double> &phi_points) {
    r_points.clear();
    phi_points.clear();
    const std::vector<double> radii = {0.0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0};
    const std::vector<double> angles = {0.0, M_PI / 4.0, M_PI / 2.0, 3.0 * M_PI / 4.0, M_PI};
    for (double r : radii)
        for (double phi : angles) {
            r_points.push_back(r);
            phi_points.push_back(phi);
        }
}

void compare_planes(const std::vector<std::vector<double>> &got,
                    const std::vector<std::vector<double>> &expected,
                    const std::vector<double> &x_points,
                    const std::vector<double> &r_points,
                    const std::vector<double> &phi_points,
                    double rel_tol, double abs_tol, const char *what) {
    REQUIRE(got.size() == expected.size());
    for (size_t p = 0; p < expected.size(); p++) {
        REQUIRE(got[p].size() == expected[p].size());
        for (size_t i = 0; i < expected[p].size(); i++) {
            if (!tinytest::approx_equal(got[p][i], expected[p][i], rel_tol, abs_tol)) {
                std::ostringstream os;
                os << what << ": mismatch at x=" << x_points[p]
                   << " r=" << r_points[i] << " phi=" << phi_points[i]
                   << " with_cutoff=" << got[p][i] << " without=" << expected[p][i]
                   << " diff=" << std::fabs(got[p][i] - expected[p][i]);
                throw tinytest::Failure(os.str());
            }
        }
    }
}

} // namespace

// The requested check: at max_root = 100 and tol = 1e-15 the cut must reproduce
// the untruncated solution over a range of x, x = 1 included.
TEST_CASE(test_cutoff_matches_full_sum_bare) {
    const std::vector<double> zi = {-0.55, 0.45};
    const std::vector<double> ui = {1.0, 0.3, 0.0};

    std::vector<double> r_points, phi_points;
    sample_points(r_points, phi_points);
    const std::vector<double> x_points = {0.0, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(100.0);
    sol.setup_bare_solution();

    REQUIRE(sol.get_cutoff_modes() == 0.0);
    const std::vector<std::vector<double>> without = sol.get_solution_at_planes(x_points, r_points, phi_points);

    sol.cutoff_modes(1e-15);
    REQUIRE(sol.get_cutoff_modes() == 1e-15);
    const std::vector<std::vector<double>> with = sol.get_solution_at_planes(x_points, r_points, phi_points);

    // The dropped modes are below 1e-15 of the plane's largest amplitude, and the
    // solution is O(1), so the two sums agree far inside single-precision noise.
    compare_planes(with, without, x_points, r_points, phi_points, 1e-12, 1e-12,
                   "bare solution with cutoff 1e-15");

    // The cut must not touch the truncation or the coefficients.
    REQUIRE(sol.get_max_root() == 100.0);
    REQUIRE(sol.get_max_K() > 0);
}

// x = 1 on its own: the regime the cut is built for, where all but a handful of
// modes have decayed and the saving is largest.
TEST_CASE(test_cutoff_matches_full_sum_at_x_one) {
    const std::vector<double> zi = {-0.55, 0.45};
    const std::vector<double> ui = {1.0, 0.3, 0.0};

    std::vector<double> r_points, phi_points;
    sample_points(r_points, phi_points);

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(100.0);
    sol.setup_bare_solution();

    const std::vector<double> without = sol.get_solution_at_points(1.0, r_points, phi_points);
    sol.cutoff_modes(1e-15);
    const std::vector<double> with = sol.get_solution_at_points(1.0, r_points, phi_points);

    REQUIRE(with.size() == without.size());
    for (size_t i = 0; i < without.size(); i++) {
        if (!tinytest::approx_equal(with[i], without[i], 1e-12, 1e-12)) {
            std::ostringstream os;
            os << "x = 1 fixed-plane mismatch at r=" << r_points[i] << " phi=" << phi_points[i]
               << " with_cutoff=" << with[i] << " without=" << without[i];
            throw tinytest::Failure(os.str());
        }
    }
}

// Same contract on the finite-Peclet path, where the rate used by the cut is the
// modified Lam and the coefficients span ~150 decades.
TEST_CASE(test_cutoff_matches_full_sum_finite_peclet) {
    const std::vector<double> zi = {-0.55, 0.45};
    const std::vector<double> ui = {1.0, 0.3, 0.0};

    std::vector<double> r_points, phi_points;
    sample_points(r_points, phi_points);
    const std::vector<double> x_points = {0.0, 1e-2, 0.1, 0.5, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(100.0);
    sol.setup_fp_solution(100.0);

    const std::vector<std::vector<double>> without = sol.get_solution_at_planes(x_points, r_points, phi_points);
    sol.cutoff_modes(1e-15);
    const std::vector<std::vector<double>> with = sol.get_solution_at_planes(x_points, r_points, phi_points);

    compare_planes(with, without, x_points, r_points, phi_points, 1e-12, 1e-12,
                   "finite-Peclet solution with cutoff 1e-15");
}

// tol = 0 is the default and must be an exact no-op, not merely a close one.
TEST_CASE(test_cutoff_zero_is_bitwise_identical) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};

    std::vector<double> r_points, phi_points;
    sample_points(r_points, phi_points);
    const std::vector<double> x_points = {0.0, 0.01, 0.2, 1.0};

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(60.0);
    sol.setup_bare_solution();

    const std::vector<std::vector<double>> reference = sol.get_solution_at_planes(x_points, r_points, phi_points);
    sol.cutoff_modes(0.0);
    const std::vector<std::vector<double>> again = sol.get_solution_at_planes(x_points, r_points, phi_points);

    REQUIRE(again.size() == reference.size());
    for (size_t p = 0; p < reference.size(); p++) {
        REQUIRE(again[p].size() == reference[p].size());
        for (size_t i = 0; i < reference[p].size(); i++)
            REQUIRE(again[p][i] == reference[p][i]);
    }
}

// The cross-sectional mean is carried by the zero-eigenvalue mode, which never
// decays: even an aggressive cut must keep it, so the far field stays put.
TEST_CASE(test_cutoff_preserves_far_field) {
    const std::vector<double> zi = {-0.55, 0.45};
    const std::vector<double> ui = {1.0, 0.3, 0.0};

    std::vector<double> r_points, phi_points;
    sample_points(r_points, phi_points);

    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(100.0);
    sol.setup_bare_solution();

    const std::vector<double> without = sol.get_solution_at_points(5.0, r_points, phi_points);
    sol.cutoff_modes(1e-3);  // deliberately coarse
    const std::vector<double> with = sol.get_solution_at_points(5.0, r_points, phi_points);

    REQUIRE(with.size() == without.size());
    for (size_t i = 0; i < without.size(); i++)
        REQUIRE_APPROX(with[i], without[i], 1e-8, 1e-10);
}

// A tolerance outside [0, 1) is a programming error, not a silent clamp.
TEST_CASE(test_cutoff_rejects_invalid_tolerance) {
    const std::vector<double> zi = {-0.3};
    const std::vector<double> ui = {0.0, 1.0};
    CDStratifiedSolution<double> sol(zi, ui, 0);

    bool threw_negative = false;
    try { sol.cutoff_modes(-1e-16); } catch (const std::invalid_argument &) { threw_negative = true; }
    REQUIRE(threw_negative);

    bool threw_one = false;
    try { sol.cutoff_modes(1.0); } catch (const std::invalid_argument &) { threw_one = true; }
    REQUIRE(threw_one);

    REQUIRE(sol.get_cutoff_modes() == 0.0);
}
