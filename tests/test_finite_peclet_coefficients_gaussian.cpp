#include "tinytest.h"
#include "finite_peclet_coefficients_gaussian.h"

#include <limits>
#include <vector>

TEST_CASE(fp_gaussian_graetz_zero_mode) {
    std::vector<SeriesTermData<double>> modes(1);
    modes[0] = {0, 0, 0.0, 0.25, 0.0, 0.0};
    modes[0].root_fp = modes[0].btilde_fp = modes[0].rate_fp = 0.0;
    // The zero mode's full-disk projection is exact regardless of quadrature
    // order: sqrt(2pi) * int_0^1 (1-r^2) r dr = sqrt(2pi)/4. cap_quad_margin=8u
    // here means this mode's own order (fp_rhs_required_order, bare root=0)
    // is ceil(0)+8=8, bucketed to 20.
    fp_graetz_coefficients_gaussian(1, modes, GramMethod::GaussJacobiQR, WallCondition::Dirichlet, FPGramGaussJacobiOptions<double>{}, 8u, RhsMethod::DirectQuadrature, 40u);
    REQUIRE_APPROX(modes[0].coeff_fp, 2.5066282746310005, 1e-13, 1e-13);
}

// cap_quad_margin = 0 is the smallest legal margin: fp_rhs_required_order
// clamps the target order to at least 1 before bucketing (make_fp_gauss_jacobi_rule
// rejects only order == 0, which bare_root + margin can never produce once
// clamped -- unlike the old fixed-order API, a direct "reject invalid
// cap_quad_points" test is no longer reachable through this entry point).
// The exact zero-mode identity still holds at this minimal order, since it
// is independent of quadrature order (see above).
TEST_CASE(fp_gaussian_graetz_zero_mode_minimal_cap_quad_margin) {
    std::vector<SeriesTermData<double>> modes(1);
    modes[0] = {0, 0, 0.0, 0.25, 0.0, 0.0};
    modes[0].root_fp = modes[0].btilde_fp = modes[0].rate_fp = 0.0;
    fp_graetz_coefficients_gaussian(1, modes, GramMethod::GaussJacobiQR, WallCondition::Dirichlet, FPGramGaussJacobiOptions<double>{}, 0u, RhsMethod::DirectQuadrature, 40u);
    REQUIRE_APPROX(modes[0].coeff_fp, 2.5066282746310005, 1e-13, 1e-13);
}
