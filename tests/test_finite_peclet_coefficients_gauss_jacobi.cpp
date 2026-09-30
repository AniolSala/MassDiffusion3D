#include "tinytest.h"
#include "finite_peclet_coefficients_gauss_jacobi.h"
#include "finite_peclet_radial.h"

#include <cmath>
#include <stdexcept>

TEST_CASE(fp_gauss_jacobi_one_point_rule)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(1);
    CHECK(rule.nodes.size() == 1);
    CHECK(std::abs(rule.nodes[0] - 0.6) < 1e-14);
    CHECK(std::abs(rule.weights[0] - 2.0 / 3.0) < 1e-14);
}

TEST_CASE(fp_gauss_jacobi_rule_moments)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(8);
    for (unsigned p = 0; p < 16; ++p) {
        double sum = 0;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i)
            sum += rule.weights[i] * std::pow(rule.nodes[i], p);
        CHECK(std::abs(sum - 2.0 / (2.0 * p + 3.0)) < 2e-12);
    }
}

TEST_CASE(fp_radial_factor_separates_r_power)
{
    const unsigned n = 3;
    const double r = 0.37, b = 4.1, bt = 5.0;
    CHECK(std::abs(psinm_r_fp(n, b, bt, r) - r*r*r*fp_radial_factor(n,b,bt,r*r)) < 1e-14);
    CHECK(std::abs(fp_radial_factor(n,b,bt,0.0) - 1.0) < 1e-14);
}

TEST_CASE(fp_gauss_jacobi_rejects_zero_order)
{
    bool thrown = false;
    try { static_cast<void>(make_fp_gauss_jacobi_rule<double>(0)); }
    catch (const std::invalid_argument &) { thrown = true; }
    CHECK(thrown);
}

TEST_CASE(fp_gauss_jacobi_cap_special_cases)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(32);
    const std::vector<double> radial_weights{0.25};
    const std::vector<double> radial_nodes{0.5};
    SeriesTermData<double> mode{};
    mode.root_fp = 0.0;
    mode.btilde_fp = 0.0;
    mode.n = 1;
    CHECK(compute_layer_nm_quadrature(mode, 1.0, rule, radial_weights, radial_nodes) == 0.0);
    CHECK_APPROX(compute_layer_nm_quadrature(mode, 0.0, rule, radial_weights, radial_nodes),
                 4.0 / (15.0 * std::sqrt(std::acos(-1.0))), 2e-13, 2e-14);
    mode.n = 2;
    CHECK(compute_layer_nm_quadrature(mode, 0.0, rule, radial_weights, radial_nodes) == 0.0);
}

TEST_CASE(fp_gauss_jacobi_constant_profile_identity)
{
    SeriesTermData<double> constant{};
    constant.n = 0; constant.m = 0;
    constant.root_fp = 0.0; constant.btilde_fp = 0.0; constant.rate_fp = 0.0;
    std::vector<SeriesTermData<double>> modes{constant};
    // cap_quad_margin=8 -> this mode's own order = ceil(0)+8=8, bucketed to
    // 20 (fp_rhs_required_order); the exact identity below is independent of
    // quadrature order regardless.
    fp_stratified_coefficients_gauss_jacobi({}, {2.0}, 1, modes,
                                             8,
                                             GramMethod::GaussJacobiQR, WallCondition::Neumann, FPGramGaussJacobiOptions<double>{},
                                             RhsMethod::DirectQuadrature, 40u);
    CHECK_APPROX(modes[0].coeff_fp, 2.0 * std::sqrt(2.0 * std::acos(-1.0)), 2e-13, 2e-13);
}
