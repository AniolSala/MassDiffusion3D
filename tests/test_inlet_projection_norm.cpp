// Tests for CDBaseSolution::get_inlet_projection_square_norm, the quantity
// sum_{n,m,k} C_nm W^n_mk C_nk of eq. (gram_matrix_weighted) reported in the
// user's own scaling.
//
// Two things are under test and they are independent. First the contraction
// identity: the solver stores chat^T b and never assembles W, so the reference
// here assembles W per angular block (the Gauss-Jacobi factor plus its
// reconstruction) and contracts the quadratic form directly. Second the
// unscaling: the solver projects the profile set_ui normalised to [0,1], and the
// getter must undo that, which the reference does through the plain three-term
// expansion rather than the completed-square form the implementation evaluates.

#include "tinytest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#undef protected
#undef private

#include "finite_peclet_gram_gauss_jacobi.h"
#include "projection_space.h"

namespace
{
constexpr double kHalfPi = 1.5707963267948966;   // int_D (1 - r^2) dA

// chat^T W chat, with every W^n assembled explicitly. Independent of the
// solver's own chat^T b shortcut.
template <typename Solver>
double gram_quadratic_form(const Solver &solver)
{
    const auto &data = solver.m_series_data;
    unsigned highest_n = 0;
    for (unsigned k = 0; k < solver.m_max_K; ++k)
        highest_n = std::max(highest_n, data[k].n);

    double total = 0.0;
    for (unsigned n = 0; n <= highest_n; ++n)
    {
        std::vector<SeriesTermData<double>> modes;
        std::vector<double> coefficients;
        for (unsigned k = 0; k < solver.m_max_K; ++k)
            if (data[k].n == n)
            {
                modes.push_back(data[k]);
                coefficients.push_back(data[k].coeff_fp);
            }
        if (modes.empty())
            continue;
        const auto factor = fp_gram_factor_gauss_jacobi(n, modes, FPGramGaussJacobiOptions<double>{});
        std::vector<std::vector<double>> gram;
        fp_gram_reconstruct_gauss_jacobi(factor, gram);
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            for (std::size_t j = 0; j < coefficients.size(); ++j)
                total += coefficients[i] * gram[i][j] * coefficients[j];
    }
    return total;
}

// The three-term expansion of the unscaling, valid because the constant is a
// retained mode of the Neumann basis: || f_K ||^2 = a^2 ||1||^2 + 2ab <1,f~> + b^2 Q.
template <typename Solver>
double unscale(const Solver &solver, double scaled_square_norm)
{
    const double a = solver.m_max_ui, b = -(solver.m_max_ui - solver.m_min_ui);
    return a * a * kHalfPi + 2.0 * a * b * kHalfPi * solver.m_constant_term
           + b * b * scaled_square_norm;
}

double radial_cap_area(double z)
{
    return std::acos(z) - z * std::sqrt(std::max(0.0, 1.0 - z * z));
}

template <typename Solver>
double unscale_radial_stratified(const Solver &solver, double scaled_square_norm)
{
    double scaled_inlet_integral = 0.0;
    double lower = -1.0;
    for (std::size_t i = 0; i < solver.m_ui.size(); ++i)
    {
        const double upper = i < solver.m_zi.size() ? solver.m_zi[i] : 1.0;
        scaled_inlet_integral += solver.m_ui[i] * (radial_cap_area(lower) - radial_cap_area(upper));
        lower = upper;
    }
    const double alpha = solver.m_max_ui, beta = -(solver.m_max_ui - solver.m_min_ui);
    return alpha * alpha * std::acos(-1.0)
           + 2.0 * alpha * beta * scaled_inlet_integral
           + beta * beta * scaled_square_norm;
}

// || f ||^2_omega of a piecewise-constant inlet, from the weighted cap areas.
template <typename Solver>
double exact_inlet_square_norm(Solver &solver, const std::vector<double> &layer_values)
{
    std::vector<double> edges;
    edges.push_back(-1.0);
    for (double z : solver.m_zi)
        edges.push_back(z);
    edges.push_back(1.0);
    double total = 0.0;
    for (std::size_t i = 0; i < layer_values.size(); ++i)
        total += layer_values[i] * layer_values[i]
               * (solver.compute_partial_flux(edges[i]) - solver.compute_partial_flux(edges[i + 1]));
    return total;
}

const std::vector<double> kInterfaces{-0.4, 0.2};
const std::vector<double> kLayers{1.0, 0.3, -0.2};
} // namespace

TEST_CASE(inlet_projection_norm_matches_assembled_gram_quadratic_form)
{
    CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
    solver.setup_solution(5.0, 40.0);
    // The stored scalar is the contraction; the reference is the assembled form.
    REQUIRE_APPROX(solver.m_inlet_projection_square_norm, gram_quadratic_form(solver), 1e-10, 0.0);
    // And the getter is that scalar carried back to the user's scaling.
    REQUIRE_APPROX(solver.get_inlet_projection_square_norm(),
                   unscale(solver, gram_quadratic_form(solver)), 1e-10, 0.0);
}

TEST_CASE(inlet_projection_norm_approaches_exact_norm_from_below)
{
    double previous_gap = 1.0;
    for (double max_root : {20.0, 40.0, 80.0, 160.0})
    {
        CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
        solver.setup_solution(5.0, max_root);
        const double exact = exact_inlet_square_norm(solver, kLayers);
        const double gap = exact - solver.get_inlet_projection_square_norm();
        // Galerkin: the projection can never exceed the datum it approximates.
        REQUIRE(gap > 0.0);
        REQUIRE(gap < previous_gap);
        previous_gap = gap;
    }
    // Halving the gap per doubling of the cutoff; a loose ceiling on the last one.
    REQUIRE(previous_gap < 3e-3);
}

TEST_CASE(inlet_projection_norm_agrees_between_bare_and_modified_bases)
{
    CDStratifiedSolution<double> modified(kInterfaces, kLayers, 0);
    modified.setup_solution(5.0, 80.0);
    CDStratifiedSolution<double> bare(kInterfaces, kLayers, 0);
    bare.set_max_root(80.0);
    bare.setup_bare_solution();

    // Two different bases, each the best approximation from its own span in the
    // same norm, so the two values agree to the size of the truncation error
    // rather than exactly.
    REQUIRE_APPROX(modified.get_inlet_projection_square_norm(),
                   bare.get_inlet_projection_square_norm(), 5e-3, 0.0);

    // On the bare path the Gram matrix is diagonal, so the stored scalar must be
    // exactly sum_k N_k^2 c_k^2 -- an independent expression of the same sum.
    double diagonal = 0.0;
    for (unsigned k = 0; k < bare.m_max_K; ++k)
        diagonal += bare.m_series_data[k].norm * bare.m_series_data[k].coeff * bare.m_series_data[k].coeff;
    REQUIRE_APPROX(bare.m_inlet_projection_square_norm, diagonal, 1e-14, 0.0);
}

TEST_CASE(inlet_projection_norm_graetz_scales_with_inlet_temperature)
{
    const double T0 = 2.0, T_wall = 0.5;
    CDGraetzIsothermalSolution<double> solver(T0, T_wall, 0);
    solver.setup_solution(5.0, 60.0);
    // Graetz's internal datum is identically one, so the whole unscaling is the
    // single factor T0^2 -- the user's inlet is the constant T0.
    REQUIRE_APPROX(solver.get_inlet_projection_square_norm(),
                   T0 * T0 * solver.m_inlet_projection_square_norm, 1e-13, 0.0);
    // || T0 ||^2_omega = T0^2 pi/2, approached from below by the Dirichlet basis.
    const double exact = T0 * T0 * kHalfPi;
    REQUIRE(solver.get_inlet_projection_square_norm() < exact);
    REQUIRE(solver.get_inlet_projection_square_norm() > 0.99 * exact);
}

TEST_CASE(inlet_projection_norm_requires_active_solution_and_supports_l2r)
{
    CDStratifiedSolution<double> fresh(kInterfaces, kLayers, 0);
    bool threw = false;
    try { fresh.get_inlet_projection_square_norm(); } catch (const std::exception &) { threw = true; }
    REQUIRE(threw);

    // Invalidation must drop the cached value, not leave it readable.
    fresh.setup_solution(5.0, 20.0);
    REQUIRE(fresh.get_inlet_projection_square_norm() > 0.0);
    fresh.set_max_root(40.0);
    threw = false;
    try { fresh.get_inlet_projection_square_norm(); } catch (const std::exception &) { threw = true; }
    REQUIRE(threw);

    // The radial contraction is returned in L2_r and correctly unscaled.
    CDStratifiedSolution<double> radial(kInterfaces, kLayers, 0);
    radial.set_projection_space(ProjectionSpace::Radial);
    radial.setup_solution(5.0, 20.0);
    REQUIRE(radial.m_inlet_projection_square_norm > 0.0);
    REQUIRE_APPROX(radial.get_inlet_projection_square_norm(),
                   unscale_radial_stratified(radial, radial.m_inlet_projection_square_norm), 1e-13, 0.0);

    // Graetz's uniform internal inlet has P_r(1) = P_r(f~), so its user-scaled
    // projection norm is T0^2 times the stored radial contraction.
    CDGraetzIsothermalSolution<double> graetz(2.0, 0.5, 0);
    graetz.set_projection_space(ProjectionSpace::Radial);
    graetz.setup_solution(5.0, 20.0);
    REQUIRE_APPROX(graetz.get_inlet_projection_square_norm(),
                   4.0 * graetz.m_inlet_projection_square_norm, 1e-13, 0.0);
}
