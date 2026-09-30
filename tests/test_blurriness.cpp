// Tests for CDBaseSolution::get_blurriness (thesis, chapter "Applications").
//
// The modal formula is checked against a direct quadrature of the definition
//     B(x)^2 = || psi_K(x) - f ||^2_omega / || psi_inf - f ||^2_omega
// with the series evaluated at Gauss-Legendre nodes of a layer-conforming
// (z, y) grid of the disk (so the step inlet is exactly integrable), for the
// bare and the finite-Peclet setups of the stratified inlet and for the
// isothermal Graetz inlet. Then the limits, the monotonicity, the scale
// invariance and the two normalisations (with / without the truncation tail).

#include "tinytest.h"

#include "../src/CDStratifiedSolution/CDStratifiedSolution.h"
#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
constexpr double kHalfPi = 1.5707963267948966;

// Gauss-Legendre nodes/weights on [a, b] (n up to 64 through Newton on Legendre polynomials).
void gauss_legendre(unsigned n, double a, double b, std::vector<double> &x, std::vector<double> &w)
{
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    for (unsigned i = 0; i < n; ++i)
    {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double pp = 0.0;
        for (unsigned it = 0; it < 100; ++it)
        {
            double p1 = 1.0, p2 = 0.0;
            for (unsigned j = 1; j <= n; ++j)
            {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double dz = p1 / pp;
            z -= dz;
            if (std::fabs(dz) < 1e-15)
                break;
        }
        x[i] = 0.5 * (b - a) * z + 0.5 * (b + a);
        w[i] = (b - a) / ((1.0 - z * z) * pp * pp);
    }
}

// Layer-conforming quadrature of the disk: nodes (y, z), weights including dS,
// the inlet value f at every node and omega = 1 - r^2.
struct DiskRule
{
    std::vector<double> r, phi, weight, f, omega;
};

// The z-integrand of every disk integral behaves as (1 - z^2)^{3/2} at the
// walls, whose derivative is singular there and defeats a plain Gauss-Legendre
// rule; the substitution z = -1 + (z_1 + 1) s^2 (and its mirror at z = +1) makes
// the integrand a smooth function of s. A layer touching both walls is split at 0.
void wall_mapped_nodes(double lower, double upper, unsigned n, std::vector<double> &zn, std::vector<double> &zw)
{
    zn.clear();
    zw.clear();
    std::vector<double> sn, sw;
    auto append_plain = [&](double a, double b) {
        gauss_legendre(n, a, b, sn, sw);
        zn.insert(zn.end(), sn.begin(), sn.end());
        zw.insert(zw.end(), sw.begin(), sw.end());
    };
    auto append_from_wall = [&](double wall, double inner) {
        // z = wall + (inner - wall) s^2, dz = 2 (inner - wall) s ds, s in (0, 1)
        gauss_legendre(n, 0.0, 1.0, sn, sw);
        for (unsigned i = 0; i < n; ++i)
        {
            zn.push_back(wall + (inner - wall) * sn[i] * sn[i]);
            zw.push_back(sw[i] * 2.0 * std::fabs(inner - wall) * sn[i]);
        }
    };
    const bool at_lower_wall = lower <= -1.0 + 1e-14;
    const bool at_upper_wall = upper >= 1.0 - 1e-14;
    if (at_lower_wall && at_upper_wall)
    {
        append_from_wall(-1.0, 0.0);
        append_from_wall(1.0, 0.0);
    }
    else if (at_lower_wall)
        append_from_wall(-1.0, upper);
    else if (at_upper_wall)
        append_from_wall(1.0, lower);
    else
        append_plain(lower, upper);
}

DiskRule layered_disk_rule(const std::vector<double> &interfaces, const std::vector<double> &values, unsigned n)
{
    std::vector<double> edges;
    edges.push_back(-1.0);
    for (double z : interfaces)
        edges.push_back(z);
    edges.push_back(1.0);
    DiskRule rule;
    std::vector<double> zn, zw, yn, yw;
    for (std::size_t i = 0; i + 1 < edges.size(); ++i)
    {
        wall_mapped_nodes(edges[i], edges[i + 1], n, zn, zw);
        for (std::size_t a = 0; a < zn.size(); ++a)
        {
            const double half = std::sqrt(std::max(0.0, 1.0 - zn[a] * zn[a]));
            gauss_legendre(n, -half, half, yn, yw);
            for (unsigned b = 0; b < n; ++b)
            {
                const double y = yn[b], z = zn[a];
                rule.r.push_back(std::min(1.0, std::sqrt(y * y + z * z)));
                rule.phi.push_back(std::atan2(y, z));
                rule.weight.push_back(yw[b] * zw[a]);
                rule.f.push_back(values[i]);
                rule.omega.push_back(std::max(0.0, 1.0 - y * y - z * z));
            }
        }
    }
    return rule;
}

template <typename Solver>
double quadrature_blurriness(Solver &solver, const DiskRule &rule, double x, double psi_inf)
{
    const std::vector<double> psi = solver.get_solution_at_points(x, rule.r, rule.phi);
    double num = 0.0, den = 0.0;
    for (std::size_t q = 0; q < rule.r.size(); ++q)
    {
        const double w = rule.weight[q] * rule.omega[q];
        num += w * (psi[q] - rule.f[q]) * (psi[q] - rule.f[q]);
        den += w * (psi_inf - rule.f[q]) * (psi_inf - rule.f[q]);
    }
    return std::sqrt(num / den);
}

const std::vector<double> kInterfaces{-0.25, 0.25};
const std::vector<double> kLayers{1110.0, 1080.0, 1000.0};
const std::vector<double> kX{0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 1.0};
} // namespace

TEST_CASE(blurriness_bare_matches_disk_quadrature)
{
    CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
    solver.set_max_root(40.0);
    solver.setup_bare_solution();
    // 64 nodes per direction and layer: the Gibbs oscillations of the truncated
    // series near the inlet need this to be integrated to round-off (40 leaves 1e-5).
    const DiskRule rule = layered_disk_rule(kInterfaces, kLayers, 64);
    const double psi_inf = solver.get_far_field_value();
    // The zero mode of the bare series is the flow-weighted inlet mean.
    double mean = 0.0, total = 0.0;
    for (std::size_t q = 0; q < rule.r.size(); ++q)
    {
        mean += rule.weight[q] * rule.omega[q] * rule.f[q];
        total += rule.weight[q] * rule.omega[q];
    }
    REQUIRE_APPROX(psi_inf, mean / total, 1e-10, 0.0);
    REQUIRE_APPROX(total, kHalfPi, 1e-10, 0.0);
    for (double x : kX)
        REQUIRE_APPROX(solver.get_blurriness(x, BlurrinessTail::Constant), quadrature_blurriness(solver, rule, x, psi_inf), 1e-9, 0.0);
    // Denominator in the user's units against the quadrature.
    double den = 0.0;
    for (std::size_t q = 0; q < rule.r.size(); ++q)
        den += rule.weight[q] * rule.omega[q] * (psi_inf - rule.f[q]) * (psi_inf - rule.f[q]);
    REQUIRE_APPROX(solver.get_inlet_far_field_square_norm(), den, 1e-9, 0.0);
    double norm = 0.0;
    for (std::size_t q = 0; q < rule.r.size(); ++q)
        norm += rule.weight[q] * rule.omega[q] * rule.f[q] * rule.f[q];
    REQUIRE_APPROX(solver.get_inlet_square_norm(), norm, 1e-10, 0.0);
}

TEST_CASE(blurriness_finite_peclet_matches_disk_quadrature)
{
    CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
    solver.setup_solution(5.0, 40.0);
    const DiskRule rule = layered_disk_rule(kInterfaces, kLayers, 64);
    const double psi_inf = solver.get_far_field_value();
    // The finite-Peclet far field differs from the flow-weighted mean by the
    // axial-diffusive flux through the entrance; it must still be the series' own
    // constant, i.e. what the series returns far downstream.
    REQUIRE_APPROX(psi_inf, solver.get_solution_at_points(50.0, {0.3}, {0.7})[0], 1e-8, 0.0);
    for (double x : kX)
        REQUIRE_APPROX(solver.get_blurriness(x, BlurrinessTail::Constant), quadrature_blurriness(solver, rule, x, psi_inf), 1e-9, 0.0);
}

TEST_CASE(blurriness_graetz_matches_disk_quadrature)
{
    CDGraetzIsothermalSolution<double> solver(2.0, 0.5, 0);
    solver.setup_solution(5.0, 40.0);
    // Uniform inlet: one "layer" spanning the disk; far field = wall value.
    const DiskRule rule = layered_disk_rule({}, {2.0}, 48);
    REQUIRE_APPROX(solver.get_far_field_value(), 0.5, 1e-13, 0.0);
    REQUIRE_APPROX(solver.get_inlet_square_norm(), 4.0 * kHalfPi, 1e-10, 0.0);
    REQUIRE_APPROX(solver.get_inlet_far_field_square_norm(), 1.5 * 1.5 * kHalfPi, 1e-10, 0.0);
    for (double x : {1e-3, 1e-2, 0.1, 0.5})
        REQUIRE_APPROX(solver.get_blurriness(x, BlurrinessTail::Constant), quadrature_blurriness(solver, rule, x, 0.5), 1e-6, 0.0);
}

TEST_CASE(blurriness_limits_monotonicity_and_normalisations)
{
    CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
    solver.setup_solution(10.0, 60.0);
    // With the tail: the floor at x = 0 is the Parseval defect, positive and small.
    const double floor_ = solver.get_blurriness_truncation_floor();
    REQUIRE(floor_ > 0.0);
    REQUIRE(floor_ < 0.2);
    REQUIRE_APPROX(solver.get_blurriness(0.0, BlurrinessTail::Constant), floor_, 1e-14, 0.0);
    // The default treatment is the asymptotic tail: zero at the inlet.
    REQUIRE_APPROX(solver.get_blurriness(0.0), 0.0, 0.0, 1e-14);
    REQUIRE_APPROX(solver.get_blurriness(1e-3), solver.get_blurriness(1e-3, BlurrinessTail::Asymptotic), 0.0, 0.0);
    // Without the tail the truncated basis starts exactly at zero.
    REQUIRE_APPROX(solver.get_blurriness(0.0, false), 0.0, 0.0, 1e-12);
    // All three reach one far downstream.
    REQUIRE_APPROX(solver.get_blurriness(std::numeric_limits<double>::infinity(), BlurrinessTail::Constant), 1.0, 1e-14, 0.0);
    REQUIRE_APPROX(solver.get_blurriness(std::numeric_limits<double>::infinity()), 1.0, 1e-14, 0.0);
    REQUIRE_APPROX(solver.get_blurriness(50.0, false), 1.0, 1e-12, 0.0);
    // Monotone in x and the vectorised overload agrees with the scalar one.
    const std::vector<double> xs{0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0};
    const std::vector<double> curve = solver.get_blurriness(xs, BlurrinessTail::Constant);
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        REQUIRE_APPROX(curve[i], solver.get_blurriness(xs[i], BlurrinessTail::Constant), 0.0, 0.0);
        if (i > 0)
            REQUIRE(curve[i] > curve[i - 1]);
    }
    // Refining the truncation lowers the floor (Parseval defect decays as beta_K^-1).
    CDStratifiedSolution<double> finer(kInterfaces, kLayers, 0);
    finer.setup_solution(10.0, 120.0);
    REQUIRE(finer.get_blurriness_truncation_floor() < floor_);
    // A negative x is rejected; a fresh object has no active solution.
    bool threw = false;
    try { solver.get_blurriness(-1.0); } catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);
    CDStratifiedSolution<double> fresh(kInterfaces, kLayers, 0);
    threw = false;
    try { fresh.get_blurriness(0.1); } catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
}

TEST_CASE(blurriness_is_invariant_under_affine_rescaling_of_the_inlet)
{
    CDStratifiedSolution<double> a(kInterfaces, kLayers, 0);
    a.setup_solution(20.0, 40.0);
    std::vector<double> rescaled;
    for (double u : kLayers)
        rescaled.push_back(0.01 * u - 3.0);
    CDStratifiedSolution<double> b(kInterfaces, rescaled, 0);
    b.setup_solution(20.0, 40.0);
    for (double x : kX)
        REQUIRE_APPROX(a.get_blurriness(x), b.get_blurriness(x), 1e-12, 0.0);
    REQUIRE_APPROX(a.get_inlet_far_field_square_norm() * 1e-4, b.get_inlet_far_field_square_norm(), 1e-10, 0.0);
}

TEST_CASE(blurriness_radial_projection_rejects_the_tail_but_not_the_modal_ratio)
{
    CDStratifiedSolution<double> radial(kInterfaces, kLayers, 0);
    radial.set_projection_space(ProjectionSpace::Radial);
    radial.setup_solution(5.0, 30.0);
    bool threw = false;
    try { radial.get_blurriness(0.1); } catch (const std::logic_error &) { threw = true; }
    REQUIRE(threw);
    const double modal = radial.get_blurriness(0.1, false);
    REQUIRE(modal > 0.0 && modal < 1.0);
    CDStratifiedSolution<double> weighted(kInterfaces, kLayers, 0);
    weighted.setup_solution(5.0, 30.0);
    // Two projections of the same inlet: same physics, agreement to the truncation level.
    REQUIRE_APPROX(modal, weighted.get_blurriness(0.1, false), 2e-2, 0.0);
}

TEST_CASE(blurriness_asymptotic_tail_starts_at_zero_and_matches_the_constant_tail_downstream)
{
    for (int variant = 0; variant < 2; ++variant)
    {
        CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
        if (variant == 0)
        {
            solver.set_max_root(60.0);
            solver.setup_bare_solution();
        }
        else
            solver.setup_solution(10.0, 60.0);
        // Exactly zero at the inlet and one far downstream.
        REQUIRE_APPROX(solver.get_blurriness(0.0, BlurrinessTail::Asymptotic), 0.0, 0.0, 1e-14);
        REQUIRE_APPROX(solver.get_blurriness(std::numeric_limits<double>::infinity(), BlurrinessTail::Asymptotic), 1.0, 1e-14, 0.0);
        // Below the truncation scale the tail follows the diffusive law: B^2 ~ x^{1/2}
        // for the bare series (Lam = beta^2) and B^2 ~ Pe x for the finite-Peclet
        // series, whose omitted modes decay on the physical scale R / beta
        // (Lam ~ beta / sqrt(kappa)).
        const double x1 = 1e-9, x2 = 1e-8;
        const double b1 = solver.get_blurriness(x1, BlurrinessTail::Asymptotic);
        const double b2 = solver.get_blurriness(x2, BlurrinessTail::Asymptotic);
        const double exponent = std::log(b2 * b2 / (b1 * b1)) / std::log(x2 / x1);
        const double expected = variant == 0 ? 0.5 : 1.0;
        REQUIRE(std::fabs(exponent - expected) < 0.05);
        // Monotone, between the modal-only and the constant-tail values, and equal
        // to the constant-tail value once the first omitted mode has decayed.
        double previous = 0.0;
        for (double x : {1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0})
        {
            const double asym = solver.get_blurriness(x, BlurrinessTail::Asymptotic);
            REQUIRE(asym >= previous);
            REQUIRE(asym >= solver.get_blurriness(x, BlurrinessTail::Dropped) - 1e-12);
            REQUIRE(asym <= solver.get_blurriness(x, BlurrinessTail::Constant) + 1e-12);
            previous = asym;
        }
        REQUIRE_APPROX(solver.get_blurriness(0.05, BlurrinessTail::Asymptotic),
                       solver.get_blurriness(0.05, BlurrinessTail::Constant), 1e-10, 0.0);
        // The bool overload is the Constant / None pair.
        REQUIRE_APPROX(solver.get_blurriness(1e-3, true), solver.get_blurriness(1e-3, BlurrinessTail::Constant), 0.0, 0.0);
        REQUIRE_APPROX(solver.get_blurriness(1e-3, false), solver.get_blurriness(1e-3, BlurrinessTail::Dropped), 0.0, 0.0);
    }
}

TEST_CASE(blurriness_asymptotic_tail_is_consistent_between_truncations)
{
    // The tail model of a coarse truncation must reproduce, within the accuracy of
    // the beta^-2 law, the blurriness of a finer truncation where the omitted
    // modes of the coarse one are retained (x above the fine truncation scale).
    CDStratifiedSolution<double> coarse(kInterfaces, kLayers, 0);
    coarse.setup_solution(100.0, 40.0);
    CDStratifiedSolution<double> fine(kInterfaces, kLayers, 0);
    fine.setup_solution(100.0, 160.0);
    for (double x : {1e-4, 3e-4, 1e-3, 3e-3})
        REQUIRE_APPROX(coarse.get_blurriness(x, BlurrinessTail::Asymptotic),
                       fine.get_blurriness(x, BlurrinessTail::Asymptotic), 3e-2, 0.0);
    // Whereas the constant tail of the coarse truncation overshoots there.
    REQUIRE(coarse.get_blurriness(1e-4, BlurrinessTail::Constant) > 1.2 * fine.get_blurriness(1e-4, BlurrinessTail::Asymptotic));
}

TEST_CASE(blurriness_inlet_value_is_the_parseval_defect_over_the_exact_variance)
{
    // The three scalars of the formula come from the Parseval analysis of the
    // inlet projection: D_K = ||f||^2 - ||f_K||^2 with ||f_K||^2 the stored
    // contraction (get_inlet_projection_square_norm), psi_inf the zero-rate term,
    // and || f - psi_inf ||^2 in closed form. Check the identity
    //   B_K(0)^2 = (||f||^2 - ||f_K||^2) / ||f - psi_inf||^2
    // in the user's scaling, and the Pythagorean split of the denominator into
    // the retained variance plus the defect (Galerkin orthogonality).
    for (int variant = 0; variant < 2; ++variant)
    {
        CDStratifiedSolution<double> solver(kInterfaces, kLayers, 0);
        if (variant == 0)
        {
            solver.set_max_root(80.0);
            solver.setup_bare_solution();
        }
        else
            solver.setup_solution(20.0, 80.0);
        const double defect = solver.get_inlet_square_norm() - solver.get_inlet_projection_square_norm();
        const double floor_ = solver.get_blurriness_truncation_floor();
        // (the defect is a difference of two norms 1e5 times larger, hence the tolerance)
        REQUIRE_APPROX(floor_ * floor_, defect / solver.get_inlet_far_field_square_norm(), 1e-9, 0.0);
        // Closed-form variance of the layered inlet about the far field.
        const double psi_inf = solver.get_far_field_value();
        // weighted area above the chord z: int_{z' > z} (1 - r^2) dA
        const auto partial_flux = [](double z) {
            const double za = std::fabs(z), y = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double f = (za * y * (2.0 * z * z - 5.0) + 3.0 * std::acos(za)) / 6.0;
            return z < 0.0 ? M_PI_2 - f : f;
        };
        std::vector<double> edges{-1.0, kInterfaces[0], kInterfaces[1], 1.0};
        double variance = 0.0;
        for (std::size_t i = 0; i < kLayers.size(); ++i)
            variance += (kLayers[i] - psi_inf) * (kLayers[i] - psi_inf)
                        * (partial_flux(edges[i]) - partial_flux(edges[i + 1]));
        REQUIRE_APPROX(solver.get_inlet_far_field_square_norm(), variance, 1e-12, 0.0);
        // Retained variance + defect = exact variance (Pythagoras in L2_omega).
        const double retained = solver.get_blurriness(1e3, BlurrinessTail::Dropped);   // == 1 by construction
        REQUIRE_APPROX(retained, 1.0, 1e-12, 0.0);
        const double amplitude = kLayers[0] - kLayers[2];
        // a_inf^T W a_inf in the user's units is the variance minus the defect:
        // the Dropped denominator; recover it from the two normalisations at a
        // downstream x where every retained mode has decayed.
        const double x = 5.0;
        const double b_dropped = solver.get_blurriness(x, BlurrinessTail::Dropped);
        const double b_constant = solver.get_blurriness(x, BlurrinessTail::Constant);
        // both are 1 - O(e^{-Lam x}); their ratio squared equals (variance - defect + eps)/(variance + eps)
        REQUIRE_APPROX(b_dropped, 1.0, 1e-9, 0.0);
        REQUIRE_APPROX(b_constant, 1.0, 1e-9, 0.0);
        static_cast<void>(amplitude);
    }
}
