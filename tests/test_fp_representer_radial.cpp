// Tests for the L2_r representer load-vector backend
// (src/finite_peclet_rhs_representer_radial.*, src/shifted_jacobi_basis_radial.*).
//
// The oracle throughout is fp_rhs_radial_stratified_inlet -- the direct cap
// quadrature on the SAME projection, itself pinned against an independent 2-D
// disk reference in tests/test_fp_radial_projection.cpp. That relationship is
// the one the weighted path already uses, and it is what catches a wrong Jacobi
// family, a missing factor of two, or a transposed index: those all survive a
// self-consistency check and none of them survives this one.
//
// Two facts that look like bugs and are not:
//  * The constant full-disk representer is sqrt(2*pi)/2 on BOTH projections.
//    That is not a copy-paste slip -- the 1/2 comes from r dr = ds/2, which
//    both measures carry, and the arccos(0) = pi/2 that produces it is cap
//    geometry.
//  * fp_representer_contract_radial has no factor of 1/2 even though the Gram
//    entry U_ij = 1/2 int s^n G_i G_j ds does. The load vector's own 1/2 is
//    already inside the cap prefactor. Same asymmetry as the weighted path.

#include "tinytest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#undef protected
#undef private

#include "finite_peclet_gram_radial.h"
#include "finite_peclet_rhs_radial.h"
#include "finite_peclet_rhs_representer_radial.h"
#include "finite_peclet_rhs_representer.h"
#include "shifted_jacobi_basis_radial.h"
#include "shifted_jacobi_basis.h"
#include "projection_space.h"
#include "rhs_method.h"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

namespace
{
constexpr double kSqrtTwoPi = 2.50662827463100050242;

std::vector<SeriesTermData<double>> angular_block(const std::vector<SeriesTermData<double>> &data, unsigned n)
{
    std::vector<SeriesTermData<double>> modes;
    for (const auto &t : data)
        if (t.n == n)
            modes.push_back(t);
    return modes;
}

// Synthetic block, independent of any root-finding machinery.
std::vector<SeriesTermData<double>> synthetic_block(unsigned n, unsigned K, double kappa)
{
    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        const double beta = 4.0 * (m + 1) + 2.0 * n + 2.0;
        const double beta2 = beta * beta;
        const double Lambda = 2.0 * beta2 / (1.0 + std::sqrt(1.0 + 4.0 * kappa * beta2));
        modes[m].n = n;
        modes[m].m = m;
        modes[m].root = beta;
        modes[m].root_fp = std::sqrt(Lambda);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * Lambda);
        modes[m].rate_fp = Lambda;
    }
    return modes;
}

// Composite Simpson on [0,1], for the independent norm/orthogonality references.
template <class G>
double simpson(unsigned panels, G &&g)
{
    if (panels % 2)
        ++panels;
    const double h = 1.0 / panels;
    double sum = 0.0;
    for (unsigned q = 0; q <= panels; ++q)
    {
        const double w = (q == 0 || q == panels) ? 1.0 : (q % 2 == 0 ? 2.0 : 4.0);
        sum += w * g(q * h);
    }
    return sum * h / 3.0;
}

double max_abs(const std::vector<double> &v)
{
    double m = 0.0;
    for (double x : v)
        m = std::max(m, std::abs(x));
    return m;
}
} // namespace

// ═══ The (0,n) shifted-Jacobi family ════════════════════════════════════════

TEST_CASE(fp_radial_shifted_jacobi_norm_matches_quadrature)
{
    for (unsigned n : {0u, 1u, 5u, 20u, 40u})
        for (unsigned j : {0u, 1u, 3u, 7u, 15u})
        {
            const double h = shifted_jacobi_norm_radial<double>(j, n);
            // Independent reference: h_j = int_0^1 s^n [P_j]^2 ds by Simpson,
            // with the polynomial itself taken from the table under test only
            // through its VALUES -- the norm formula is what is being checked.
            const auto ref = simpson(200000u, [&](double s) {
                const std::vector<double> node{s};
                const auto tab = shifted_jacobi_table_radial<double>(j + 1, n, node);
                return std::pow(s, static_cast<double>(n)) * tab[j] * tab[j];
            });
            REQUIRE(std::abs(h - ref) / ref < 1e-8);
        }

    // Anchors: h_0 is the total mass, and n = 0 is shifted Legendre.
    for (unsigned n : {0u, 3u, 17u})
        REQUIRE(std::abs(shifted_jacobi_norm_radial<double>(0u, n) - 1.0 / (n + 1.0)) < 1e-15);
    for (unsigned j : {0u, 1u, 9u, 30u})
        REQUIRE(std::abs(shifted_jacobi_norm_radial<double>(j, 0u) - 1.0 / (2.0 * j + 1.0)) < 1e-15);

    // It must NOT coincide with the (1,n) family's norm, except where the two
    // formulas genuinely agree -- a guard against including the wrong header.
    REQUIRE(std::abs(shifted_jacobi_norm_radial<double>(3u, 5u)
                     - shifted_jacobi_norm<double>(3u, 5u)) > 1e-3);
}

TEST_CASE(fp_radial_shifted_jacobi_table_is_orthogonal_in_its_own_measure)
{
    for (unsigned n : {0u, 2u, 9u})
    {
        const unsigned J = 8u;
        std::vector<double> nodes;
        const unsigned panels = 40000u;
        for (unsigned q = 0; q <= panels; ++q)
            nodes.push_back(static_cast<double>(q) / panels);
        const auto tab = shifted_jacobi_table_radial<double>(J, n, nodes);

        for (unsigned i = 0; i < J; ++i)
            for (unsigned j = 0; j < J; ++j)
            {
                double sum = 0.0;
                for (unsigned q = 0; q <= panels; ++q)
                {
                    const double s = nodes[q];
                    const double w = (q == 0 || q == panels) ? 1.0 : (q % 2 == 0 ? 2.0 : 4.0);
                    sum += w * std::pow(s, static_cast<double>(n))
                             * tab[static_cast<std::size_t>(i) * nodes.size() + q]
                             * tab[static_cast<std::size_t>(j) * nodes.size() + q];
                }
                sum *= (1.0 / panels) / 3.0;
                const double expected = (i == j) ? shifted_jacobi_norm_radial<double>(i, n) : 0.0;
                REQUIRE(std::abs(sum - expected) < 1e-7 * shifted_jacobi_norm_radial<double>(0u, n));
            }
    }
}

// ═══ The order rule ═════════════════════════════════════════════════════════

TEST_CASE(fp_radial_representer_order_is_one_degree_below_the_weighted_rule)
{
    FPRepresenterRadialOptions<double> ro;
    FPRepresenterOptions<double> wo;
    ro.representer_margin = wo.representer_margin = 0u;

    // n >= 1: exact, and cheaper by exactly the degree the (1-eta) factor cost.
    for (unsigned n : {1u, 2u, 3u, 7u, 20u, 99u})
        for (unsigned J : {4u, 10u, 41u, 220u})
        {
            const unsigned half = (n - 1u) / 2u;
            const unsigned expected = (J + half + 1u) / 2u; // ceil((J + half)/2)
            REQUIRE(fp_representer_order_radial<double>(n, J, 0.4, ro) == expected);
            // and never more than the weighted rule, which stays valid here
            REQUIRE(fp_representer_order_radial<double>(n, J, 0.4, ro)
                    <= fp_representer_order<double>(n, J, 0.4, wo));
        }

    // n == 0: the 1/a form, with the clamp at a = 0.01.
    for (double a : {0.05, 0.2, 0.9})
    {
        const unsigned J = 40u;
        const unsigned expected = (J + 1u) / 2u + static_cast<unsigned>(std::ceil(6.0 / a));
        REQUIRE(fp_representer_order_radial<double>(0u, J, a, ro) == expected);
    }
    REQUIRE(fp_representer_order_radial<double>(0u, 40u, 0.0, ro)
            == fp_representer_order_radial<double>(0u, 40u, 0.01, ro));
}

// ═══ The oracle: representer against the direct cap quadrature ══════════════

TEST_CASE(fp_radial_representer_matches_direct_stratified_load_vector)
{
    const std::vector<double> zi{-0.4, 0.2};
    const std::vector<double> ui{1.0, 0.3, -0.2};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(40.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    FPRepresenterRadialOptions<double> options;
    for (unsigned n : {0u, 1u, 2u, 3u, 5u, 10u})
    {
        const auto modes = angular_block(sol.m_series_data, n);
        if (modes.empty())
            continue;
        FPGramRadialOptions<double> gram_options;
        gram_options.retain_samples = true;
        const auto factor = fp_gram_factor_radial(n, modes, gram_options);

        const auto direct = fp_rhs_radial_stratified_inlet<double>(n, modes, zi, ui, 100u);
        const auto repr = fp_rhs_radial_stratified_inlet_representer<double>(n, factor, zi, ui, options);
        REQUIRE(direct.size() == repr.size());
        const double scale = max_abs(direct);
        REQUIRE(scale > 0.0);
        for (std::size_t m = 0; m < direct.size(); ++m)
            REQUIRE(std::abs(direct[m] - repr[m]) / scale < 1e-9);
    }
}

TEST_CASE(fp_radial_representer_matches_direct_uniform_inlet_load_vector)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(40.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    const auto modes = angular_block(sol.m_series_data, 0u);
    REQUIRE(modes.size() >= 3u);
    FPGramRadialOptions<double> gram_options;
    gram_options.retain_samples = true;
    const auto factor = fp_gram_factor_radial(0u, modes, gram_options);
    // Dirichlet wall: no exact constant mode, so the representer route is
    // genuinely exercised rather than short-circuited by the Gram column.
    REQUIRE(!factor.has_constant_mode);

    FPRepresenterRadialOptions<double> options;
    const auto direct = fp_rhs_radial_uniform_inlet<double>(0u, modes, 100u);
    const auto repr = fp_rhs_radial_uniform_inlet_representer<double>(0u, factor, options);
    const double scale = max_abs(direct);
    REQUIRE(scale > 0.0);
    for (std::size_t m = 0; m < direct.size(); ++m)
        REQUIRE(std::abs(direct[m] - repr[m]) / scale < 1e-9);
}

// Degenerate caps: an interface at exactly z = 0 drives the n = 0 "center"
// branch (representer returns empty, caller substitutes half the full disk) and
// the even-n "zero" branch. Both must agree with the direct route.
TEST_CASE(fp_radial_representer_degenerate_cap_branches_match_direct)
{
    const std::vector<double> zi{0.0};
    const std::vector<double> ui{1.0, 0.0};
    CDStratifiedSolution<double> sol(zi, ui, 0);
    sol.set_max_root(40.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    FPRepresenterRadialOptions<double> options;
    bool saw_center = false, saw_zero = false;
    for (unsigned n : {0u, 1u, 2u, 3u, 4u})
    {
        const auto modes = angular_block(sol.m_series_data, n);
        if (modes.empty())
            continue;
        FPGramRadialOptions<double> gram_options;
        gram_options.retain_samples = true;
        const auto factor = fp_gram_factor_radial(n, modes, gram_options);

        const auto psi = fp_cap_representer_radial<double>(n, 0.0, factor.node_count,
                                                          factor.quadrature_nodes, options);
        if (n == 0u)
        {
            REQUIRE(psi.empty()); // center: caller substitutes half the full disk
            saw_center = true;
        }
        else if (n % 2u == 0u)
        {
            REQUIRE(psi.size() == factor.node_count);
            for (double v : psi)
                REQUIRE(v == 0.0); // zero branch
            saw_zero = true;
        }

        const auto direct = fp_rhs_radial_stratified_inlet<double>(n, modes, zi, ui, 100u);
        const auto repr = fp_rhs_radial_stratified_inlet_representer<double>(n, factor, zi, ui, options);
        const double scale = std::max(max_abs(direct), 1e-300);
        for (std::size_t m = 0; m < direct.size(); ++m)
            REQUIRE(std::abs(direct[m] - repr[m]) <= 1e-9 * scale + 1e-14);
    }
    REQUIRE(saw_center);
    REQUIRE(saw_zero);

    // a >= 1 is the empty cap, on either projection.
    const auto rule = make_fp_gauss_jacobi_rule<double>(40u);
    REQUIRE(fp_rhs_radial_build_cap_kernel<double>(0u, 1.0, rule).empty);
}

// ═══ The constant full-disk representer ═════════════════════════════════════

TEST_CASE(fp_radial_constant_representer_matches_full_disk_projection)
{
    const auto modes = synthetic_block(0u, 10u, 1e-2);
    FPGramRadialOptions<double> gram_options;
    gram_options.retain_samples = true;
    const auto factor = fp_gram_factor_radial(0u, modes, gram_options);

    // sqrt(2*pi)/2 -- numerically the SAME constant the weighted path uses.
    const std::vector<double> psi(factor.node_count, kSqrtTwoPi / 2.0);
    const auto contracted = fp_representer_contract_radial(factor, psi);
    const auto quadrature = fp_rhs_radial_full_disk_projection<double>(modes, 100u);
    REQUIRE(contracted.size() == quadrature.size());
    const double scale = max_abs(quadrature);
    REQUIRE(scale > 0.0);
    for (std::size_t m = 0; m < contracted.size(); ++m)
        REQUIRE(std::abs(contracted[m] - quadrature[m]) / scale < 1e-9);
}

// With a Neumann n = 0 block the exact Gram column applies; the representer
// route must reproduce it, since both are exact identities for the same term.
TEST_CASE(fp_radial_constant_representer_matches_gram_column)
{
    CDStratifiedSolution<double> sol({-0.4, 0.2}, {1.0, 0.3, -0.2}, 0);
    sol.set_max_root(40.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    const auto modes = angular_block(sol.m_series_data, 0u);
    FPGramRadialOptions<double> gram_options;
    gram_options.retain_samples = true;
    const auto factor = fp_gram_factor_radial(0u, modes, gram_options);
    REQUIRE(factor.has_constant_mode);

    const auto from_column = fp_rhs_radial_full_disk_from_gram_column<double>(factor.constant_mode_gram_column);
    const std::vector<double> psi(factor.node_count, kSqrtTwoPi / 2.0);
    const auto from_representer = fp_representer_contract_radial(factor, psi);
    const double scale = max_abs(from_column);
    REQUIRE(scale > 0.0);
    for (std::size_t m = 0; m < from_column.size(); ++m)
        REQUIRE(std::abs(from_column[m] - from_representer[m]) / scale < 1e-10);
}

// ═══ Contract preconditions and the design contract ═════════════════════════

TEST_CASE(fp_radial_representer_contract_throws_without_retained_samples)
{
    const auto modes = synthetic_block(0u, 10u, 1e-2);
    FPGramRadialOptions<double> gram_options; // retain_samples defaults to false
    const auto factor = fp_gram_factor_radial(0u, modes, gram_options);
    REQUIRE(!factor.samples_retained);

    const std::vector<double> psi(factor.node_count, 1.0);
    bool threw = false;
    try { fp_representer_contract_radial(factor, psi); }
    catch (const std::runtime_error &) { threw = true; }
    REQUIRE(threw);
}

TEST_CASE(fp_radial_representer_never_calls_fp_radial_factor_source_grep)
{
    // Zero confluent-hypergeometric evaluations is the entire point of this
    // backend. Search for a call site, not the bare identifier: the header
    // comment names it in prose.
    for (const char *file : {"/src/finite_peclet_rhs_representer_radial.cpp",
                             "/src/shifted_jacobi_basis_radial.cpp"})
    {
        std::ifstream in(std::string(TEST_DATA_DIR) + file);
        REQUIRE(in.good());
        std::ostringstream buf;
        buf << in.rdbuf();
        const std::string text = buf.str();
        REQUIRE(text.find("fp_radial_factor(") == std::string::npos);
        REQUIRE(text.find("fp_radial_factor<") == std::string::npos);
        REQUIRE(text.find("psinm_r_fp(") == std::string::npos);
    }
}

// ═══ The cancellation ratio -- the acceptance gate ══════════════════════════
//
// The truncated representer is enormous where the exact one is singular
// (max|psi| runs to ~1e43 at high n), and the result is only accurate because
// the s^n of the Gram measure compensates it pointwise. The quantity that
// certifies this is the cancellation ratio
//     K = sum_q w_q |G psi| / |L|  >= 1 ,
// the number of digits it costs being log10(K). The weighted path measures
// K <= 2e3 over the full retained range. The theory predicts the unweighted
// value should be comparable or slightly smaller, since its cap functional is
// not damped at the wall -- that was a prediction, and this test is where it
// gets measured.
TEST_CASE(fp_radial_representer_cancellation_ratio_stays_bounded)
{
    CDStratifiedSolution<double> sol({-0.4, 0.2}, {1.0, 0.3, -0.2}, 0);
    sol.set_max_root(60.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.setup_fp_solution(8.0);

    FPRepresenterRadialOptions<double> options;
    double worst = 1.0;
    unsigned highest_n = 0;
    for (const auto &t : sol.m_series_data)
        highest_n = std::max(highest_n, t.n);

    for (unsigned n = 0; n <= highest_n; ++n)
    {
        const auto modes = angular_block(sol.m_series_data, n);
        if (modes.empty())
            continue;
        FPGramRadialOptions<double> gram_options;
        gram_options.retain_samples = true;
        const auto factor = fp_gram_factor_radial(n, modes, gram_options);
        for (double a : {0.2, 0.4, 0.75})
        {
            const auto psi = fp_cap_representer_radial<double>(n, a, factor.node_count,
                                                              factor.quadrature_nodes, options);
            if (psi.empty())
                continue;
            const auto exact = fp_representer_contract_radial(factor, psi);
            for (unsigned m = 0; m < factor.mode_count; ++m)
            {
                double abs_sum = 0.0;
                for (unsigned q = 0; q < factor.node_count; ++q)
                    abs_sum += factor.quadrature_weights[q]
                             * std::abs(factor.radial_samples[static_cast<std::size_t>(q) * factor.mode_count + m]
                                        * psi[q]);
                if (std::abs(exact[m]) > 0.0)
                    worst = std::max(worst, abs_sum / std::abs(exact[m]));
            }
        }
    }
    // The weighted path's measured bound is 2e3; this asserts the unweighted
    // one is no worse by more than an order of magnitude, which is what the
    // "comparable or slightly smaller" prediction has to earn.
    REQUIRE(worst >= 1.0);
    REQUIRE(worst < 2e4);
}

// ═══ End to end ═════════════════════════════════════════════════════════════

TEST_CASE(fp_radial_representer_field_agreement_stratified)
{
    const std::vector<double> zi{-0.4, 0.82};
    const std::vector<double> ui{1.0, 0.3, 0.0};
    const double max_root = 40.0;

    CDStratifiedSolution<double> direct(zi, ui, 0);
    direct.set_max_root(max_root);
    direct.set_projection_space(ProjectionSpace::Radial);
    direct.set_rhs_method(RhsMethod::DirectQuadrature);
    direct.setup_fp_solution(8.0);

    CDStratifiedSolution<double> repr(zi, ui, 0);
    repr.set_max_root(max_root);
    repr.set_projection_space(ProjectionSpace::Radial);
    repr.set_rhs_method(RhsMethod::Representer);
    repr.setup_fp_solution(8.0);

    const std::vector<double> x{0.0, 0.05, 0.3, 0.9};
    const std::vector<double> r{0.1, 0.5, 0.9, 0.3};
    const std::vector<double> phi{0.0, 1.0, 2.5, -1.2};
    const auto vd = direct.get_solution_at_points(x, r, phi);
    const auto vr = repr.get_solution_at_points(x, r, phi);
    REQUIRE(vd.size() == vr.size());
    double scale = 1.0;
    for (double v : vd)
        scale = std::max(scale, std::abs(v));
    for (std::size_t i = 0; i < vd.size(); ++i)
    {
        REQUIRE(std::isfinite(vd[i]));
        REQUIRE(std::isfinite(vr[i]));
        REQUIRE(std::abs(vd[i] - vr[i]) / scale < 1e-9);
    }
}

TEST_CASE(fp_radial_representer_coefficient_agreement_graetz)
{
    CDGraetzIsothermalSolution<double> direct(1.0, 0.0, 0);
    direct.set_max_root(40.0);
    direct.set_projection_space(ProjectionSpace::Radial);
    direct.set_rhs_method(RhsMethod::DirectQuadrature);
    direct.setup_fp_solution(8.0);

    CDGraetzIsothermalSolution<double> repr(1.0, 0.0, 0);
    repr.set_max_root(40.0);
    repr.set_projection_space(ProjectionSpace::Radial);
    repr.set_rhs_method(RhsMethod::Representer);
    repr.setup_fp_solution(8.0);

    REQUIRE(direct.m_series_data.size() == repr.m_series_data.size());
    double scale = 0.0;
    for (const auto &t : direct.m_series_data)
        scale = std::max(scale, std::abs(t.coeff_fp));
    REQUIRE(scale > 0.0);
    for (std::size_t k = 0; k < direct.m_series_data.size(); ++k)
        REQUIRE(std::abs(direct.m_series_data[k].coeff_fp - repr.m_series_data[k].coeff_fp) / scale < 1e-9);
}

// The weighted path must be untouched by any of this: same configuration, same
// frozen values as tests/test_fp_radial_projection.cpp pins.
TEST_CASE(fp_radial_representer_leaves_weighted_path_unchanged)
{
    CDStratifiedSolution<double> sol({-0.4, 0.2}, {1.0, 0.3, -0.2}, 0);
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Weighted);
    REQUIRE(sol.get_rhs_method() == RhsMethod::Representer);
    sol.set_max_root(30.0);
    sol.setup_fp_solution(5.0);
    const std::vector<double> reference{
        1.5092733103397653,
        0.22056901642144769,
        -0.14620962306460902,
        -0.19678120997867271,
    };
    double scale = 0.0;
    for (double v : reference)
        scale = std::max(scale, std::abs(v));
    for (std::size_t k = 0; k < reference.size(); ++k)
        REQUIRE(std::abs(sol.m_series_data[k].coeff_fp - reference[k]) / scale < 1e-14);
}
