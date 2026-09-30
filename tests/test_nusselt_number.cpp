#include "tinytest.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "nusselt_golden_dump.h"
#include "nusselt_postprocessing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double sqrt_two_pi = 2.5066282746310005;

template <class F> bool throws_logic_containing(F f, const std::string &fragment)
{
    try { f(); }
    catch (const std::logic_error &e) { return std::string(e.what()).find(fragment) != std::string::npos; }
    return false;
}

template <class F> double simpson(F f, unsigned intervals = 4000)
{
    double sum = f(0.0) + f(1.0);
    for (unsigned j = 1; j < intervals; ++j)
        sum += (j & 1u ? 4.0 : 2.0) * f(double(j) / intervals);
    return sum / (3.0 * intervals);
}

// Direct radial integration and a Richardson wall derivative of the evaluated
// temperature series. Neither uses the Nusselt mode records.
template <class Evaluator> double direct_nusselt(Evaluator eval, double x)
{
    std::vector<double> r(401);
    for (unsigned j=0; j<=400; ++j) r[j] = double(j)/400.0;
    r.insert(r.end(), {0.9995,0.999,0.998});
    const auto values = eval(x, r);
    double sum = 0.0;
    for (unsigned j=0; j<=400; ++j) {
        const double weight = (j==0 || j==400) ? 1.0 : (j&1u ? 4.0 : 2.0);
        sum += weight*(1.0-r[j]*r[j])*values[j]*r[j];
    }
    const double bulk_integral = sum/1200.0;
    const auto slope = [&](double h, double f1, double f2) {
        return (-4.0*f1+f2)/(2.0*h);
    };
    const double derivative = (4.0*slope(5e-4,values[401],values[402])
                               -slope(1e-3,values[402],values[403]))/3.0;
    return -derivative / (2.0 * bulk_integral);
}

CDGraetzIsothermalSolution<double> graetz(double max_root = 25.0)
{
    CDGraetzIsothermalSolution<double> s(1.0, 0.0, 0);
    s.set_max_root(max_root);
    return s;
}

// L2_r projection: its weight does not vanish at the wall, so the retained
// c_k R_k'(1) reach the Bessel limit -2 sqrt(2 pi) across the whole block. The
// Weighted projection leaves the second half of the block short of it, which the
// split evaluations cannot repair (see get_nusselt_number in CDBaseSolution.h).
CDGraetzIsothermalSolution<double> graetz_radial(double max_root, double pe)
{
    auto s = graetz(max_root);
    s.set_rhs_method(RhsMethod::DirectQuadrature);
    s.set_projection_space(ProjectionSpace::Radial);
    s.setup_fp_solution(pe);
    return s;
}
}

TEST_CASE(nusselt_stratified_always_throws)
{
    CDStratifiedSolution<double> s({0.0}, {1.0, 0.0}, 0);
    const auto check = [&]() {
        REQUIRE(throws_logic_containing([&] { s.get_nusselt_number(0.1); }, "mass transport"));
        REQUIRE(throws_logic_containing([&] { s.get_nusselt_number(std::vector<double>{0.1}); }, "mass transport"));
        REQUIRE(throws_logic_containing([&] { s.get_fully_developed_nusselt_number(); }, "mass transport"));
        REQUIRE(throws_logic_containing([&] { s.get_nusselt_mode_data(); }, "mass transport"));
    };
    check();
    s.set_max_root(10.0);
    s.setup_bare_solution(); check();
    s.setup_fp_solution(5.0); check();
}

TEST_CASE(nusselt_requires_active_solution)
{
    auto s = graetz();
    REQUIRE(throws_logic_containing([&] { s.get_nusselt_number(0.1); }, "solution"));
    REQUIRE(throws_logic_containing([&] { s.get_fully_developed_nusselt_number(); }, "solution"));
}

TEST_CASE(nusselt_rejects_negative_x_and_qep)
{
    auto s = graetz(); s.setup_bare_solution();
    bool bad_x = false, nan_x = false;
    try { s.get_nusselt_number(-0.1); } catch (const std::invalid_argument &) { bad_x = true; }
    try { s.get_nusselt_number(std::numeric_limits<double>::quiet_NaN()); }
    catch (const std::invalid_argument &) { nan_x = true; }
    REQUIRE(bad_x && nan_x);
    s.setup_qep_solution(5.0);
    REQUIRE(throws_logic_containing([&] { s.get_nusselt_number(0.1); }, "QEP"));
}

TEST_CASE(nusselt_bare_mode_identities)
{
    auto s = graetz(40.0); s.setup_bare_solution();
    const auto modes = s.get_nusselt_mode_data();
    const auto roots = s.get_bare_root_catalog().at(0);
    const auto coeff = s.get_coefficients().at(0);
    const auto norms = s.get_norms().at(0);
    REQUIRE(!modes.empty());
    for (const auto &v : modes) {
        const unsigned m = static_cast<unsigned>(v[0]);
        const double beta = roots.at(m);
        REQUIRE_APPROX(v[1], beta*beta, 1e-13, 0.0);
        REQUIRE_APPROX(v[2], coeff.at(m), 1e-13, 0.0);
        REQUIRE_APPROX(v[3], dpsinm_r<double>(1, 0, beta, 1.0), 1e-12, 1e-13);
        REQUIRE_APPROX(v[1]*v[4] + v[3], 0.0, 0.0, 1e-9 * std::max(1.0, std::fabs(v[3])));
        REQUIRE_APPROX(sqrt_two_pi*v[4], v[2]*norms.at(m), 1e-9, 1e-10);
    }
}

TEST_CASE(nusselt_inlet_bulk_matches_projection_norm)
{
    for (double pe : {0.0, 5.0, 50.0}) {
        auto s = graetz(25.0);
        if (pe == 0.0) s.setup_bare_solution(); else s.setup_fp_solution(pe);
        double projection = 0.0;
        for (const auto &v : s.get_nusselt_mode_data()) projection += v[2]*sqrt_two_pi*v[4];
        REQUIRE_APPROX(projection, s.get_inlet_projection_square_norm(), 1e-8, 1e-10);
    }
}

TEST_CASE(nusselt_fp_flux_balance)
{
    for (double pe : {5.0, 1.0}) {
        auto s = graetz(25.0); s.setup_fp_solution(pe);
        for (const auto &v : s.get_nusselt_mode_data()) {
            const double b = std::sqrt(v[1]);
            const double bt = b * (1.0 + v[1]/(pe*pe));
            const double a = simpson([&](double r) { return psinm_r_fp<double>(0, b, bt, r)*r; });
            const double moment = simpson([&](double r) { return (1-r*r)*psinm_r_fp<double>(0,b,bt,r)*r; });
            const double residual = v[1]*v[4] + v[3] + v[1]*v[1]*a/(pe*pe);
            const double scale = std::max({1.0, std::fabs(v[1]*v[4]), std::fabs(v[3]), std::fabs(v[1]*v[1]*a/(pe*pe))});
            REQUIRE(std::fabs(residual) < 1e-7*scale);
            REQUIRE_APPROX(v[4], moment, 1e-7, 1e-9);
        }
    }
}

TEST_CASE(nusselt_does_not_perturb_solution_state)
{
    for (double pe : {0.0, 5.0}) {
        auto s = graetz(25.0);
        if (pe == 0.0) s.setup_bare_solution(); else s.setup_fp_solution(pe);
        const auto eval = [&]() { return s.get_solution_at_points(0.2, std::vector<double>{0.0,0.4,0.8}, std::vector<double>{0,0,0}); };
        const auto before = eval();
        const auto coeff = s.get_coefficients();
        const auto modified = s.get_modified_data();
        const double norm = s.get_inlet_projection_square_norm();
        s.get_nusselt_number(std::vector<double>{0.1,1.0});
        s.get_nusselt_number(0.2);
        s.get_fully_developed_nusselt_number();
        s.get_nusselt_mode_data();
        s.nusselt_converged_x_min(1e-8);
        if (pe != 0.0) {
            s.get_nusselt_number(std::vector<double>{0.05,1.0}, NusseltEvaluation::SpectralSplit);
            s.get_nusselt_number(0.05, NusseltEvaluation::SlugSplit);
            s.get_nusselt_tail_rates(0.05);
        }
        REQUIRE(eval() == before);
        REQUIRE(s.get_coefficients() == coeff);
        REQUIRE(s.get_modified_data() == modified);
        REQUIRE(s.get_inlet_projection_square_norm() == norm);
    }
}

TEST_CASE(nusselt_overloads_and_long_double)
{
    auto s = graetz(); s.setup_bare_solution();
    const std::vector<double> xs{0.01,0.1,1.0};
    const auto values = s.get_nusselt_number(xs);
    for (std::size_t i=0; i<xs.size(); ++i) REQUIRE(values[i] == s.get_nusselt_number(xs[i]));
    REQUIRE(s.get_nusselt_number(std::vector<double>{}).empty());
    CDGraetzIsothermalSolution<long double> ld(1.0L,0.0L,0);
    ld.set_max_root(25.0L); ld.setup_bare_solution();
    const auto root = ld.get_bare_root_catalog().at(0).at(0);
    REQUIRE_APPROX(ld.get_fully_developed_nusselt_number(), root*root/2.0L, 1e-12L, 0.0L);
}

TEST_CASE(nusselt_classical_fully_developed)
{
    auto s = graetz(); s.setup_bare_solution();
    const double root = s.get_bare_root_catalog().at(0).at(0);
    const double expected = root*root/2.0;
    REQUIRE_APPROX(s.get_fully_developed_nusselt_number(), expected, 1e-10, 0.0);
    REQUIRE_APPROX(s.get_nusselt_number(1000.0), expected, 1e-12, 0.0);
    REQUIRE_APPROX(s.get_nusselt_number(std::numeric_limits<double>::infinity()), expected, 1e-12, 0.0);
    REQUIRE_APPROX(expected, 3.6568, 0.0, 5e-4);
}

TEST_CASE(nusselt_classical_leveque_asymptote)
{
    auto s = graetz(400.0); s.setup_bare_solution();
    const double c = 6.0*std::cbrt(2.0/9.0)/std::tgamma(1.0/3.0);
    const double d1 = std::fabs(s.get_nusselt_number(1e-3)*std::cbrt(1e-3)/c - 1.0);
    const double d2 = std::fabs(s.get_nusselt_number(2e-4)*std::cbrt(2e-4)/c - 1.0);
    REQUIRE(d1 < 0.07);
    REQUIRE(d2 < 0.04);
    REQUIRE(d2 < d1);
}

TEST_CASE(nusselt_pure_conduction_limit)
{
    const double q = 2.404825557695773;
    const double expected = std::pow(q,4)/8.0;
    auto s1 = graetz(25.0); s1.setup_fp_solution(1e-2);
    auto s2 = graetz(25.0); s2.setup_fp_solution(1e-3);
    const double e1 = std::fabs(s1.get_fully_developed_nusselt_number()/expected - 1.0);
    const double e2 = std::fabs(s2.get_fully_developed_nusselt_number()/expected - 1.0);
    REQUIRE(e1 < 0.05);
    REQUIRE(e2 < 0.01);
    REQUIRE(e2 < e1);
}

TEST_CASE(nusselt_matches_brute_force_from_get_solution)
{
    for (double pe : {0.0,5.0,50.0}) {
        auto s = graetz(25.0);
        if (pe == 0.0) s.setup_bare_solution(); else s.setup_fp_solution(pe);
        const auto eval = [&](double x,const std::vector<double> &r) {
            return s.get_solution_at_points(x,r,std::vector<double>(r.size(),0.0));
        };
        for (double x : {0.05,0.2,1.0})
            REQUIRE_APPROX(s.get_nusselt_number(x), direct_nusselt(eval,x), 1e-5, 0.0);
    }
}

TEST_CASE(nusselt_matches_qep_method)
{
    for (double pe : {5.0,50.0}) {
        // The bare-basis QEP wall derivative converges slowly at low Pe.
        // At beta_K=400 its Pe=5, x=0.1 residual is still about 9e-4;
        // compare that station by convergence rather than disguising it with
        // a looser agreement threshold.
        auto fp = graetz(400.0); fp.setup_fp_solution(pe);
        auto qep = graetz(400.0);
        const auto eval = [&](double x,const std::vector<double> &r) {
            return qep.get_QEP_solution_at_points(pe,std::vector<double>(r.size(),x),r,
                                                   std::vector<double>(r.size(),0.0));
        };
        for (double x : {0.5,1.0})
            REQUIRE_APPROX(fp.get_nusselt_number(x),direct_nusselt(eval,x),1e-4,0.0);
        if (pe == 50.0)
            REQUIRE_APPROX(fp.get_nusselt_number(0.1),direct_nusselt(eval,0.1),1e-4,0.0);
        else {
            auto qep_low = graetz(80.0);
            const auto eval_low = [&](double x,const std::vector<double> &r) {
                return qep_low.get_QEP_solution_at_points(pe,std::vector<double>(r.size(),x),r,
                                                           std::vector<double>(r.size(),0.0));
            };
            const double target = fp.get_nusselt_number(0.1);
            REQUIRE(std::fabs(direct_nusselt(eval,0.1)-target)
                  < std::fabs(direct_nusselt(eval_low,0.1)-target));
        }
    }
}

TEST_CASE(nusselt_monotone_and_above_asymptote)
{
    for (double pe : {0.0,5.0}) {
        auto s = graetz(100.0);
        if (pe == 0.0) s.setup_bare_solution(); else s.setup_fp_solution(pe);
        const double developed = s.get_fully_developed_nusselt_number();
        double previous = std::numeric_limits<double>::infinity();
        for (unsigned i=0; i<=30; ++i) {
            const double x = 1e-3*std::pow(5000.0,double(i)/30.0);
            const double nu = s.get_nusselt_number(x);
            // Once higher modes round to zero, the computed ratio is exactly
            // the fully developed double value and strict inequalities cease.
            REQUIRE(nu <= previous);
            REQUIRE(nu >= developed);
            if (previous - developed > 1e-12) REQUIRE(nu < previous);
            previous = nu;
        }
    }
}

TEST_CASE(nusselt_fp_reduces_to_bare_at_large_peclet)
{
    auto bare = graetz(25.0); bare.setup_bare_solution();
    auto fp = graetz(25.0); fp.setup_fp_solution(1e4);
    for (double x : {0.05,0.2,1.0})
        REQUIRE_APPROX(fp.get_nusselt_number(x), bare.get_nusselt_number(x), 1e-5, 0.0);
    REQUIRE_APPROX(fp.get_fully_developed_nusselt_number(),bare.get_fully_developed_nusselt_number(),1e-6,0.0);
}

TEST_CASE(nusselt_fp_fully_developed_monotone_in_peclet)
{
    auto bare = graetz(); bare.setup_bare_solution();
    double previous = 4.1807;
    for (double pe : {1.0,5.0,50.0}) {
        auto s = graetz(); s.setup_fp_solution(pe);
        const double nu = s.get_fully_developed_nusselt_number();
        REQUIRE(nu < previous);
        REQUIRE(nu > bare.get_fully_developed_nusselt_number());
        previous = nu;
    }
}

// Shah and London (1978), circular duct Tables 13 and 15, classical column.
// Their x* is half this implementation's x. The tabulated local values are
// rounded to two decimal places, so 1e-3 is the precision they support here.
// https://www.sciencedirect.com/topics/engineering/graetz-problem
TEST_CASE(nusselt_classical_local_values_tabulated)
{
    auto s = graetz(400.0); s.setup_bare_solution();
    const double stations[][2] = {
        {0.001,10.13},{0.002,8.04},{0.003,7.04},{0.004,6.43},
        {0.005,6.00},{0.010,4.92},{0.020,4.17},{0.050,3.71},
        {0.100,3.66},{0.200,3.66}
    };
    for (const auto &row : stations)
        REQUIRE_APPROX(s.get_nusselt_number(2.0*row[0]),row[1],1e-3,0.0);
}

// Shah and London (1978), circular duct Table 6, independent tabulation of
// the finite-Pe Dirichlet wall asymptote. Pe uses U_mean*D/alpha = U_max*R/alpha.
// https://www.sciencedirect.com/topics/engineering/graetz-problem
TEST_CASE(nusselt_fully_developed_with_axial_conduction_tabulated)
{
    const double values[][2] = {
        {1.0,4.030},{2.0,3.925},{5.0,3.769},
        {10.0,3.697},{20.0,3.670},{50.0,3.660}
    };
    for (const auto &row : values) {
        auto s = graetz(25.0); s.setup_fp_solution(row[0]);
        REQUIRE_APPROX(s.get_fully_developed_nusselt_number(),row[1],1e-3,0.0);
    }
}

TEST_CASE(nusselt_scaling_and_radial_projection)
{
    auto normalized = graetz(); normalized.setup_bare_solution();
    CDGraetzIsothermalSolution<double> scaled(4.0,2.0,0);
    scaled.set_max_root(25.0); scaled.setup_bare_solution();
    for (double x : {0.05,0.2,1.0})
        REQUIRE_APPROX(scaled.get_nusselt_number(x),normalized.get_nusselt_number(x),1e-12,0.0);

    auto radial = graetz();
    radial.set_rhs_method(RhsMethod::DirectQuadrature);
    radial.set_projection_space(ProjectionSpace::Radial);
    radial.setup_fp_solution(5.0);
    REQUIRE(std::isfinite(radial.get_nusselt_number(0.2)));
    REQUIRE(std::isfinite(radial.get_fully_developed_nusselt_number()));
}

// Golden files written by tests/nusselt_golden_dump.h compiled against commit
// f36fa63, the tree before any Nusselt code. The solution path must reproduce
// them bitwise.
TEST_CASE(solution_values_unchanged_by_this_change)
{
    const auto check = [](const std::string &name, const std::string &current) {
        std::ifstream stream(std::string(TEST_DATA_DIR) + "/tests/golden/" + name);
        REQUIRE(stream.good());
        std::ostringstream golden;
        golden << stream.rdbuf();
        if (golden.str() != current) std::printf("  golden mismatch: %s\n", name.c_str());
        REQUIRE(golden.str() == current);
    };
    for (const auto &c : nusselt_golden::cases()) {
        check(nusselt_golden::file_name<double>(c), nusselt_golden::dump<double>(c));
        check(nusselt_golden::file_name<long double>(c), nusselt_golden::dump<long double>(c));
    }
}

TEST_CASE(nusselt_split_api_contract)
{
    auto bare = graetz(); bare.setup_bare_solution();
    REQUIRE(throws_logic_containing([&] { bare.get_nusselt_number(0.1, NusseltEvaluation::SpectralSplit); }, "finite-Peclet"));
    REQUIRE(throws_logic_containing([&] { bare.get_nusselt_number(0.1, NusseltEvaluation::SlugSplit); }, "finite-Peclet"));
    REQUIRE(throws_logic_containing([&] { bare.get_nusselt_tail_rates(0.1); }, "finite-Peclet"));
    REQUIRE(bare.get_nusselt_number(std::vector<double>{0.1}, NusseltEvaluation::Plain)
            == bare.get_nusselt_number(std::vector<double>{0.1}));

    auto s = graetz(); s.setup_fp_solution(5.0);
    const auto invalid = [&](auto f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    REQUIRE(invalid([&] { s.get_nusselt_number(0.0, NusseltEvaluation::SpectralSplit); }));
    REQUIRE(invalid([&] { s.get_nusselt_number(0.0, NusseltEvaluation::SlugSplit); }));
    REQUIRE(invalid([&] { s.nusselt_converged_x_min(0.0); }));
    REQUIRE(invalid([&] { s.nusselt_converged_x_min(1.0); }));
    const std::vector<double> xs{0.02, 0.1, std::numeric_limits<double>::infinity()};
    const auto rates = s.get_nusselt_tail_rates(0.02);
    REQUIRE(invalid([&] { s.get_nusselt_number(xs, NusseltEvaluation::Plain, &rates); }));
    REQUIRE(invalid([&] { s.get_nusselt_number(xs, NusseltEvaluation::SlugSplit, &rates); }));
    const std::vector<double> short_rates(rates.begin(), rates.begin() + 5);
    REQUIRE(invalid([&] { s.get_nusselt_number(xs, NusseltEvaluation::SpectralSplit, &short_rates); }));
    REQUIRE(s.get_nusselt_number(xs, NusseltEvaluation::SpectralSplit, &rates)
            == s.get_nusselt_number(xs, NusseltEvaluation::SpectralSplit));

    const auto split = s.get_nusselt_number(xs, NusseltEvaluation::SpectralSplit);
    const auto plain = s.get_nusselt_number(xs);
    REQUIRE(split[2] == plain[2]);
    REQUIRE(s.get_nusselt_number(xs, NusseltEvaluation::SlugSplit)[2] == plain[2]);
    REQUIRE_APPROX(split[2], s.get_fully_developed_nusselt_number(), 1e-13, 0.0);
    for (std::size_t i = 0; i < xs.size(); ++i)
        // A lone x sizes its own tail, so agreement is to the 16 eps tail tolerance.
        REQUIRE_APPROX(s.get_nusselt_number(xs[i], NusseltEvaluation::SpectralSplit), split[i], 1e-13, 0.0);

    const auto modes = s.get_nusselt_mode_data();
    REQUIRE_APPROX(s.nusselt_converged_x_min(1e-8), std::log(1e8) / modes.back()[1], 1e-14, 0.0);
}

// Plan B10, run on the Radial projection (see graetz_radial) and with SlugSplit
// at x <= 1e-3: the exact rates of SpectralSplit cost minutes there at Pe = 5
// (boost 1F1 slows down with |a| ~ Lam^{3/2}/(4 Pe^2)). Slug and spectral tails
// are compared directly in nusselt_slug_split_matches_spectral_split.
TEST_CASE(nusselt_split_is_K_independent)
{
    const double pe = 5.0;
    const std::vector<double> xs{1e-2, 1e-3, 1e-4};
    auto coarse = graetz_radial(200.0, pe);
    auto fine = graetz_radial(400.0, pe);
    const auto split_coarse = coarse.get_nusselt_number(xs, NusseltEvaluation::SlugSplit);
    const auto split_fine = fine.get_nusselt_number(xs, NusseltEvaluation::SlugSplit);
    const auto plain_coarse = coarse.get_nusselt_number(xs);
    const auto plain_fine = fine.get_nusselt_number(xs);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        std::printf("  x=%g plain %.6f -> %.6f, split %.6f -> %.6f\n",
                    xs[i], plain_coarse[i], plain_fine[i], split_coarse[i], split_fine[i]);
        REQUIRE_APPROX(split_coarse[i], split_fine[i], 1e-3, 0.0);
    }
    REQUIRE(std::fabs(plain_fine[2] / plain_coarse[2] - 1.0) > 0.5);

    const auto modes = fine.get_nusselt_mode_data();
    std::vector<double> delta;
    for (const auto &v : modes) delta.push_back(v[2] * v[3] + 2.0 * sqrt_two_pi);
    std::printf("  delta_k:");
    for (std::size_t k = 0; k < delta.size(); k += 9) std::printf(" %.2e", delta[k]);
    std::printf(" ... %.2e\n", delta.back());
    for (std::size_t k = 1; k < 10; ++k) REQUIRE(std::fabs(delta[k]) < std::fabs(delta[k - 1]));
    for (std::size_t k = delta.size() / 2; k < delta.size(); ++k)
        REQUIRE(std::fabs(delta[k]) < 0.02 * 2.0 * sqrt_two_pi);
}

TEST_CASE(nusselt_split_matches_plain_where_converged)
{
    for (double pe : {5.0, 50.0}) {
        auto s = graetz(200.0); s.setup_fp_solution(pe);
        const double x_c = s.nusselt_converged_x_min(1e-8);
        const std::vector<double> xs{x_c, 2.0 * x_c, 0.1, 1.0};
        const auto plain = s.get_nusselt_number(xs);
        const auto spectral = s.get_nusselt_number(xs, NusseltEvaluation::SpectralSplit);
        const auto slug = s.get_nusselt_number(xs, NusseltEvaluation::SlugSplit);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::printf("  Pe=%g x=%g |spectral/plain-1|=%.2e |slug/plain-1|=%.2e\n", pe, xs[i],
                        std::fabs(spectral[i] / plain[i] - 1.0), std::fabs(slug[i] / plain[i] - 1.0));
            REQUIRE_APPROX(spectral[i], plain[i], 1e-8, 0.0);
            REQUIRE_APPROX(slug[i], plain[i], 1e-8, 0.0);
        }
    }
}

// Where the exact finite-Peclet tail is affordable, the slug-flow tail must give
// the same Nusselt number: the two differ only through the rates beyond the block,
// by about x (Lam - Lam^s) per tail term; that mismatch is still ~8 at the end of
// the Pe = 50 block (it decays like 1/m), hence 1e-5 there.
TEST_CASE(nusselt_slug_split_matches_spectral_split)
{
    const struct { double pe; std::vector<double> xs; } cases[] = {{5.0, {1e-2, 2e-2}}, {50.0, {1e-3, 2e-3}}};
    for (const auto &c : cases) {
        auto s = graetz_radial(200.0, c.pe);
        const auto spectral = s.get_nusselt_number(c.xs, NusseltEvaluation::SpectralSplit);
        const auto slug = s.get_nusselt_number(c.xs, NusseltEvaluation::SlugSplit);
        for (std::size_t i = 0; i < c.xs.size(); ++i) {
            std::printf("  Pe=%g x=%g spectral %.10f slug %.10f\n", c.pe, c.xs[i], spectral[i], slug[i]);
            REQUIRE_APPROX(slug[i], spectral[i], 1e-5, 0.0);
        }
    }
}

// G_inf of eq. G_inf_split_asymptotic with its sum taken over all the terms a
// direct sum of e^{-Lam^inf_m x} needs, so that only the geometric closed form
// separates the two. Pe = 50 has m0 = 5 non-positive Lam^a_m, whose exponentials
// would overflow at x = 1 without the regrouping.
TEST_CASE(spectral_function_g_inf_closed_form)
{
    for (double pe : {5.0, 50.0}) {
        const double kappa = 1.0 / (pe * pe);
        const auto rates = bessel_mode_rates(bessel_j0_zeros<double>(1u, 2000u), kappa);
        for (double x : {1e-2, 1.0}) {
            double direct = 0.0;
            for (std::size_t m = rates.size(); m-- > 0;) direct += std::exp(-rates[m] * x);
            REQUIRE(std::exp(-rates.back() * x) < 1e-30);
            REQUIRE_APPROX(spectral_function_g_inf(rates, kappa, x, 0.0), direct, 1e-13, 0.0);
        }
    }
    REQUIRE(asymptotic_progression_non_positive_terms(1.0 / 25.0) == 0u);
    REQUIRE(asymptotic_progression_non_positive_terms(1.0 / 2500.0) == 5u);
    REQUIRE(asymptotic_progression_rate(6u, 1.0 / 2500.0) > 0.0);
    REQUIRE(asymptotic_progression_rate(5u, 1.0 / 2500.0) < 0.0);
}

// G = G_0 + G_inf (eqs. G_split and G_inf_split_asymptotic). With a long G_inf sum
// it is the slug-flow tail again (the rates differ only through <omega>_m =
// (2/3)(1 + 1/q^2) against 2/3); with the default M = 200 the dropped part of the
// G_inf sum is below 1e-6 at Pe = 5 and grows with Pe.
TEST_CASE(nusselt_asymptotic_split)
{
    const auto invalid = [](auto f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    auto bare = graetz(); bare.setup_bare_solution();
    REQUIRE(throws_logic_containing([&] { bare.get_nusselt_number(0.1, NusseltEvaluation::AsymptoticSplit); },
                                    "finite-Peclet"));
    auto s = graetz(); s.setup_fp_solution(5.0);
    REQUIRE(s.get_nusselt_asymptotic_terms() == 200u);
    REQUIRE(invalid([&] { s.set_nusselt_asymptotic_terms(0u); }));
    REQUIRE(invalid([&] { s.get_nusselt_number(0.0, NusseltEvaluation::AsymptoticSplit); }));
    const auto rates = s.get_nusselt_tail_rates(0.1);
    REQUIRE(invalid([&] { s.get_nusselt_number(std::vector<double>{0.1}, NusseltEvaluation::AsymptoticSplit, &rates); }));
    const std::vector<double> far{1.0, std::numeric_limits<double>::infinity()};
    REQUIRE(s.get_nusselt_number(far, NusseltEvaluation::AsymptoticSplit) == s.get_nusselt_number(far));

    const struct { double pe; double tol_200; } cases[] = {{5.0, 1e-6}, {50.0, 1e-4}};
    const std::vector<double> xs{1e-4, 1e-3, 1e-2};
    for (const auto &c : cases) {
        auto t = graetz_radial(200.0, c.pe);
        const auto slug = t.get_nusselt_number(xs, NusseltEvaluation::SlugSplit);
        const auto short_sum = t.get_nusselt_number(xs, NusseltEvaluation::AsymptoticSplit);
        t.set_nusselt_asymptotic_terms(20000u);
        const auto long_sum = t.get_nusselt_number(xs, NusseltEvaluation::AsymptoticSplit);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::printf("  Pe=%g x=%g slug %.10f asymptotic M=200 %.10f M=20000 %.10f\n",
                        c.pe, xs[i], slug[i], short_sum[i], long_sum[i]);
            REQUIRE_APPROX(long_sum[i], slug[i], 1e-6, 0.0);
            REQUIRE_APPROX(short_sum[i], slug[i], c.tol_200, 0.0);
        }
    }
}

TEST_CASE(spectral_function_bessel_closed_form)
{
    for (double xi : {1e-3, 1e-2, 1e-1, 1.0}) {
        const unsigned count = static_cast<unsigned>(std::ceil(45.0 / (M_PI * xi))) + 10u;
        const auto zeros = bessel_j0_zeros<double>(1u, count);
        REQUIRE(std::exp(-zeros.back() * xi) < 1e-18);
        REQUIRE_APPROX(spectral_function_sum(zeros, xi), bessel_spectral_function_closed_form(xi), 1e-8, 0.0);
    }
    REQUIRE_APPROX(bessel_spectral_function_closed_form<long double>(0.1L),
                   static_cast<long double>(bessel_spectral_function_closed_form(0.1)), 1e-12L, 0.0L);
}

TEST_CASE(rates_beyond_retained_block_are_exact)
{
    for (double pe : {5.0, 50.0}) {
        auto s = graetz(400.0); s.setup_fp_solution(pe);
        const double kappa = 1.0 / (pe * pe);
        std::vector<double> retained;
        for (const auto &row : s.get_modified_data()) retained.push_back(row[0]);
        const std::size_t K = retained.size();
        const auto again = fp_dirichlet_rates_beyond<double>(kappa, retained[K - 12], retained[K - 11], 10u, 1e-13, 200u);
        for (unsigned i = 0; i < 10u; ++i)
            REQUIRE_APPROX(again[i], retained[K - 10 + i], 1e-11, 0.0);

        const auto beyond = fp_dirichlet_rates_beyond<double>(kappa, retained[K - 2], retained[K - 1], 200u, 1e-13, 200u);
        const double pi_pe = M_PI * pe;
        std::vector<double> spacing{beyond[0] - retained[K - 1]};
        for (std::size_t i = 0; i < beyond.size(); ++i) {
            const double below = fp_char<double>(0u, beyond[i] * (1.0 - 1e-9), kappa, WallCondition::Dirichlet);
            const double above = fp_char<double>(0u, beyond[i] * (1.0 + 1e-9), kappa, WallCondition::Dirichlet);
            REQUIRE((below > 0.0) != (above > 0.0));
            if (i > 0) spacing.push_back(beyond[i] - beyond[i - 1]);
        }
        // Consecutive rates are never closer than the last retained spacing and
        // approach pi Pe from below (the plan expected them from above).
        std::printf("  Pe=%g spacing %.6f -> %.6f (pi Pe = %.6f)\n", pe, spacing.front(), spacing.back(), pi_pe);
        for (std::size_t i = 1; i < spacing.size(); ++i) {
            REQUIRE(spacing[i] >= spacing[i - 1] * (1.0 - 1e-9));
            REQUIRE(spacing[i] < pi_pe * (1.0 + 1e-9));
        }
        REQUIRE(pi_pe - spacing.back() < pi_pe - spacing.front());
    }
}

// Quarter-plane corner law of the semi-infinite problem, Nu ~ 4/(pi Pe x): the
// wall gradient -2 sqrt(2 pi) G(x), G ~ 1/(pi Pe x), over the inlet bulk
// integral (pi/2)/sqrt(2 pi).
TEST_CASE(nusselt_corner_law_finite_peclet)
{
    const double pe = 5.0;
    auto s = graetz_radial(400.0, pe);
    const std::vector<double> xs{1e-3, 3e-4, 1e-4};
    const auto nu = s.get_nusselt_number(xs, NusseltEvaluation::SlugSplit);
    double previous = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double deviation = std::fabs(nu[i] * xs[i] * M_PI * pe / 4.0 - 1.0);
        std::printf("  x=%g Nu x pi Pe/4 - 1 = %.2e\n", xs[i], nu[i] * xs[i] * M_PI * pe / 4.0 - 1.0);
        REQUIRE(deviation < 0.02);
        REQUIRE(deviation < previous);
        previous = deviation;
    }
}

// Lahjomri and Oubarra (1999), Fig. 2, curve I (Pe = 5): 3.7673 at X/Pe = x = 1,
// the value quoted from that figure in theory/analytical_solution.tex Sec. 7.
TEST_CASE(nusselt_lahjomri_figures)
{
    auto s = graetz(25.0); s.setup_fp_solution(5.0);
    REQUIRE_APPROX(s.get_nusselt_number(1.0), 3.7673, 1e-3, 0.0);
    REQUIRE_APPROX(s.get_fully_developed_nusselt_number(), 3.7673, 1e-3, 0.0);
}
