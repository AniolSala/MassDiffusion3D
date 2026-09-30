// Tests for the L2_r finite-Peclet inlet projection
// (plans/PLAN_L2R_PROJECTION.md): projection_space.h, finite_peclet_gram_radial.*,
// finite_peclet_rhs_radial.*, finite_peclet_coefficients_radial.* and the
// CDBaseSolution selector that reaches them.
//
// Every reference below is INDEPENDENT code: composite Simpson / Gauss-Legendre
// quadratures and an independent characteristic-root scan, sharing nothing with
// the modules under test beyond the radial primitives of finite_peclet_radial.h
// (themselves pinned by tests/test_psinm_r.cpp).
//
// Two facts worth stating up front, because they look like bugs and are not:
//  * The two projections return DIFFERENT coefficient vectors for the same
//    inlet -- each is the best approximation from the same span in its own
//    norm. Nothing here compares them entry by entry; T7 compares invariants.
//  * U^n does NOT become diagonal as kappa -> 0. The bare modes are orthogonal
//    in L2_omega, not in L2_r, so U tends to the (non-diagonal) unweighted
//    overlap matrix. The right kappa -> 0 cross-check is T4's pencil identity.

#include "tinytest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
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
#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_rhs_radial.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_coefficients_radial.h"
#include "finite_peclet_radial.h"
#include "finite_peclet_roots.h"
#include "gram_method.h"
#include "projection_space.h"
#include "rhs_method.h"

namespace
{
constexpr double kSqrtTwoPi = 2.50662827463100050242;
constexpr double kInvSqrtTwoPi = 0.39894228040143267794;
constexpr double kInvSqrtPi = 0.56418958354775628695;

// ─── independent quadrature primitives ───────────────────────────────────────

// Gauss-Legendre rule on [-1,1] by Newton iteration on the Legendre polynomial.
void gauss_legendre(unsigned N, std::vector<double> &x, std::vector<double> &w)
{
    x.assign(N, 0.0);
    w.assign(N, 0.0);
    for (unsigned i = 0; i < N; ++i)
    {
        double z = std::cos(M_PI * (i + 0.75) / (N + 0.5)), dp = 0.0;
        for (int it = 0; it < 100; ++it)
        {
            double p0 = 1.0, p1 = 0.0;
            for (unsigned j = 0; j < N; ++j)
            {
                const double p2 = p1;
                p1 = p0;
                p0 = ((2.0 * j + 1.0) * z * p1 - j * p2) / (j + 1.0);
            }
            dp = N * (z * p0 - p1) / (z * z - 1.0);
            const double dz = -p0 / dp;
            z += dz;
            if (std::abs(dz) < 1e-16)
                break;
        }
        x[N - 1 - i] = z;
        w[N - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

// Composite Simpson of g over [a,b] with `panels` (even) subintervals.
template <class G>
double simpson(double a, double b, unsigned panels, G &&g)
{
    if (panels % 2)
        ++panels;
    const double h = (b - a) / panels;
    double sum = 0.0;
    for (unsigned q = 0; q <= panels; ++q)
    {
        const double weight = (q == 0 || q == panels) ? 1.0 : (q % 2 == 0 ? 2.0 : 4.0);
        sum += weight * g(a + q * h);
    }
    return sum * h / 3.0;
}

// ─── independent Gram / norm references (s = r^2 form) ───────────────────────
//
// The references below are the composite-Simpson brute force of
// tests/test_fp_gram_backends.cpp's reference_gram_entry, with (and, for the
// L2_r case, without) its (1-s) factor. They are assembled from a SINGLE
// sampling of every mode on the Simpson grid rather than one sampling per
// matrix entry: K*(panels+1) radial evaluations instead of K^2*(panels+1),
// which is the difference between a 10-second and a 3-minute test. The
// arithmetic that follows is unchanged, so this is the same reference.
constexpr unsigned kReferencePanels = 20000u;

struct SimpsonSamples
{
    std::vector<double> s, base_weight;                // base_weight = simpson coeff * h/3 * s^n
    std::vector<std::vector<double>> radial_factor;    // [mode][node] = G_m(s)
};

SimpsonSamples sample_modes_on_simpson_grid(unsigned n, const std::vector<SeriesTermData<double>> &modes)
{
    const double h = 1.0 / kReferencePanels;
    SimpsonSamples out;
    out.s.resize(kReferencePanels + 1);
    out.base_weight.resize(kReferencePanels + 1);
    for (unsigned q = 0; q <= kReferencePanels; ++q)
    {
        const double s = q * h;
        const double coefficient = (q == 0 || q == kReferencePanels) ? 1.0 : (q % 2 == 0 ? 2.0 : 4.0);
        out.s[q] = s;
        out.base_weight[q] = coefficient * (h / 3.0) * std::pow(s, static_cast<double>(n));
    }
    out.radial_factor.assign(modes.size(), std::vector<double>(kReferencePanels + 1, 0.0));
    for (std::size_t m = 0; m < modes.size(); ++m)
        for (unsigned q = 0; q <= kReferencePanels; ++q)
            out.radial_factor[m][q] = fp_radial_factor(n, modes[m].root_fp, modes[m].btilde_fp, out.s[q]);
    return out;
}

// U^n_{mk} = 1/2 int_0^1 s^n G_m G_k ds  (include_omega = false),
// W^n_{mk} = 1/2 int_0^1 (1-s) s^n G_m G_k ds  (include_omega = true).
std::vector<std::vector<double>> reference_gram_matrix(unsigned n, const std::vector<SeriesTermData<double>> &modes,
                                                       bool include_omega)
{
    const auto samples = sample_modes_on_simpson_grid(n, modes);
    const std::size_t K = modes.size();
    std::vector<std::vector<double>> g(K, std::vector<double>(K, 0.0));
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (unsigned q = 0; q <= kReferencePanels; ++q)
            {
                const double weight = include_omega ? samples.base_weight[q] * (1.0 - samples.s[q])
                                                    : samples.base_weight[q];
                sum += weight * samples.radial_factor[i][q] * samples.radial_factor[j][q];
            }
            g[i][j] = g[j][i] = 0.5 * sum;
        }
    return g;
}

// N^n_mm = int_0^1 [omega + 2 kappa Lambda_nm] R_nm^2 r dr, the diagonal of T4's pencil.
std::vector<double> reference_generalised_norms(unsigned n, double kappa, const std::vector<SeriesTermData<double>> &modes)
{
    const auto samples = sample_modes_on_simpson_grid(n, modes);
    std::vector<double> out(modes.size(), 0.0);
    for (std::size_t m = 0; m < modes.size(); ++m)
    {
        double sum = 0.0;
        for (unsigned q = 0; q <= kReferencePanels; ++q)
            sum += samples.base_weight[q] * ((1.0 - samples.s[q]) + 2.0 * kappa * modes[m].rate_fp)
                 * samples.radial_factor[m][q] * samples.radial_factor[m][q];
        out[m] = 0.5 * sum;
    }
    return out;
}

// ─── independent finite-Peclet eigenvalues ───────────────────────────────────
//
// Scans the characteristic function of the modified radial ODE for sign changes
// and bisects. Independent of fp_solve_ladder and of the bare-root catalogue,
// which is what lets T4 cover the DIRICHLET wall at n > 0 -- the repo ships a
// Dirichlet root table for n = 0 only (the Graetz basis).
double characteristic(unsigned n, double lambda, double kappa, WallCondition wall)
{
    const double b = std::sqrt(lambda), bt = b * (1.0 + kappa * lambda);
    return wall == WallCondition::Dirichlet ? psi_at_1_fp(n, b, bt) : dpsi_dr_at_1_fp(n, b, bt);
}

std::vector<SeriesTermData<double>> independent_modes(unsigned n, unsigned K, double kappa, WallCondition wall)
{
    std::vector<double> rates;
    const bool has_zero_mode = (wall == WallCondition::Neumann && n == 0);
    if (has_zero_mode)
        rates.push_back(0.0); // R == const satisfies R'(1) = 0 exactly
    // Start past the zero mode's neighbourhood when it exists, so its own root
    // is not re-detected by the sign scan.
    double previous = has_zero_mode ? 1.0 : 1e-6;
    double f_previous = characteristic(n, previous, kappa, wall);
    while (rates.size() < K && previous < 1e8)
    {
        const double step = 0.05 * (1.0 + std::sqrt(previous));
        const double current = previous + step, f_current = characteristic(n, current, kappa, wall);
        if ((f_previous < 0.0) != (f_current < 0.0))
        {
            double lo = previous, hi = current, f_lo = f_previous;
            for (int it = 0; it < 200; ++it)
            {
                const double mid = 0.5 * (lo + hi), f_mid = characteristic(n, mid, kappa, wall);
                if ((f_lo < 0.0) != (f_mid < 0.0))
                    hi = mid;
                else
                {
                    lo = mid;
                    f_lo = f_mid;
                }
            }
            rates.push_back(0.5 * (lo + hi));
        }
        previous = current;
        f_previous = f_current;
    }
    if (rates.size() < K)
        throw std::runtime_error("independent_modes: root scan did not find enough eigenvalues");

    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        modes[m].n = n;
        modes[m].m = m;
        modes[m].rate_fp = rates[m];
        modes[m].root_fp = std::sqrt(rates[m]);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * rates[m]);
        // Bare root, used only to size the RHS cap quadrature. The exact
        // relation kappa*Lambda^2 + Lambda = beta^2 holds only for the seed
        // ratio alpha = 1; sqrt(Lambda(1+kappa*Lambda)) is the matching
        // conservative estimate, which is all fp_rhs_required_order needs.
        modes[m].root = std::sqrt(rates[m] * (1.0 + kappa * rates[m]));
    }
    return modes;
}

// ─── dense linear algebra for the assertions ────────────────────────────────

// Cyclic Jacobi eigenvalues of a symmetric matrix, ascending.
std::vector<double> symmetric_eigenvalues(std::vector<std::vector<double>> a)
{
    const std::size_t K = a.size();
    for (int sweep = 0; sweep < 100; ++sweep)
    {
        double off = 0.0;
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = i + 1; j < K; ++j)
                off += a[i][j] * a[i][j];
        if (off < 1e-30)
            break;
        for (std::size_t p = 0; p < K; ++p)
            for (std::size_t q = p + 1; q < K; ++q)
            {
                if (std::abs(a[p][q]) < 1e-300)
                    continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (std::size_t k = 0; k < K; ++k)
                {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < K; ++k)
                {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
    }
    std::vector<double> eig(K);
    for (std::size_t i = 0; i < K; ++i)
        eig[i] = a[i][i];
    std::sort(eig.begin(), eig.end());
    return eig;
}

// Returns false when the matrix is not positive definite.
bool cholesky_succeeds(const std::vector<std::vector<double>> &a)
{
    const std::size_t K = a.size();
    std::vector<std::vector<double>> L(K, std::vector<double>(K, 0.0));
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            double v = a[i][j];
            for (std::size_t k = 0; k < j; ++k)
                v -= L[i][k] * L[j][k];
            if (i == j)
            {
                if (!(v > 0.0))
                    return false;
                L[i][j] = std::sqrt(v);
            }
            else
                L[i][j] = v / L[j][j];
        }
    return true;
}

std::vector<std::vector<double>> normalise_to_unit_diagonal(const std::vector<std::vector<double>> &a)
{
    const std::size_t K = a.size();
    std::vector<std::vector<double>> out(K, std::vector<double>(K, 0.0));
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            out[i][j] = a[i][j] / std::sqrt(a[i][i] * a[j][j]);
    return out;
}

// ─── independent 2-D disk quadrature ────────────────────────────────────────
//
// The inlet profile is piecewise constant in z = r cos(phi), so its jumps are
// straight chords -- fatal for a naive (r, phi) tensor rule. Integrate in the
// Cartesian element dz dy == r dr dphi instead, with z = cos(theta) (so the
// half-width sqrt(1-z^2) = sin(theta) is smooth) and y = sin(theta) * t:
//
//   int_disk g = int_0^pi sin^2(theta) int_{-1}^{1} g(r, phi) dt dtheta ,
//   r = sqrt(cos^2 theta + sin^2 theta t^2) ,  phi = atan2(sin(theta) t, cos(theta)) .
//
// Splitting theta at arccos(z_i) puts every discontinuity on a panel boundary,
// so Gauss-Legendre converges spectrally on each panel.
struct DiskRule
{
    std::vector<double> theta, theta_weight, t, t_weight;
};

DiskRule make_disk_rule(const std::vector<double> &interfaces, unsigned nodes_theta, unsigned nodes_t)
{
    std::vector<double> breaks{0.0, M_PI};
    for (double z : interfaces)
        if (z > -1.0 && z < 1.0)
            breaks.push_back(std::acos(z));
    std::sort(breaks.begin(), breaks.end());

    std::vector<double> gx, gw;
    gauss_legendre(nodes_theta, gx, gw);
    DiskRule rule;
    for (std::size_t seg = 0; seg + 1 < breaks.size(); ++seg)
    {
        const double a = breaks[seg], b = breaks[seg + 1];
        if (!(b > a))
            continue;
        const double half = 0.5 * (b - a), mid = 0.5 * (a + b);
        for (unsigned q = 0; q < nodes_theta; ++q)
        {
            const double th = mid + half * gx[q];
            rule.theta.push_back(th);
            rule.theta_weight.push_back(gw[q] * half * std::sin(th) * std::sin(th));
        }
    }
    gauss_legendre(nodes_t, rule.t, rule.t_weight);
    return rule;
}

// g(r, phi, z) integrated over the unit disk with the element r dr dphi.
template <class G>
double disk_integral(const DiskRule &rule, G &&g)
{
    double total = 0.0;
    for (std::size_t i = 0; i < rule.theta.size(); ++i)
    {
        const double z = std::cos(rule.theta[i]), half_width = std::sin(rule.theta[i]);
        double inner = 0.0;
        for (std::size_t j = 0; j < rule.t.size(); ++j)
        {
            const double y = half_width * rule.t[j];
            const double r = std::sqrt(z * z + y * y);
            inner += rule.t_weight[j] * g(r, std::atan2(y, z), z);
        }
        total += rule.theta_weight[i] * inner;
    }
    return total;
}

double angular_basis(unsigned n, double phi)
{
    return n == 0u ? kInvSqrtTwoPi : kInvSqrtPi * std::cos(n * phi);
}

// Piecewise-constant inlet value at height z: layer i on (z_{i-1}, z_i].
double layer_value(const std::vector<double> &interfaces, const std::vector<double> &values, double z)
{
    std::size_t i = 0;
    while (i < interfaces.size() && z > interfaces[i])
        ++i;
    return values[i];
}

// ─── small conveniences ──────────────────────────────────────────────────────

std::vector<SeriesTermData<double>> angular_block(const std::vector<SeriesTermData<double>> &data, unsigned n)
{
    std::vector<SeriesTermData<double>> modes;
    for (const auto &t : data)
        if (t.n == n)
            modes.push_back(t);
    return modes;
}

std::vector<std::vector<double>> radial_gram(unsigned n, const std::vector<SeriesTermData<double>> &modes)
{
    const auto factor = fp_gram_factor_radial(n, modes, FPGramRadialOptions<double>{});
    std::vector<std::vector<double>> U;
    fp_gram_reconstruct_radial(factor, U);
    return U;
}

std::vector<std::vector<double>> weighted_gram(unsigned n, const std::vector<SeriesTermData<double>> &modes)
{
    const auto factor = fp_gram_factor_gauss_jacobi(n, modes, FPGramGaussJacobiOptions<double>{});
    std::vector<std::vector<double>> W;
    fp_gram_reconstruct_gauss_jacobi(factor, W);
    return W;
}

// Extracts the backend's own (alpha=0, beta=n) rule of order N: a one-mode block
// with retain_samples stores its quadrature nodes and weights verbatim. Nothing
// else in the public API exposes them, and re-deriving the recurrence here would
// test a copy rather than the shipped rule.
void radial_rule_of_order(unsigned n, unsigned N, std::vector<double> &x, std::vector<double> &w)
{
    std::vector<SeriesTermData<double>> modes(1);
    modes[0].n = n;
    modes[0].m = 0;
    modes[0].root_fp = 0.0;   // G == 1: a well-conditioned single column
    modes[0].btilde_fp = 0.0;
    modes[0].rate_fp = 1.0;   // non-zero, so the constant-mode branch stays out of the way
    FPGramRadialOptions<double> o;
    o.oversampling_factor = 1u;
    o.oversampling_margin = N - 1u;
    o.minimum_factor = 1u;
    o.retain_samples = true;
    const auto factor = fp_gram_factor_radial(n, modes, o);
    if (factor.node_count != N)
        throw std::runtime_error("radial_rule_of_order: unexpected node count");
    x = factor.quadrature_nodes;
    w = factor.quadrature_weights;
}
} // namespace

// ═══ T1 — the new quadrature rule ════════════════════════════════════════════

TEST_CASE(fp_radial_gram_rule_is_gauss_jacobi_alpha0_beta_n)
{
    for (unsigned n : {0u, 1u, 2u, 5u, 20u})
        for (unsigned N : {5u, 20u, 60u})
        {
            std::vector<double> x, w;
            radial_rule_of_order(n, N, x, w);
            REQUIRE(x.size() == N);
            REQUIRE(w.size() == N);

            double mass = 0.0;
            for (unsigned q = 0; q < N; ++q)
            {
                REQUIRE(x[q] > 0.0 && x[q] < 1.0);
                REQUIRE(w[q] > 0.0);
                if (q)
                    REQUIRE(x[q] > x[q - 1]);
                mass += w[q];
            }
            // Total mass of s^n ds on [0,1]. (n = 0 is the shifted-Legendre
            // special case of the recurrence -- the 0/0 diagonal at k = 0.)
            const double expected_mass = 1.0 / (n + 1.0);
            REQUIRE(std::abs(mass - expected_mass) / expected_mass < 1e-14);

            // Exactness: sum_q w_q s_q^p = int_0^1 s^{n+p} ds for p < 2N.
            for (unsigned p = 0; p < 2 * N; ++p)
            {
                double moment = 0.0;
                for (unsigned q = 0; q < N; ++q)
                    moment += w[q] * std::pow(x[q], static_cast<double>(p));
                const double exact = 1.0 / (n + p + 1.0);
                REQUIRE(std::abs(moment - exact) / exact < 1e-13);
            }
        }
}

// ═══ T2 — Gram entries against an independent brute force ════════════════════

TEST_CASE(fp_radial_gram_matches_brute_force_reference)
{
    const std::vector<std::pair<unsigned, unsigned>> cases{{0u, 8u}, {3u, 8u}, {20u, 12u}};
    for (double kappa : {1e-4, 1e-2, 1.0})
        for (auto [n, K] : cases)
        {
            const auto modes = independent_modes(n, K, kappa, WallCondition::Neumann);
            const auto U = radial_gram(n, modes);
            const auto reference = reference_gram_matrix(n, modes, false);
            for (unsigned i = 0; i < K; ++i)
                for (unsigned j = 0; j < K; ++j)
                {
                    // Cauchy-Schwarz scale: the meaningful relative measure for
                    // a Gram matrix whose diagonal spans several decades.
                    const double scale = std::sqrt(std::abs(U[i][i] * U[j][j]));
                    REQUIRE(std::abs(U[i][j] - reference[i][j]) / scale < 1e-10);
                }
        }
}

// ═══ T3 — symmetry, positive definiteness, and the solve ═════════════════════

TEST_CASE(fp_radial_gram_is_symmetric_positive_definite)
{
    for (double kappa : {1e-3, 1e-1})
        for (unsigned n : {0u, 2u})
        {
            const unsigned K = 8u;
            const auto modes = independent_modes(n, K, kappa, WallCondition::Neumann);
            const auto U = radial_gram(n, modes);

            for (unsigned i = 0; i < K; ++i)
                for (unsigned j = 0; j < K; ++j)
                {
                    const double scale = std::sqrt(std::abs(U[i][i] * U[j][j]));
                    REQUIRE(std::abs(U[i][j] - U[j][i]) / scale < 1e-14);
                }

            REQUIRE(cholesky_succeeds(U));
            const auto eig = symmetric_eigenvalues(normalise_to_unit_diagonal(U));
            REQUIRE(eig.front() > 0.0);

            // U x = U x0 must return x0.
            std::vector<double> x0(K);
            for (unsigned i = 0; i < K; ++i)
                x0[i] = 1.0 + 0.25 * i - 0.05 * i * i;
            std::vector<double> b(K, 0.0);
            for (unsigned i = 0; i < K; ++i)
                for (unsigned j = 0; j < K; ++j)
                    b[i] += U[i][j] * x0[j];
            const auto factor = fp_gram_factor_radial(n, modes, FPGramRadialOptions<double>{});
            std::vector<double> c;
            fp_gram_solve_radial(factor, b, c);
            REQUIRE(c.size() == K);
            for (unsigned i = 0; i < K; ++i)
                REQUIRE(std::abs(c[i] - x0[i]) / std::abs(x0[i]) < 1e-10);
        }
}

// ═══ T4 — the pencil identity (the strongest cross-check) ════════════════════
//
//     W^n + kappa (Lambda U^n + U^n Lambda) = diag(N^n_mm)
//
// It follows from Green's identity for the modified radial ODE: the boundary
// term r (R_k R_m' - R_m R_k') vanishes at r = 1 under EITHER wall condition and
// at r = 0 by regularity, leaving
//     W_mk + kappa (Lambda_m + Lambda_k) U_mk = 0   for Lambda_m != Lambda_k.
// It ties the new matrix to the existing one through theory alone, and a wrong
// Jacobi alpha in the new rule cannot pass it.

namespace
{
void check_pencil_identity(unsigned n, double kappa, WallCondition wall)
{
    const unsigned K = 8u;
    const auto modes = independent_modes(n, K, kappa, wall);
    const auto U = radial_gram(n, modes);
    const auto W = weighted_gram(n, modes);

    std::vector<std::vector<double>> pencil(K, std::vector<double>(K, 0.0));
    double diagonal_scale = 0.0;
    for (unsigned i = 0; i < K; ++i)
        for (unsigned j = 0; j < K; ++j)
        {
            pencil[i][j] = W[i][j] + kappa * (modes[i].rate_fp + modes[j].rate_fp) * U[i][j];
            if (i == j)
                diagonal_scale = std::max(diagonal_scale, std::abs(pencil[i][j]));
        }
    REQUIRE(diagonal_scale > 0.0);

    for (unsigned i = 0; i < K; ++i)
        for (unsigned j = 0; j < K; ++j)
        {
            if (i == j)
                continue;
            REQUIRE(std::abs(pencil[i][j]) / diagonal_scale < 1e-11);
        }

    const auto reference = reference_generalised_norms(n, kappa, modes);
    for (unsigned i = 0; i < K; ++i)
    {
        REQUIRE(pencil[i][i] > 0.0);
        REQUIRE(std::abs(pencil[i][i] - reference[i]) / std::abs(reference[i]) < 1e-10);
    }
}
} // namespace

TEST_CASE(fp_radial_gram_satisfies_pencil_identity_neumann)
{
    for (double kappa : {1e-3, 1e-1})
        for (unsigned n : {0u, 2u})
            check_pencil_identity(n, kappa, WallCondition::Neumann);
}

TEST_CASE(fp_radial_gram_satisfies_pencil_identity_dirichlet)
{
    for (double kappa : {1e-3, 1e-1})
        for (unsigned n : {0u, 2u})
            check_pencil_identity(n, kappa, WallCondition::Dirichlet);
}

// ═══ T5 — load vector against an independent 2-D reference ═══════════════════

namespace
{
std::vector<double> reference_radial_load_vector(unsigned n, const std::vector<SeriesTermData<double>> &modes,
                                                 const std::vector<double> &zi, const std::vector<double> &ui,
                                                 const DiskRule &rule)
{
    std::vector<double> b(modes.size(), 0.0);
    for (std::size_t m = 0; m < modes.size(); ++m)
        b[m] = disk_integral(rule, [&](double r, double phi, double z) {
            // NO omega anywhere: this is the L2_r load vector.
            return layer_value(zi, ui, z)
                 * psinm_r_fp(n, modes[m].root_fp, modes[m].btilde_fp, r)
                 * angular_basis(n, phi);
        });
    return b;
}
} // namespace

namespace
{
// Compares fp_rhs_radial_stratified_inlet against the 2-D reference for
// n = 0..3. The per-block relative test carries an absolute floor tied to the
// LARGEST block's magnitude, because a block's own load vector can legitimately
// vanish identically (see the symmetric profile below), leaving both sides at
// pure round-off with nothing meaningful to divide by.
void check_stratified_load_vector_against_reference(const std::vector<double> &zi, const std::vector<double> &ui)
{
    const double kappa = 1e-2;
    const auto rule = make_disk_rule(zi, 60u, 120u);
    const std::vector<unsigned> angular_indices{0u, 1u, 2u, 3u};

    std::vector<std::vector<SeriesTermData<double>>> modes;
    std::vector<std::vector<double>> got, ref;
    double global_scale = 0.0;
    for (unsigned n : angular_indices)
    {
        modes.push_back(independent_modes(n, 6u, kappa, WallCondition::Neumann));
        got.push_back(fp_rhs_radial_stratified_inlet<double>(n, modes.back(), zi, ui, 100u));
        ref.push_back(reference_radial_load_vector(n, modes.back(), zi, ui, rule));
        REQUIRE(got.back().size() == ref.back().size());
        for (double v : ref.back())
            global_scale = std::max(global_scale, std::abs(v));
    }
    REQUIRE(global_scale > 0.0);

    for (std::size_t i = 0; i < angular_indices.size(); ++i)
    {
        double scale = 0.0;
        for (double v : ref[i])
            scale = std::max(scale, std::abs(v));
        for (std::size_t m = 0; m < got[i].size(); ++m)
            REQUIRE(std::abs(got[i][m] - ref[i][m]) <= 1e-8 * scale + 1e-12 * global_scale);
    }
}
} // namespace

TEST_CASE(fp_radial_rhs_stratified_matches_two_dimensional_reference)
{
    // Symmetric interface pair with equal jumps: every EVEN n >= 2 cancels
    // identically (signed_cap flips the sign of the reflected cap there), so
    // this configuration also exercises the exact-cancellation path.
    check_stratified_load_vector_against_reference({-0.4, 0.4}, {1.0, 0.5, 0.0});
    // Asymmetric, so no block vanishes and every n is a genuine relative test.
    check_stratified_load_vector_against_reference({-0.4, 0.2}, {1.0, 0.5, 0.0});
}

// Companion to the symmetric case above: the n = 2 block really is zero, not
// merely small. Pinned separately so that the absolute floor in the comparison
// above can never quietly mask a genuine n = 2 defect.
TEST_CASE(fp_radial_rhs_symmetric_profile_cancels_even_angular_blocks)
{
    const std::vector<double> zi{-0.4, 0.4};
    const std::vector<double> ui{1.0, 0.5, 0.0};
    const auto modes_n0 = independent_modes(0u, 6u, 1e-2, WallCondition::Neumann);
    const auto b0 = fp_rhs_radial_stratified_inlet<double>(0u, modes_n0, zi, ui, 100u);
    double scale = 0.0;
    for (double v : b0)
        scale = std::max(scale, std::abs(v));
    REQUIRE(scale > 0.1);

    for (unsigned n : {2u, 4u})
    {
        const auto modes = independent_modes(n, 6u, 1e-2, WallCondition::Neumann);
        for (double v : fp_rhs_radial_stratified_inlet<double>(n, modes, zi, ui, 100u))
            REQUIRE(std::abs(v) < 1e-14 * scale);
    }
}

// The a = 0 "center" branch: the cap degenerates to the half-disk, whose n = 0
// projection is HALF THE FULL DISK -- and, on this path, half the r-weighted
// one. Reusing the solver's stored Gaussian rule (which already bakes in
// omega r dr, see finite_peclet_coefficients_gaussian.h) would silently produce
// half the OMEGA-weighted disk instead. The 2-D reference below is what
// distinguishes them, and the companion assertion pins that the two candidate
// values are far apart, so this test genuinely discriminates.
TEST_CASE(fp_radial_rhs_center_branch_uses_the_r_weighted_full_disk)
{
    const std::vector<double> zi{0.0};
    const std::vector<double> ui{1.0, 0.0};
    const double kappa = 1e-2;
    const auto rule = make_disk_rule(zi, 60u, 120u);

    const auto modes = independent_modes(0u, 6u, kappa, WallCondition::Neumann);
    const auto got = fp_rhs_radial_stratified_inlet<double>(0u, modes, zi, ui, 100u);
    const auto ref = reference_radial_load_vector(0u, modes, zi, ui, rule);
    double scale = 0.0;
    for (double v : ref)
        scale = std::max(scale, std::abs(v));
    for (std::size_t m = 0; m < got.size(); ++m)
        REQUIRE(std::abs(got[m] - ref[m]) / scale < 1e-8);

    const auto radial_full = fp_rhs_radial_full_disk_projection<double>(modes, 100u);
    const auto weighted_full = fp_rhs_full_disk_projection<double>(modes, 100u);
    bool clearly_different = false;
    for (std::size_t m = 0; m < modes.size(); ++m)
        if (std::abs(radial_full[m] - weighted_full[m]) > 1e-3 * scale)
            clearly_different = true;
    REQUIRE(clearly_different);
}

TEST_CASE(fp_radial_rhs_uniform_inlet_matches_two_dimensional_reference)
{
    const double kappa = 1e-2;
    const auto rule = make_disk_rule({}, 60u, 120u);
    const auto modes = independent_modes(0u, 6u, kappa, WallCondition::Dirichlet);

    const auto got = fp_rhs_radial_uniform_inlet<double>(0u, modes, 100u);
    std::vector<double> ref(modes.size());
    for (std::size_t m = 0; m < modes.size(); ++m)
        ref[m] = disk_integral(rule, [&](double r, double phi, double) {
            return psinm_r_fp(0u, modes[m].root_fp, modes[m].btilde_fp, r) * angular_basis(0u, phi);
        });
    double scale = 0.0;
    for (double v : ref)
        scale = std::max(scale, std::abs(v));
    for (std::size_t m = 0; m < got.size(); ++m)
        REQUIRE(std::abs(got[m] - ref[m]) / scale < 1e-8);

    // Closed form: b^0_m = sqrt(2 pi) int_0^1 R_{0m} r dr.
    for (std::size_t m = 0; m < modes.size(); ++m)
    {
        const double closed = kSqrtTwoPi * 0.5 * simpson(0.0, 1.0, 20000u, [&](double s) {
            return fp_radial_factor(0u, modes[m].root_fp, modes[m].btilde_fp, s);
        });
        REQUIRE(std::abs(got[m] - closed) / scale < 1e-9);

    }
    // n > 0 carries no uniform-inlet content.
    const auto modes_n2 = independent_modes(2u, 4u, kappa, WallCondition::Dirichlet);
    for (double v : fp_rhs_radial_uniform_inlet<double>(2u, modes_n2, 100u))
        REQUIRE(v == 0.0);
}

TEST_CASE(fp_radial_rhs_cap_kernel_degenerate_branches)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(40u);

    // a == 1: the cap is empty. Purely geometric, hence weight-independent.
    const auto empty = fp_rhs_radial_build_cap_kernel<double>(0u, 1.0, rule);
    REQUIRE(empty.empty);
    // a == 0, n == 0: the half-disk "center" case.
    const auto center = fp_rhs_radial_build_cap_kernel<double>(0u, 0.0, rule);
    REQUIRE(center.center);
    // a == 0, n > 0 even: sin(n*pi/2) == 0.
    const auto zero = fp_rhs_radial_build_cap_kernel<double>(2u, 0.0, rule);
    REQUIRE(zero.zero);
    // a == 0, n odd: NOT degenerate.
    const auto odd = fp_rhs_radial_build_cap_kernel<double>(3u, 0.0, rule);
    REQUIRE(!odd.empty && !odd.center && !odd.zero);

    // The degenerate flags must agree with the weighted builder's, since they
    // encode cap geometry, not the radial weight.
    for (unsigned n : {0u, 1u, 2u, 3u})
        for (double a : {0.0, 0.35, 1.0})
        {
            const auto r = fp_rhs_radial_build_cap_kernel<double>(n, a, rule);
            const auto w = fp_rhs_build_cap_kernel<double>(n, a, rule);
            REQUIRE(r.empty == w.empty);
            REQUIRE(r.center == w.center);
            REQUIRE(r.zero == w.zero);
        }

    bool threw = false;
    try { fp_rhs_radial_build_cap_kernel<double>(0u, 1.5, rule); }
    catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);
}

TEST_CASE(fp_radial_rhs_full_disk_from_gram_column_matches_quadrature)
{
    // Neumann n = 0 block WITH the exact constant mode (Lambda = 0), so the
    // Gram assembly records U^0_{:,0} and the identity <1, R_0m>_r = U^0_{m,0}
    // applies.
    const auto modes = independent_modes(0u, 8u, 1e-2, WallCondition::Neumann);
    REQUIRE(modes[0].rate_fp == 0.0);
    const auto factor = fp_gram_factor_radial(0u, modes, FPGramRadialOptions<double>{});
    REQUIRE(factor.has_constant_mode);
    REQUIRE(factor.constant_mode_local_index == 0u);

    const auto exact = fp_rhs_radial_full_disk_from_gram_column<double>(factor.constant_mode_gram_column);
    const auto quadrature = fp_rhs_radial_full_disk_projection<double>(modes, 100u);
    REQUIRE(exact.size() == quadrature.size());
    double scale = 0.0;
    for (double v : quadrature)
        scale = std::max(scale, std::abs(v));
    for (std::size_t m = 0; m < exact.size(); ++m)
        REQUIRE(std::abs(exact[m] - quadrature[m]) / scale < 1e-10);
}

// ═══ T6 — exact reproduction of in-span data ════════════════════════════════
//
// Any projection reproduces data already in its span, in ANY norm, so this
// validates Gram assembly and solve jointly and is norm-independent -- the
// weighted control below runs the identical protocol with omega restored.

TEST_CASE(fp_radial_projection_reproduces_in_span_data)
{
    const unsigned n = 0u, K = 6u;
    const double kappa = 1e-2;
    const auto modes = independent_modes(n, K, kappa, WallCondition::Neumann);

    std::mt19937 rng(20240917u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> a(K);
    for (double &v : a)
        v = dist(rng);

    // b_m = <f, R_m> with f = sum_k a_k R_k, by INDEPENDENT quadrature.
    const auto reference_u = reference_gram_matrix(n, modes, false);
    const auto reference_w = reference_gram_matrix(n, modes, true);
    std::vector<double> b_radial(K, 0.0), b_weighted(K, 0.0);
    for (unsigned m = 0; m < K; ++m)
        for (unsigned k = 0; k < K; ++k)
        {
            b_radial[m] += a[k] * reference_u[m][k];
            b_weighted[m] += a[k] * reference_w[m][k];
        }

    std::vector<double> c_radial, c_weighted;
    fp_gram_solve_radial(fp_gram_factor_radial(n, modes, FPGramRadialOptions<double>{}), b_radial, c_radial);
    fp_gram_solve_gauss_jacobi(fp_gram_factor_gauss_jacobi(n, modes, FPGramGaussJacobiOptions<double>{}),
                               b_weighted, c_weighted);

    double scale = 0.0;
    for (double v : a)
        scale = std::max(scale, std::abs(v));
    for (unsigned m = 0; m < K; ++m)
    {
        REQUIRE(std::abs(c_radial[m] - a[m]) / scale < 1e-9);
        REQUIRE(std::abs(c_weighted[m] - a[m]) / scale < 1e-9);
    }
}

// ═══ T7 — each projection wins in its own norm ══════════════════════════════
//
// This is what the selector MEANS, and the only test here that a swapped enum
// would fail. The inlet residuals are computed by the independent 2-D disk
// quadrature above, evaluating the series from the solver's own modal data --
// no Gram matrix, no load vector, no projection machinery.

namespace
{
struct InletResiduals
{
    double radial_norm, weighted_norm;
};

InletResiduals inlet_residuals(CDStratifiedSolution<double> &sol, const DiskRule &rule)
{
    // Work in the solver's internal (scaled) units: m_ui / m_zi are exactly the
    // profile the projection was run against.
    const std::vector<double> zi = sol.m_zi, ui = sol.m_ui;
    const auto &data = sol.m_series_data;

    double radial = 0.0, weighted = 0.0;
    for (std::size_t i = 0; i < rule.theta.size(); ++i)
    {
        const double z_axis = std::cos(rule.theta[i]), half_width = std::sin(rule.theta[i]);
        double inner_r = 0.0, inner_w = 0.0;
        for (std::size_t j = 0; j < rule.t.size(); ++j)
        {
            const double y = half_width * rule.t[j];
            const double r = std::sqrt(z_axis * z_axis + y * y);
            const double phi = std::atan2(y, z_axis);
            double psi = 0.0;
            for (const auto &term : data)
                psi += term.coeff_fp * psinm_r_fp(term.n, term.root_fp, term.btilde_fp, r)
                     * angular_basis(term.n, phi);
            const double residual = psi - layer_value(zi, ui, z_axis);
            inner_r += rule.t_weight[j] * residual * residual;
            inner_w += rule.t_weight[j] * residual * residual * (1.0 - r * r);
        }
        radial += rule.theta_weight[i] * inner_r;
        weighted += rule.theta_weight[i] * inner_w;
    }
    return {std::sqrt(radial), std::sqrt(weighted)};
}
} // namespace

TEST_CASE(fp_projection_each_space_minimises_its_own_inlet_norm)
{
    const std::vector<double> zi{-0.4, 0.4};
    const std::vector<double> ui{1.0, 0.5, 0.0};
    const double max_root = 25.0, peclet = 5.0;

    CDStratifiedSolution<double> weighted_solver(zi, ui, 0);
    weighted_solver.set_max_root(max_root);
    weighted_solver.setup_fp_solution(peclet);

    CDStratifiedSolution<double> radial_solver(zi, ui, 0);
    radial_solver.set_max_root(max_root);
    radial_solver.set_rhs_method(RhsMethod::DirectQuadrature);
    radial_solver.set_projection_space(ProjectionSpace::Radial);
    radial_solver.setup_fp_solution(peclet);

    const auto rule = make_disk_rule(radial_solver.m_zi, 80u, 160u);
    const auto from_radial = inlet_residuals(radial_solver, rule);
    const auto from_weighted = inlet_residuals(weighted_solver, rule);

    // Each projection is the best approximation from the SAME span in its own
    // norm, so it cannot lose there. The slack absorbs quadrature noise only.
    REQUIRE(from_radial.radial_norm <= from_weighted.radial_norm + 1e-12);
    REQUIRE(from_weighted.weighted_norm <= from_radial.weighted_norm + 1e-12);

    // ... and the two really are different functions, so the comparison above
    // is not vacuous.
    REQUIRE(std::abs(from_radial.radial_norm - from_weighted.radial_norm) > 1e-9);
}

// ═══ T8 — monotone refinement ═══════════════════════════════════════════════
//
// The residual is the quadratic form ||f||^2 - 2 c.b + c^T U c, assembled from
// the shipped load vector and Gram matrix -- both pinned against independent
// references by T2 and T5. A full 2-D field quadrature is not usable here: at
// max_root = 400 the retained set exceeds 7500 modes, so evaluating the series
// on a 2-D grid would cost tens of millions of Kummer evaluations. What this
// test checks is monotonicity under refinement, not absolute correctness.

namespace
{
double radial_inlet_residual_squared(CDStratifiedSolution<double> &sol)
{
    const std::vector<double> zi = sol.m_zi, ui = sol.m_ui;
    const auto rule = make_disk_rule(zi, 60u, 120u);
    double norm_squared = disk_integral(rule, [&](double, double, double z) {
        const double f = layer_value(zi, ui, z);
        return f * f;
    });

    // Build the two quadrature rule caches ONCE from the whole retained mode
    // set, exactly as the production driver does. Letting each block build its
    // own (the cache-less overloads) rebuilds every Golub-Welsch rule ~100
    // times over at max_root = 400 and dominates this test's run time.
    const unsigned margin = sol.m_fp_cap_quad_margin;
    const auto cap_cache = fp_rhs_build_rule_cache(sol.m_series_data, margin);
    const auto full_disk_cache = fp_rhs_radial_build_full_disk_rule_cache(sol.m_series_data, margin);

    unsigned highest_n = 0;
    for (const auto &t : sol.m_series_data)
        highest_n = std::max(highest_n, t.n);

    // Blocks are independent; at max_root = 400 there are ~100 of them with
    // K up to 100, so this loop is worth parallelising exactly as the driver's
    // own passes are.
    std::vector<double> per_block(highest_n + 1, 0.0);
    std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic)
    for (long long raw = 0; raw <= static_cast<long long>(highest_n); ++raw)
        try
        {
            const unsigned n = static_cast<unsigned>(raw);
            const auto modes = angular_block(sol.m_series_data, n);
            if (modes.empty())
                continue;
            const auto U = radial_gram(n, modes);
            const auto b = fp_rhs_radial_stratified_inlet<double>(n, modes, zi, ui, margin, {},
                                                                  &cap_cache, &full_disk_cache);
            double block = 0.0;
            for (std::size_t i = 0; i < modes.size(); ++i)
            {
                block -= 2.0 * modes[i].coeff_fp * b[i];
                for (std::size_t j = 0; j < modes.size(); ++j)
                    block += modes[i].coeff_fp * U[i][j] * modes[j].coeff_fp;
            }
            per_block[n] = block;
        }
        catch (...)
        {
#pragma omp critical
            {
                if (!failure)
                    failure = std::current_exception();
            }
        }
    if (failure)
        std::rethrow_exception(failure);
    for (double v : per_block)
        norm_squared += v;
    return norm_squared;
}
} // namespace

TEST_CASE(fp_radial_projection_inlet_error_is_monotone_under_refinement)
{
    const std::vector<double> zi{-0.4, 0.4};
    const std::vector<double> ui{1.0, 0.5, 0.0};
    std::vector<double> errors;
    for (double max_root : {100.0, 200.0, 400.0})
    {
        CDStratifiedSolution<double> sol(zi, ui, 0);
        sol.set_max_root(max_root);
        sol.set_rhs_method(RhsMethod::DirectQuadrature);
        sol.set_projection_space(ProjectionSpace::Radial);
        sol.setup_fp_solution(5.0);
        const double squared = radial_inlet_residual_squared(sol);
        REQUIRE(squared > 0.0);
        errors.push_back(std::sqrt(squared));
    }
    for (std::size_t i = 1; i < errors.size(); ++i)
        REQUIRE(errors[i] <= errors[i - 1] + 1e-10);
}

// ═══ T9 — default unchanged (regression guard) ══════════════════════════════
//
// The literals below were produced on the pre-change tree (a build of the
// sources at HEAD before this feature existed) with exactly the configuration
// set up here, and are frozen so that any drift of the DEFAULT (L2_omega) path
// is caught immediately. A configuration that never touches
// set_projection_space must remain bit-comparable.

namespace
{
// Compared against the vector's own scale, not entry by entry: individual
// coefficients legitimately span many decades, and a per-entry relative test on
// the small ones would pin round-off rather than behaviour.
void require_matches_frozen_reference(const std::vector<SeriesTermData<double>> &data,
                                      const std::vector<double> &reference)
{
    REQUIRE(data.size() >= reference.size());
    double scale = 0.0;
    for (double v : reference)
        scale = std::max(scale, std::abs(v));
    REQUIRE(scale > 0.0);
    for (std::size_t k = 0; k < reference.size(); ++k)
        REQUIRE(std::abs(data[k].coeff_fp - reference[k]) / scale < 1e-14);
}
} // namespace

TEST_CASE(fp_projection_default_is_weighted_and_unchanged)
{
    CDStratifiedSolution<double> strat({-0.4, 0.2}, {1.0, 0.3, -0.2}, 0);
    REQUIRE(strat.get_projection_space() == ProjectionSpace::Weighted);
    CDGraetzIsothermalSolution<double> graetz(1.0, 0.0, 0);
    REQUIRE(graetz.get_projection_space() == ProjectionSpace::Weighted);

    strat.set_max_root(30.0);
    strat.setup_fp_solution(5.0);
    require_matches_frozen_reference(strat.m_series_data, {
        1.5092733103397653,
        0.22056901642144769,
        -0.14620962306460902,
        -0.19678120997867271,
        -0.099296991956928343,
        0.067011961930111261,
        0.11243882029303136,
        0.044638207207296346,
        1.9781483173731556,
        -0.73036962991115673,
        -1.0498275269910091,
        -0.37922295892756225,
        0.26304236152558197,
        -0.086130980682117109,
        -1.2961583521482618,
        -0.56329203581392917,
    });

    graetz.set_max_root(30.0);
    graetz.setup_fp_solution(5.0);
    require_matches_frozen_reference(graetz.m_series_data, {
        3.9488496385127636,
        -2.5714000720270307,
        2.0480665762700059,
        -1.6678756238276262,
        1.3448286800451077,
        -1.0362742050756446,
        0.69165760394189446,
    });
}

// ═══ T10 — loud failure, no fallback ════════════════════════════════════════

TEST_CASE(fp_projection_space_string_selector_round_trips_and_rejects)
{
    CDStratifiedSolution<double> sol({0.0}, {1.0, 0.3}, 0);
    sol.set_projection_space(std::string("r"));
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Radial);
    sol.set_projection_space(std::string("w"));
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Weighted);
    sol.set_projection_space(std::string("Radial"));
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Radial);
    sol.set_projection_space(std::string("OMEGA"));
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Weighted);

    bool threw = false;
    try { sol.set_projection_space(std::string("q")); }
    catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);
    // The rejected string must not have changed the selection.
    REQUIRE(sol.get_projection_space() == ProjectionSpace::Weighted);
}

// This case previously pinned the OPPOSITE: that Radial + Representer was
// rejected at setup. That restriction is gone -- finite_peclet_rhs_representer_radial.h
// supplies the L2_r representer -- so what is pinned now is that BOTH backends
// run on the radial path and agree, which is the property the lifted
// restriction has to earn. Detailed agreement lives in
// tests/test_fp_representer_radial.cpp; this is the solver-level check.
TEST_CASE(fp_projection_radial_supports_both_rhs_backends)
{
    const std::vector<double> zi{-0.4, 0.2};
    const std::vector<double> ui{1.0, 0.3, -0.2};

    CDStratifiedSolution<double> direct(zi, ui, 0);
    direct.set_max_root(30.0);
    direct.set_projection_space(ProjectionSpace::Radial);
    direct.set_rhs_method(RhsMethod::DirectQuadrature);
    direct.setup_fp_solution(5.0);
    REQUIRE(direct.get_solution_method() == SolutionMethod::ModifiedRoots);

    CDStratifiedSolution<double> representer(zi, ui, 0);
    representer.set_max_root(30.0);
    representer.set_projection_space(ProjectionSpace::Radial);
    representer.set_rhs_method(RhsMethod::Representer);
    representer.setup_fp_solution(5.0);
    REQUIRE(representer.get_solution_method() == SolutionMethod::ModifiedRoots);

    REQUIRE(direct.m_series_data.size() == representer.m_series_data.size());
    double scale = 0.0;
    for (const auto &t : direct.m_series_data)
        scale = std::max(scale, std::abs(t.coeff_fp));
    REQUIRE(scale > 0.0);
    for (std::size_t k = 0; k < direct.m_series_data.size(); ++k)
        REQUIRE(std::abs(direct.m_series_data[k].coeff_fp
                         - representer.m_series_data[k].coeff_fp) / scale < 1e-9);
}

TEST_CASE(fp_projection_radial_with_ultraspherical_gram_throws_no_fallback)
{
    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.set_max_root(15.0);
    sol.set_projection_space(ProjectionSpace::Radial);
    sol.set_rhs_method(RhsMethod::DirectQuadrature);
    sol.set_gram_method(GramMethod::Ultraspherical);

    bool threw = false;
    std::string message;
    try { sol.setup_fp_solution(5.0); }
    catch (const std::invalid_argument &e) { threw = true; message = e.what(); }
    REQUIRE(threw);
    REQUIRE(message.find("ultraspherical") != std::string::npos);
    REQUIRE(sol.get_solution_method() == SolutionMethod::Uninitialized);

    sol.set_gram_method(GramMethod::GaussJacobiQR);
    sol.setup_fp_solution(5.0);
    REQUIRE(sol.get_solution_method() == SolutionMethod::ModifiedRoots);
}

// ═══ T11 — conditioning characterisation (the point of the exercise) ════════
//
// The modified modes are a Riesz basis of L2_r but not of L2_omega: omega
// vanishes at the wall, so multiplication by it is positive but not boundedly
// invertible. Normalised to unit diagonal, lambda_min of U is therefore flat in
// the truncation level K while that of W decays like 1/K. Thresholds are loose
// on purpose -- this documents intent and guards against a regression that
// silently reintroduces omega into U.

TEST_CASE(fp_radial_gram_conditioning_is_flat_in_truncation)
{
    CDStratifiedSolution<double> sol({-0.4, 0.4}, {1.0, 0.5, 0.0}, 0);
    sol.set_max_root(120.0);
    sol.setup_fp_solution(5.0);

    const auto all_n0 = angular_block(sol.m_series_data, 0u);
    const unsigned K = 10u;
    REQUIRE(all_n0.size() >= 2u * K);
    const std::vector<SeriesTermData<double>> small(all_n0.begin(), all_n0.begin() + K);
    const std::vector<SeriesTermData<double>> large(all_n0.begin(), all_n0.begin() + 2 * K);

    const double u_small = symmetric_eigenvalues(normalise_to_unit_diagonal(radial_gram(0u, small))).front();
    const double u_large = symmetric_eigenvalues(normalise_to_unit_diagonal(radial_gram(0u, large))).front();
    const double w_small = symmetric_eigenvalues(normalise_to_unit_diagonal(weighted_gram(0u, small))).front();
    const double w_large = symmetric_eigenvalues(normalise_to_unit_diagonal(weighted_gram(0u, large))).front();

    REQUIRE(u_small > 0.0 && u_large > 0.0 && w_small > 0.0 && w_large > 0.0);
    // U: flat under doubling. Measured ~0.747 at both K for this configuration.
    REQUIRE(u_large >= 0.9 * u_small);
    // W: decaying like 1/K. Measured 0.1405 -> 0.0644.
    REQUIRE(w_large <= 0.75 * w_small);
}
