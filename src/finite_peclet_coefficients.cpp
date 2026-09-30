// ---------------------------------------------------------------------------
// finite_peclet_coefficients.cpp — L2_omega Galerkin projection of the inlet
// onto the modified (finite-Peclet) radial basis. See the header for why this
// must be a linear solve rather than the closed form of the tex draft.
//
// Layout:
//   * angular_cap / radial_integral  — building blocks of the right-hand side
//   * build_gram                     — W_mk = int omega R_m R_k r dr
//   * solve_gram                     — equilibrated Cholesky, LU fall-back
//   * fp_graetz_coefficients / fp_stratified_coefficients
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h>

#include "finite_peclet_coefficients.h"
#include "finite_peclet_radial.h"
#include "finite_peclet_quadrature.h"

namespace
{
constexpr long double LD_SQRT_2PI   = 2.50662827463100050242L; // sqrt(2*pi)
constexpr long double LD_SQRT_2_PI   = 0.79788456080286535588L; // sqrt(2/pi)
constexpr long double LD_INV_SQRT_PI = 0.56418958354775628695L; // 1/sqrt(pi)

// Angular integral A_n(phi_c) = int_{-phi_c}^{phi_c} Phi_n^S(phi) dphi over a
// symmetric cap of half-angle phi_c in [0, pi]:
//   n = 0 : sqrt(2/pi) * phi_c          (Phi_0^S = 1/sqrt(2*pi))
//   n > 0 : 2 sin(n phi_c)/(n sqrt(pi)) (Phi_n^S = cos(n phi)/sqrt(pi))
// The antisymmetric (sine) modes integrate to zero over a symmetric cap, so a
// z-only stratified inlet excites the symmetric modes alone.
template <typename Ttype>
inline Ttype angular_cap(unsigned n, const Ttype &phi_c)
{
    if (n == 0)
        return static_cast<Ttype>(LD_SQRT_2_PI) * phi_c;
    return static_cast<Ttype>(2.0L) * std::sin(static_cast<Ttype>(n) * phi_c) *
           static_cast<Ttype>(LD_INV_SQRT_PI) / static_cast<Ttype>(n);
}

// int_a^c omega(r) R_fp(n,b,bt,r) * ang(r) * r dr, with omega = 1 - r^2.
// ang(r) is either a constant (full disk) or the r-dependent cap half-angle
// arccos(z_i/r). The weight is omega ALONE: the Galerkin test functions are
// omega * R_m * Phi_n, exactly as in the classical (kappa = 0) projection.
template <typename Ttype>
Ttype radial_integral(unsigned n, const Ttype &b, const Ttype &bt,
                      const Ttype &a, const Ttype &c,
                      bool use_cap, const Ttype &z_i, const Ttype &ang_const,
                      const std::vector<Ttype> &nodes, const std::vector<Ttype> &wts)
{
    if (!(c > a)) return static_cast<Ttype>(0);
    const Ttype span = c - a;
    Ttype total = static_cast<Ttype>(0);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const Ttype r = a + span * nodes[i];
        const Ttype wgt = span * wts[i];
        Ttype ang = ang_const;
        if (use_cap)
        {
            Ttype ratio = z_i / r;
            ratio = std::max(static_cast<Ttype>(-1), std::min(static_cast<Ttype>(1), ratio));
            ang = angular_cap<Ttype>(n, std::acos(ratio));
        }
        total += wgt * (static_cast<Ttype>(1) - r * r) * psinm_r_fp(n, b, bt, r) * ang * r;
    }
    return total;
}

template <typename Ttype>
Ttype full_domain_integral_from_row(const FPQuadratureRule<Ttype> &quadrature,
                                    const Ttype *radial_row,
                                    const Ttype &angular_constant)
{
    Ttype total = static_cast<Ttype>(0);
    for (std::size_t q = 0; q < quadrature.size(); ++q)
        total += quadrature.omega_measure[q] * radial_row[q] * angular_constant;
    return total;
}

// Gram matrix W_mk = int_0^1 omega R_m R_k r dr for one angular index.
// The modes are tabulated once at the quadrature nodes and the entries are then
// dot products, keeping the count of hypergeometric evaluations at O(K * Q)
// rather than O(K^2 * Q).
template <typename Ttype>
void build_gram_from_block(const FPQuadratureRule<Ttype> &quadrature,
                           const FPRadialBlock<Ttype> &block,
                           std::vector<std::vector<Ttype>> &W)
{
    const size_t K = block.mode_count(), Q = quadrature.size();
    W.assign(K, std::vector<Ttype>(K, static_cast<Ttype>(0)));
    for (size_t m = 0; m < K; ++m)
        for (size_t k = m; k < K; ++k)
        {
            Ttype s = static_cast<Ttype>(0);
            const Ttype *row_m = block.row_data(m), *row_k = block.row_data(k);
            for (size_t q = 0; q < Q; ++q)
                s += quadrature.omega_measure[q] * row_m[q] * row_k[q];
            W[m][k] = s;
            W[k][m] = s;
        }
}

// Solve W c = b for a symmetric positive-definite Gram matrix W.
//
// Stability measures, in order:
//  1. Symmetric equilibration by the diagonal, W~ = D^-1 W D^-1 with
//     D = diag(sqrt(W_mm)). This normalises the diagonal to 1 and removes any
//     scale disparity between modes; for a Gram matrix it is close to the
//     condition-optimal diagonal scaling.
//  2. Cholesky on W~ (valid: W is SPD by construction).
//  3. If a Cholesky pivot is non-positive — which can only happen if two rates
//     have collided and the modes became numerically dependent — fall back to
//     partial-pivot Gaussian elimination rather than returning garbage.
//  4. A non-finite result throws.
template <typename Ttype>
void solve_gram(std::vector<std::vector<Ttype>> W, std::vector<Ttype> rhs,
                std::vector<Ttype> &x)
{
    const size_t K = W.size();
    x.assign(K, static_cast<Ttype>(0));
    if (K == 0) return;

    // --- 1. equilibrate ---------------------------------------------------
    std::vector<Ttype> d(K);
    for (size_t i = 0; i < K; ++i)
    {
        if (!(W[i][i] > static_cast<Ttype>(0)))
            throw std::runtime_error("fp_coefficients: non-positive Gram diagonal "
                                     "(a modified mode vanished identically).");
        d[i] = std::sqrt(W[i][i]);
    }
    for (size_t i = 0; i < K; ++i)
    {
        for (size_t j = 0; j < K; ++j) W[i][j] /= (d[i] * d[j]);
        rhs[i] /= d[i];
    }

    // --- 2. Cholesky W~ = L L^T -------------------------------------------
    bool spd = true;
    std::vector<std::vector<Ttype>> L(K, std::vector<Ttype>(K, static_cast<Ttype>(0)));
    for (size_t i = 0 ; i < K && spd; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            Ttype s = W[i][j];
            for (size_t k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j)
            {
                if (!(s > static_cast<Ttype>(0))) { spd = false; break; }
                L[i][i] = std::sqrt(s);
            }
            else
            {
                L[i][j] = s / L[j][j];
            }
        }
    }

    std::vector<Ttype> y = rhs;
    if (spd)
    {
        for (size_t i = 0; i < K; ++i)                       // forward: L y = rhs
        {
            Ttype s = y[i];
            for (size_t k = 0; k < i; ++k) s -= L[i][k] * y[k];
            y[i] = s / L[i][i];
        }
        for (size_t ii = K; ii-- > 0;)                       // back: L^T x = y
        {
            Ttype s = y[ii];
            for (size_t k = ii + 1; k < K; ++k) s -= L[k][ii] * y[k];
            y[ii] = s / L[ii][ii];
        }
    }
    else
    {
        // --- 3. fall-back: partial-pivot Gaussian elimination --------------
        std::vector<std::vector<Ttype>> A = W;
        y = rhs;
        for (size_t i = 0; i < K; ++i)
        {
            size_t best = i;
            Ttype bv = std::fabs(A[i][i]);
            for (size_t r = i + 1; r < K; ++r)
                if (std::fabs(A[r][i]) > bv) { bv = std::fabs(A[r][i]); best = r; }
            if (best != i) { std::swap(A[i], A[best]); std::swap(y[i], y[best]); }
            if (!(std::fabs(A[i][i]) > static_cast<Ttype>(0)))
                throw std::runtime_error("fp_coefficients: singular Gram matrix — two "
                                         "finite-Peclet rates have collided.");
            for (size_t r = i + 1; r < K; ++r)
            {
                const Ttype f = A[r][i] / A[i][i];
                if (f == static_cast<Ttype>(0)) continue;
                for (size_t c = i; c < K; ++c) A[r][c] -= f * A[i][c];
                y[r] -= f * y[i];
            }
        }
        for (size_t ii = K; ii-- > 0;)
        {
            Ttype s = y[ii];
            for (size_t k = ii + 1; k < K; ++k) s -= A[ii][k] * y[k];
            y[ii] = s / A[ii][ii];
        }
    }

    // --- 4. un-equilibrate and validate -----------------------------------
    for (size_t i = 0; i < K; ++i)
    {
        x[i] = y[i] / d[i];
        if (!std::isfinite(static_cast<long double>(x[i])))
            throw std::runtime_error("fp_coefficients: non-finite coefficient from the Gram solve.");
    }
}

} // namespace

template <typename Ttype>
void fp_graetz_coefficients(const Ttype &kappa, unsigned max_K,
                            std::vector<SeriesTermData<Ttype>> &series_data,
                            unsigned panels)
{
    FPRadialTable<Ttype> table = fp_build_radial_table(series_data, max_K, panels);
    fp_graetz_coefficients_from_table(kappa, max_K, series_data, table);
}

template <typename Ttype>
void fp_graetz_coefficients_from_table(const Ttype &, unsigned max_K,
                                       std::vector<SeriesTermData<Ttype>> &series_data,
                                       const FPRadialTable<Ttype> &table)
{
    const FPRadialBlock<Ttype> *block = table.find_block(0);
    if (!block) throw std::runtime_error("fp_graetz_coefficients: missing n=0 block");
    if (table.total_modes != max_K || max_K > series_data.size())
        throw std::invalid_argument("fp_graetz_coefficients: incompatible table");

    // The uniform inlet excites only the axisymmetric n = 0 family.
    for (unsigned k = 0; k < max_K; ++k)
        if (series_data[k].n != 0) series_data[k].coeff_fp = static_cast<Ttype>(0);
    const size_t K = block->mode_count();
    if (K == 0) return;

    std::vector<Ttype> rhs(K);
    for (size_t m = 0; m < K; ++m)
        rhs[m] = static_cast<Ttype>(LD_SQRT_2PI) *
                 full_domain_integral_from_row(table.quadrature, block->row_data(m),
                                               static_cast<Ttype>(1));

    std::vector<std::vector<Ttype>> W;
    build_gram_from_block(table.quadrature, *block, W);
    std::vector<Ttype> C;
    solve_gram<Ttype>(W, rhs, C);

    for (size_t m = 0; m < K; ++m) series_data[block->flat_indices[m]].coeff_fp = C[m];
}

template <typename Ttype>
void fp_stratified_coefficients(const std::vector<Ttype> &zi,
                                const std::vector<Ttype> &ui,
                                const Ttype &kappa, unsigned max_K,
                                std::vector<SeriesTermData<Ttype>> &series_data,
                                unsigned panels)
{
    FPRadialTable<Ttype> table = fp_build_radial_table(series_data, max_K, panels);
    fp_stratified_coefficients_from_table(zi, ui, kappa, max_K, series_data, table);
}

template <typename Ttype>
void fp_stratified_coefficients_from_table(const std::vector<Ttype> &zi,
                                           const std::vector<Ttype> &ui,
                                           const Ttype &, unsigned max_K,
                                           std::vector<SeriesTermData<Ttype>> &series_data,
                                           const FPRadialTable<Ttype> &table)
{
    if (max_K == 0) return;
    if (table.total_modes != max_K || max_K > series_data.size())
        throw std::invalid_argument("fp_stratified_coefficients: incompatible table");

    const unsigned n_layers = static_cast<unsigned>(ui.size());
    const unsigned n_interfaces = (n_layers >= 1) ? n_layers - 1 : 0;
    const Ttype u0 = (n_layers > 0) ? ui[0] : static_cast<Ttype>(0);
    const Ttype one = static_cast<Ttype>(1);

    std::vector<unsigned> active;
    active.reserve(n_interfaces);
    for (unsigned i = 0; i < n_interfaces; ++i)
        if (ui[i + 1] != ui[i]) active.push_back(i);

    std::exception_ptr err = nullptr;
#pragma omp parallel for schedule(dynamic)
    for (long long block_index = 0; block_index < static_cast<long long>(table.blocks.size()); ++block_index)
    {
        try
        {
            const FPRadialBlock<Ttype> &block = table.blocks[static_cast<std::size_t>(block_index)];
            const unsigned n = block.n;
            const std::vector<unsigned> &g = block.flat_indices;
            const size_t K = g.size();
            std::vector<Ttype> rhs(K, static_cast<Ttype>(0));

            for (size_t m = 0; m < K; ++m)
            {
                const SeriesTermData<Ttype> &t = series_data[g[m]];

                Ttype acc = static_cast<Ttype>(0);

                // Baseline u0 over the whole disk (axisymmetric part only).
                if (n == 0)
                    acc += u0 * static_cast<Ttype>(LD_SQRT_2PI) *
                           full_domain_integral_from_row(table.quadrature, block.row_data(m),
                                                         static_cast<Ttype>(1));

                // Step jumps: sum_i du_i * int over the cap {z > z_i}.
                for (size_t a = 0; a < active.size(); ++a)
                {
                    const unsigned i = active[a];
                    const Ttype z_i = zi[i], du = ui[i + 1] - ui[i];
                    const Ttype z_abs = std::fabs(z_i);

                    // Outer partial cap r in [|z_i|, 1]: half-angle arccos(z_i/r).
                    Ttype T;
                    if (z_i == static_cast<Ttype>(0))
                    {
                        T = static_cast<Ttype>(0);
                        for (std::size_t q = 0; q < table.quadrature.size(); ++q)
                        {
                            const Ttype r = table.quadrature.nodes[q];
                            Ttype ang = angular_cap<Ttype>(n, static_cast<Ttype>(std::acos(0)));
                            T += table.quadrature.omega_measure[q] * block.at(m, q) * ang;
                        }
                    }
                    else T = radial_integral<Ttype>(n, t.root_fp, t.btilde_fp, z_abs, one,
                                /*use_cap=*/true, z_i, static_cast<Ttype>(0),
                                table.quadrature.nodes, table.quadrature.weights);

                    // Inner full disk r in [0, |z_i|] exists only for z_i < 0; the
                    // complete angular circle retains n = 0 alone.
                    if (z_i < static_cast<Ttype>(0) && n == 0)
                        T += radial_integral<Ttype>(0, t.root_fp, t.btilde_fp,
                                static_cast<Ttype>(0), z_abs, /*use_cap=*/false,
                                static_cast<Ttype>(0), static_cast<Ttype>(LD_SQRT_2PI),
                                table.quadrature.nodes, table.quadrature.weights);

                    acc += du * T;
                }
                rhs[m] = acc;
            }

            std::vector<std::vector<Ttype>> W;
            build_gram_from_block(table.quadrature, block, W);
            std::vector<Ttype> C;
            solve_gram<Ttype>(W, rhs, C);

            for (size_t m = 0; m < K; ++m) series_data[g[m]].coeff_fp = C[m];
        }
        catch (...)
        {
#pragma omp critical
            { if (!err) err = std::current_exception(); }
        }
    }
    if (err) std::rethrow_exception(err);
}

// --- Explicit instantiations -------------------------------------------------
template void fp_graetz_coefficients<double>(const double &, unsigned, std::vector<SeriesTermData<double>> &, unsigned);
template void fp_graetz_coefficients<long double>(const long double &, unsigned, std::vector<SeriesTermData<long double>> &, unsigned);
template void fp_stratified_coefficients<double>(const std::vector<double> &, const std::vector<double> &, const double &, unsigned, std::vector<SeriesTermData<double>> &, unsigned);
template void fp_stratified_coefficients<long double>(const std::vector<long double> &, const std::vector<long double> &, const long double &, unsigned, std::vector<SeriesTermData<long double>> &, unsigned);
template void fp_graetz_coefficients_from_table<double>(const double &, unsigned, std::vector<SeriesTermData<double>> &, const FPRadialTable<double> &);
template void fp_graetz_coefficients_from_table<long double>(const long double &, unsigned, std::vector<SeriesTermData<long double>> &, const FPRadialTable<long double> &);
template void fp_stratified_coefficients_from_table<double>(const std::vector<double> &, const std::vector<double> &, const double &, unsigned, std::vector<SeriesTermData<double>> &, const FPRadialTable<double> &);
template void fp_stratified_coefficients_from_table<long double>(const std::vector<long double> &, const std::vector<long double> &, const long double &, unsigned, std::vector<SeriesTermData<long double>> &, const FPRadialTable<long double> &);
