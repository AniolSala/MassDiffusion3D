// ---------------------------------------------------------------------------
// finite_peclet_roots.cpp — implementation of the finite-Peclet decay-rate
// solver. See finite_peclet_roots.h for the interface and theory references.
//
// Adapted and re-derived from the reference solver on the `main` branch. The
// bracketed scan is deliberately the source of correctness: the analytic seed
// (fp_seed) only warm-starts, and the strict upper bound Lam < beta^2
// (eq. rate_ordering) plus a monotone upward scan guarantee that the m-th sign
// change above the previous rate is exactly Lam_{nm}. Every produced rate is
// validated against the ordering/bound invariants before being returned.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <omp.h>

#include "finite_peclet_roots.h"
#include "finite_peclet_radial.h"

namespace
{
// McMahon asymptotic zeros of J_n / J_n', used only to seed the
// diffusion-dominated (large-kappa) branch. Accurate to <1e-3 for m >= 2 and
// adequate as a seed for m = 1; never used as a reference value.
template <typename Ttype>
Ttype bessel_zero_seed(unsigned n, unsigned m, WallCondition wall)
{
    const long double N = static_cast<long double>(n);
    const long double M = static_cast<long double>(m + 1); // 1-based
    const long double mu = 4.0L * N * N;
    const long double PI = 3.14159265358979323846L;
    const long double beta = (wall == WallCondition::Dirichlet)
        ? (M + 0.5L * N - 0.25L) * PI
        : (M + 0.5L * N - 0.75L) * PI;
    const long double b8 = 8.0L * beta;
    const long double z = beta - (mu - 1.0L) / b8
                - 4.0L * (mu - 1.0L) * (7.0L * mu - 31.0L) / (3.0L * b8 * b8 * b8);
    return static_cast<Ttype>(z);
}

// Representative alpha_{nm} used only when the caller has no tabulated
// pseudo-products/norms data to compute the exact eq:seed_root ratio (e.g. a
// ladder solve on synthetic beta2 arrays in tests). alpha = 2 exactly for the
// Neumann axisymmetric family (virial identity); alpha in (1.25, 1.9) otherwise
// in practice. Only ever a Newton warm-start.
template <typename Ttype>
Ttype fallback_alpha(unsigned n, WallCondition wall)
{
    return (wall == WallCondition::Neumann && n == 0)
        ? static_cast<Ttype>(2.0L) : static_cast<Ttype>(1.55L);
}
} // namespace

template <typename Ttype>
Ttype fp_char(const unsigned &n, const Ttype &Lam, const Ttype &kappa, WallCondition wall)
{
    if (!(Lam > static_cast<Ttype>(0)))
        return static_cast<Ttype>(0);
    const Ttype b  = std::sqrt(Lam);
    const Ttype bt = b * (static_cast<Ttype>(1) + kappa * Lam);
    return (wall == WallCondition::Dirichlet) ? psi_at_1_fp(n, b, bt)
                                              : dpsi_dr_at_1_fp(n, b, bt);
}

template <typename Ttype>
Ttype fp_seed(const unsigned &n, unsigned m, const Ttype &base_beta2,
              const Ttype &kappa, const Ttype &alpha, WallCondition wall)
{
    // Diagonal-resummation seed (eq. seed_root). alpha = alpha_{nm} = U^n_mm / N_nm^2
    // is the caller-supplied exact bare-mode ratio (tabulated pseudo-products
    // diagonal over the weighted norm). Only a seed: correctness enforced downstream.
    const Ttype s1 = static_cast<Ttype>(2) * base_beta2 /
        (static_cast<Ttype>(1) + std::sqrt(static_cast<Ttype>(1) +
            static_cast<Ttype>(4) * kappa * alpha * base_beta2));

    // Diffusion-dominated seed: kappa*Lam^2 + Lam = q^2, q a Bessel zero.
    const Ttype q = bessel_zero_seed<Ttype>(n, m, wall);
    const Ttype s2 = (std::sqrt(static_cast<Ttype>(1) +
        static_cast<Ttype>(4) * kappa * q * q) - static_cast<Ttype>(1)) /
        (static_cast<Ttype>(2) * kappa);

    return (kappa * base_beta2 > static_cast<Ttype>(4)) ? s2 : std::min(s1, s2);
}

template <typename Ttype>
Ttype fp_solve_bracketed(const unsigned &n, const Ttype &kappa, WallCondition wall,
                         Ttype lo, Ttype hi, const Ttype &rel_tol, unsigned max_iter)
{
    unsigned dummy_iters = 0;
    return fp_solve_bracketed(n, kappa, wall, lo, hi, rel_tol, max_iter, dummy_iters);
}

template <typename Ttype>
Ttype fp_solve_bracketed(const unsigned &n, const Ttype &kappa, WallCondition wall,
                         Ttype lo, Ttype hi, const Ttype &rel_tol, unsigned max_iter,
                         unsigned &final_iters, Ttype seed)
{
    final_iters = 0;
    Ttype flo = fp_char(n, lo, kappa, wall);
    Ttype fhi = fp_char(n, hi, kappa, wall);
    if (!std::isfinite(flo) || !std::isfinite(fhi))
        throw std::runtime_error("fp_solve_bracketed: non-finite characteristic function.");
    if (flo == static_cast<Ttype>(0)) return lo;
    if (fhi == static_cast<Ttype>(0)) return hi;
    if ((flo > static_cast<Ttype>(0)) == (fhi > static_cast<Ttype>(0)))
        throw std::runtime_error("fp_solve_bracketed: invalid bracket (no sign change).");

    Ttype x  = (seed > lo && seed < hi && std::isfinite(seed))
               ? seed
               : static_cast<Ttype>(0.5L) * (lo + hi);
    Ttype fx = fp_char(n, x, kappa, wall);

    // Secant slope, not a finite-difference derivative: needs one extra distinct
    // point, not two extra fp_char evaluations per iteration (confluent
    // hypergeometric evaluations are not cheap, especially at high n/Lam). Seed
    // it with the bracket endpoint of opposite sign to x — already evaluated
    // above (flo/fhi), guaranteed distinct from x, and every later iteration
    // reuses the previous accepted (x, fx) instead of paying for new points.
    Ttype x_prev = ((flo > static_cast<Ttype>(0)) == (fx > static_cast<Ttype>(0))) ? hi : lo;
    Ttype f_prev = ((flo > static_cast<Ttype>(0)) == (fx > static_cast<Ttype>(0))) ? fhi : flo;

    for (unsigned it = 0; it < max_iter; ++it)
    {
        final_iters = it + 1;

        // The candidate is still required to land strictly inside (lo, hi) before
        // acceptance — that containment check, not the slope estimate, is what
        // keeps the bracket valid and guarantees convergence; a degenerate secant
        // slope just falls back to bisection for that one iteration.
        Ttype x_new = static_cast<Ttype>(0.5L) * (lo + hi);
        if (x != x_prev)
        {
            const Ttype d = (fx - f_prev) / (x - x_prev);
            if (std::isfinite(d) && d != static_cast<Ttype>(0))
            {
                const Ttype cand = x - fx / d;
                if (std::isfinite(cand) && cand > lo && cand < hi) x_new = cand;
            }
        }

        const Ttype f_new = fp_char(n, x_new, kappa, wall);
        if (!std::isfinite(f_new))
            throw std::runtime_error("fp_solve_bracketed: non-finite iterate.");

        if ((flo > static_cast<Ttype>(0)) == (f_new > static_cast<Ttype>(0)))
             { lo = x_new; flo = f_new; }
        else { hi = x_new; }

        x_prev = x; f_prev = fx;

        const bool converged =
            (std::fabs(hi - lo) <= rel_tol * std::max(static_cast<Ttype>(1), std::fabs(x_new))) ||
            (f_new == static_cast<Ttype>(0));
        x = x_new; fx = f_new;
        if (converged) return x;
    }
    return x;
}

template <typename Ttype>
std::vector<Ttype> fp_solve_ladder(const unsigned &n,
                                   const std::vector<Ttype> &base_beta2,
                                   const Ttype &kappa,
                                   WallCondition wall,
                                   bool skip_zero_mode,
                                   const Ttype &rel_tol,
                                   unsigned max_iter,
                                   const std::vector<Ttype> &alpha_by_m)
{
    std::vector<unsigned> dummy_iters;
    return fp_solve_ladder(n, base_beta2, kappa, wall, skip_zero_mode, rel_tol, max_iter, dummy_iters, alpha_by_m);
}

template <typename Ttype>
std::vector<Ttype> fp_solve_ladder(const unsigned &n,
                                   const std::vector<Ttype> &base_beta2,
                                   const Ttype &kappa,
                                   WallCondition wall,
                                   bool skip_zero_mode,
                                   const Ttype &rel_tol,
                                   unsigned max_iter,
                                   std::vector<unsigned> &final_iters,
                                   const std::vector<Ttype> &alpha_by_m)
{
    const size_t M = base_beta2.size();
    std::vector<Ttype> out(M, static_cast<Ttype>(0));
    final_iters.assign(M, 0);
    if (M == 0) return out;

    // kappa == 0 is the bare limit: rates are exactly beta^2 (no root-finding).
    if (!(kappa > static_cast<Ttype>(0)))
    {
        for (size_t m = 0; m < M; ++m) out[m] = base_beta2[m];
        return out;
    }

    const Ttype PI = static_cast<Ttype>(3.14159265358979323846L);
    const Ttype Pe = static_cast<Ttype>(1) / std::sqrt(kappa);

    size_t m0 = 0;
    if (skip_zero_mode) { out[0] = static_cast<Ttype>(0); m0 = 1; }

    Ttype prev = static_cast<Ttype>(0);
    for (size_t m = m0; m < M; ++m)
    {
        // Strict upper bound (eq. rate_ordering): Lam_m(kappa) < beta_m^2.
        const Ttype upper = base_beta2[m];
        if (!(upper > prev))
            throw std::runtime_error("fp_solve_ladder: base rates are not ascending.");

        // Scan step below the smallest credible gap.
        const Ttype base_gap = upper - prev;
        Ttype step = static_cast<Ttype>(0.2L) * std::min(PI * Pe, base_gap);
        if (!(step > static_cast<Ttype>(0)) || !std::isfinite(step))
            step = static_cast<Ttype>(0.2L) * upper;

        // Start clear of the previous root: fp_char vanishes there (exactly so for
        // the Neumann n=0 zero mode where prev == 0), so a guard ~1e-3*step keeps
        // the scan from reporting the previous root as a spurious bracket while
        // remaining far below the smallest genuine gap.
        const Ttype guard = static_cast<Ttype>(1e-3L) * step;
        Ttype lo = prev + guard;
        if (!(lo < upper))
            throw std::runtime_error("fp_solve_ladder: degenerate scan interval.");

        Ttype flo = fp_char(n, lo, kappa, wall);
        if (!std::isfinite(flo))
            throw std::runtime_error("fp_solve_ladder: non-finite scan value.");
        if (flo == static_cast<Ttype>(0))
            throw std::runtime_error("fp_solve_ladder: rates closer than the scan guard.");

        Ttype root = static_cast<Ttype>(-1);
        const Ttype cap = upper * static_cast<Ttype>(1.000001L);

        // March upward to the FIRST sign change above prev: that is Lam_m. Sign
        // comparison only, never a product (flo*fhi can overflow).
        for (Ttype hi = std::min(lo + step, cap); ; hi = std::min(hi + step, cap))
        {
            const Ttype fhi = fp_char(n, hi, kappa, wall);
            if (!std::isfinite(fhi))
                throw std::runtime_error("fp_solve_ladder: non-finite scan value.");
            if (fhi == static_cast<Ttype>(0)) { root = hi; break; }
            if ((flo > static_cast<Ttype>(0)) != (fhi > static_cast<Ttype>(0)))
            {
                const Ttype alpha = (m < alpha_by_m.size()) ? alpha_by_m[m] : fallback_alpha<Ttype>(n, wall);
                const Ttype seed = fp_seed<Ttype>(n, static_cast<unsigned>(m), base_beta2[m], kappa, alpha, wall);
                root = fp_solve_bracketed(n, kappa, wall, lo, hi, rel_tol, max_iter, final_iters[m], seed);
                break;
            }
            lo = hi; flo = fhi;
            if (hi >= cap) break;
        }

        if (!(root > prev) || !(root <= upper) || !std::isfinite(root))
        {
            std::ostringstream msg;
            msg << "fp_solve_ladder: failed to locate rate (n=" << n << ", m=" << m
                << ", kappa=" << static_cast<long double>(kappa) << ").";
            throw std::runtime_error(msg.str());
        }
        out[m] = root;
        prev = root;
    }
    return out;
}

template <typename Ttype>
void fp_modify_roots(std::vector<SeriesTermData<Ttype>> &series_data, unsigned max_K,
                     const Ttype &kappa, WallCondition wall,
                     const Ttype &rel_tol, unsigned max_iter,
                     const std::vector<std::vector<std::vector<Ttype>>> &pseudo_products_matrix)
{
    if (max_K == 0) return;

    // Group flat indices by angular index n (contiguous ascending m already).
    unsigned max_n = 0;
    for (unsigned k = 0; k < max_K; ++k) max_n = std::max(max_n, series_data[k].n);
    std::vector<std::vector<unsigned>> groups(max_n + 1);
    for (unsigned k = 0; k < max_K; ++k) groups[series_data[k].n].push_back(k);

    // One independent ladder per n. Parallel over n; sequential inside.
#pragma omp parallel for schedule(dynamic)
    for (unsigned n = 0; n <= max_n; ++n)
    {
        const std::vector<unsigned> &g = groups[n];
        if (g.empty()) continue;

        std::vector<Ttype> base_beta2(g.size());
        std::vector<Ttype> alpha_by_m(g.size());
        const bool have_row = n < pseudo_products_matrix.size();
        for (size_t j = 0; j < g.size(); ++j)
        {
            const SeriesTermData<Ttype> &t = series_data[g[j]];
            base_beta2[j] = t.exp_rate; // β² (bare decay rate)

            // alpha_{nm} = U^n_mm / N_nm^2 (eq. seed_root): the tabulated unweighted
            // pseudo-products diagonal over the weighted bare norm — a lookup, not a
            // computation. Falls back to a representative constant only for a mode
            // outside the tabulated pseudo-products range (Newton warm-start only).
            const bool have_entry = have_row && t.m < pseudo_products_matrix[n].size() &&
                t.m < pseudo_products_matrix[n][t.m].size() && t.norm > static_cast<Ttype>(0);
            alpha_by_m[j] = have_entry
                ? pseudo_products_matrix[n][t.m][t.m] / t.norm
                : fallback_alpha<Ttype>(n, wall);
        }

        // The zero/constant mode has base rate ≈ 0 (Neumann n = 0, m = 0).
        const bool skip_zero =
            base_beta2[0] < static_cast<Ttype>(1e-12L);

        const std::vector<Ttype> Lam =
            fp_solve_ladder<Ttype>(n, base_beta2, kappa, wall, skip_zero, rel_tol, max_iter, alpha_by_m);

        for (size_t j = 0; j < g.size(); ++j)
        {
            SeriesTermData<Ttype> &t = series_data[g[j]];
            const Ttype L = Lam[j];
            const Ttype b = std::sqrt(L);
            t.rate_fp   = L;
            t.root_fp   = b;
            t.btilde_fp = btilde_from_b(b, kappa);
        }
    }
}

// --- Explicit instantiations -------------------------------------------------
template void fp_modify_roots<double>(std::vector<SeriesTermData<double>> &, unsigned, const double &, WallCondition, const double &, unsigned, const std::vector<std::vector<std::vector<double>>> &);
template void fp_modify_roots<long double>(std::vector<SeriesTermData<long double>> &, unsigned, const long double &, WallCondition, const long double &, unsigned, const std::vector<std::vector<std::vector<long double>>> &);
template double      fp_char<double>(const unsigned &, const double &, const double &, WallCondition);
template long double fp_char<long double>(const unsigned &, const long double &, const long double &, WallCondition);
template double      fp_seed<double>(const unsigned &, unsigned, const double &, const double &, const double &, WallCondition);
template long double fp_seed<long double>(const unsigned &, unsigned, const long double &, const long double &, const long double &, WallCondition);
template double      fp_solve_bracketed<double>(const unsigned &, const double &, WallCondition, double, double, const double &, unsigned);
template long double fp_solve_bracketed<long double>(const unsigned &, const long double &, WallCondition, long double, long double, const long double &, unsigned);
template double      fp_solve_bracketed<double>(const unsigned &, const double &, WallCondition, double, double, const double &, unsigned, unsigned &, double);
template long double fp_solve_bracketed<long double>(const unsigned &, const long double &, WallCondition, long double, long double, const long double &, unsigned, unsigned &, long double);
template std::vector<double>      fp_solve_ladder<double>(const unsigned &, const std::vector<double> &, const double &, WallCondition, bool, const double &, unsigned, const std::vector<double> &);
template std::vector<long double> fp_solve_ladder<long double>(const unsigned &, const std::vector<long double> &, const long double &, WallCondition, bool, const long double &, unsigned, const std::vector<long double> &);
template std::vector<double>      fp_solve_ladder<double>(const unsigned &, const std::vector<double> &, const double &, WallCondition, bool, const double &, unsigned, std::vector<unsigned> &, const std::vector<double> &);
template std::vector<long double> fp_solve_ladder<long double>(const unsigned &, const std::vector<long double> &, const long double &, WallCondition, bool, const long double &, unsigned, std::vector<unsigned> &, const std::vector<long double> &);
