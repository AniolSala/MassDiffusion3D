// Finite-Peclet inlet coefficients by L2_r projection: a thin orchestrator over
// the inlet-blind radial Gram backend (finite_peclet_gram_radial.h) and the
// inlet-specific r-weighted RHS builders (finite_peclet_rhs_radial.h). See
// finite_peclet_rhs_radial.h for the design contract this file must respect:
// it defines no cap/cap-rule/full-disk/inlet-profile logic of its own.

#include "finite_peclet_coefficients_radial.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace
{
    template <typename T>
    bool finite(T x) { return std::isfinite(static_cast<long double>(x)); }

    template <typename T>
    std::vector<std::vector<unsigned>> groups(unsigned K, const std::vector<SeriesTermData<T>> &d)
    {
        if (K > d.size())
            throw std::invalid_argument("finite-Peclet radial coefficients: max_K exceeds series data");
        unsigned high = 0;
        for (unsigned k = 0; k < K; ++k)
        {
            const auto &t = d[k];
            if (!finite(t.root_fp) || !finite(t.btilde_fp) || !finite(t.rate_fp) || t.root_fp < T(0) || t.btilde_fp < T(0) || t.rate_fp < T(0))
                throw std::runtime_error("finite-Peclet radial coefficients: invalid mode data");
            high = std::max(high, t.n);
        }
        std::vector<std::vector<unsigned>> g(K ? high + 1 : 0);
        for (unsigned k = 0; k < K; ++k)
            g[d[k].n].push_back(k);
        for (auto &v : g)
            for (std::size_t i = 1; i < v.size(); ++i)
                if (!(d[v[i]].m > d[v[i - 1]].m))
                    throw std::runtime_error("finite-Peclet radial coefficients: non-increasing radial index");
        return g;
    }

    template <typename T>
    std::vector<SeriesTermData<T>> block_modes(const std::vector<SeriesTermData<T>> &d, const std::vector<unsigned> &ids)
    {
        std::vector<SeriesTermData<T>> modes;
        modes.reserve(ids.size());
        for (unsigned id : ids)
            modes.push_back(d[id]);
        return modes;
    }
}

template <typename T>
void fp_stratified_coefficients_radial(const std::vector<T> &p, const std::vector<T> &v, unsigned K,
                                       std::vector<SeriesTermData<T>> &d, unsigned cap_quad_margin,
                                       WallCondition wall_condition, const FPGramRadialOptions<T> &gram_options,
                                       RhsMethod rhs_method, unsigned rhs_representer_margin,
                                       bool enable_timing, T *projection_square_norm)
{
    if (!K)
        return;
    if (rhs_method != RhsMethod::DirectQuadrature && rhs_method != RhsMethod::Representer)
        throw std::invalid_argument("finite-Peclet radial coefficients: unhandled RHS method");
    // Accepted for signature symmetry with the weighted driver; the radial Gram
    // backend is boundary-blind (it infers the exact constant mode from
    // rate_fp == 0 alone), so nothing here branches on it.
    static_cast<void>(wall_condition);
    auto g = groups(K, d);
    const std::size_t num_blocks = g.size();
    std::exception_ptr failure;

    // Pass 1: Gram matrix assembly and factorization for every angular block.
    // Gram-first (not RHS-first) so the n=0 block's exact constant-mode column
    // U^0_{:,0}, when present, is available before the RHS pass runs -- see
    // finite_peclet_gram_radial.h's constant_mode_gram_column and
    // finite_peclet_rhs_radial.h's fp_rhs_radial_full_disk_from_gram_column.
    std::vector<FPGramRadialFactor<T>> factors(num_blocks);

    // Local copy: the caller's options must not be mutated. Only the
    // representer backend needs the Gram assembly's samples retained.
    FPGramRadialOptions<T> gram_options_local = gram_options;
    gram_options_local.retain_samples = (rhs_method == RhsMethod::Representer);

    const auto gram_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
#pragma omp parallel for schedule(dynamic)
    for (long long raw = 0; raw < static_cast<long long>(num_blocks); ++raw)
        try
        {
            std::size_t ni = std::size_t(raw);
            if (g[ni].empty())
                continue;
            const auto modes = block_modes(d, g[ni]);
            factors[ni] = fp_gram_factor_radial(unsigned(ni), modes, gram_options_local);
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
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial Gram matrix assembly (stratified, "
                  << num_blocks << " angular blocks) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - gram_t0).count() << " s." << std::endl;

    // Pass 2: RHS load-vector assembly for every angular block. The n=0 block
    // uses the Gram-assembled exact constant-mode column when the assembly
    // recorded one (Neumann wall); every other case uses the independent
    // full-disk quadrature (DirectQuadrature) or the constant representer
    // (Representer).
    //
    // The quadrature rule caches are pure waste under Representer, which needs
    // no per-mode rule at all, so they are built only on the DirectQuadrature
    // path. The cap rule cache is finite_peclet_rhs.h's own -- the (0,1/2) cap
    // rule is measure-independent -- while the full-disk cache must come from
    // THIS path's (0,0) shifted-Legendre builder. Both are built ONCE here,
    // from the WHOLE retained mode set, and shared read-only across every
    // block's call.
    const std::vector<SeriesTermData<T>> all_active_modes(d.begin(), d.begin() + K);
    FPCapRuleCache<T> cap_rule_cache, full_disk_rule_cache;
    if (rhs_method == RhsMethod::DirectQuadrature)
    {
        cap_rule_cache = fp_rhs_build_rule_cache(all_active_modes, cap_quad_margin);
        full_disk_rule_cache = fp_rhs_radial_build_full_disk_rule_cache(all_active_modes, cap_quad_margin);
    }
    FPRepresenterRadialOptions<T> representer_options;
    representer_options.representer_margin = rhs_representer_margin;

    std::vector<std::vector<T>> all_rhs(num_blocks);
    const auto rhs_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
#pragma omp parallel for schedule(dynamic)
    for (long long raw = 0; raw < static_cast<long long>(num_blocks); ++raw)
        try
        {
            std::size_t ni = std::size_t(raw);
            if (g[ni].empty())
                continue;
            std::vector<T> full_disk_column;
            if (ni == 0 && factors[0].has_constant_mode)
                full_disk_column = fp_rhs_radial_full_disk_from_gram_column(factors[0].constant_mode_gram_column);
            if (rhs_method == RhsMethod::DirectQuadrature)
            {
                const auto modes = block_modes(d, g[ni]);
                all_rhs[ni] = fp_rhs_radial_stratified_inlet(unsigned(ni), modes, p, v, cap_quad_margin, full_disk_column,
                                                             &cap_rule_cache, &full_disk_rule_cache);
            }
            else
            {
                all_rhs[ni] = fp_rhs_radial_stratified_inlet_representer(unsigned(ni), factors[ni], p, v,
                                                                        representer_options, full_disk_column);
                // Release the retained samples as soon as this block's load
                // vector exists; they are the pass's only significant memory.
                std::vector<T>().swap(factors[ni].radial_samples);
                factors[ni].samples_retained = false;
            }
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
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial RHS vector assembly (stratified, "
                  << num_blocks << " angular blocks) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_t0).count() << " s." << std::endl;

    // Pass 3: per-block linear solve.
    std::vector<std::vector<T>> results(num_blocks);
    const auto solve_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
#pragma omp parallel for schedule(dynamic)
    for (long long raw = 0; raw < static_cast<long long>(num_blocks); ++raw)
        try
        {
            std::size_t ni = std::size_t(raw);
            if (g[ni].empty())
                continue;
            fp_gram_solve_radial(factors[ni], all_rhs[ni], results[ni]);
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
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial linear solve (stratified, "
                  << num_blocks << " angular blocks) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_t0).count() << " s." << std::endl;

    // chat^T b == chat^T U chat == || f_K ||^2 in L2_r (not L2_omega: this path
    // solves with the unweighted Gram matrix).
    T inlet_square_norm = 0;
    for (std::size_t n = 0; n < num_blocks; ++n)
    {
        if (results[n].size() != g[n].size())
            throw std::runtime_error("finite-Peclet radial coefficients: unwritten group");
        for (std::size_t i = 0; i < g[n].size(); ++i)
        {
            d[g[n][i]].coeff_fp = results[n][i];
            inlet_square_norm += results[n][i] * all_rhs[n][i];
        }
    }
    if (projection_square_norm)
        *projection_square_norm = inlet_square_norm;
}

template <typename T>
void fp_graetz_coefficients_radial(unsigned K, std::vector<SeriesTermData<T>> &d,
                                   WallCondition wall_condition, const FPGramRadialOptions<T> &gram_options,
                                   unsigned cap_quad_margin, RhsMethod rhs_method,
                                   unsigned rhs_representer_margin, bool enable_timing,
                                   T *projection_square_norm)
{
    if (!K)
        return;
    static_cast<void>(wall_condition);
    if (rhs_method != RhsMethod::DirectQuadrature && rhs_method != RhsMethod::Representer)
        throw std::invalid_argument("fp_graetz_coefficients_radial: unhandled RHS method");
    auto g = groups(K, d);
    for (unsigned k = 0; k < K; ++k)
        d[k].coeff_fp = std::numeric_limits<T>::quiet_NaN();
    if (g.empty() || g[0].empty())
        throw std::runtime_error("fp_graetz_coefficients_radial: missing n=0 group");
    for (std::size_t n = 1; n < g.size(); ++n)
        for (unsigned index : g[n])
            d[index].coeff_fp = static_cast<T>(0);

    const auto modes = block_modes(d, g[0]);

    // Gram first, then RHS, then solve (matching the stratified driver above):
    // the RHS needs to know whether the Gram assembly recorded an exact
    // constant-mode column before it can choose between the exact
    // U^0_{:,0} identity and the independent quadrature. Graetz has a
    // Dirichlet wall, so in practice there is no Lambda = 0 mode and the
    // quadrature branch is the one taken; the identity branch is kept because
    // it is a mathematical selection, not a backend fallback.
    FPGramRadialOptions<T> gram_options_local = gram_options;
    gram_options_local.retain_samples = (rhs_method == RhsMethod::Representer);
    const auto gram_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    const auto factor = fp_gram_factor_radial(0u, modes, gram_options_local);
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial Gram matrix assembly (graetz) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - gram_t0).count() << " s." << std::endl;

    const auto rhs_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    // Mathematical-identity selection, not a backend fallback: the exact
    // Gram-column formula is used whenever it applies, otherwise the selected
    // RHS backend is used. There is still no retry across backends.
    std::vector<T> rhs;
    if (factor.has_constant_mode)
        rhs = fp_rhs_radial_full_disk_from_gram_column(factor.constant_mode_gram_column);
    else if (rhs_method == RhsMethod::DirectQuadrature)
        rhs = fp_rhs_radial_uniform_inlet(0u, modes, cap_quad_margin);
    else
    {
        FPRepresenterRadialOptions<T> representer_options;
        representer_options.representer_margin = rhs_representer_margin;
        rhs = fp_rhs_radial_uniform_inlet_representer(0u, factor, representer_options);
    }
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial RHS vector assembly (graetz) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_t0).count() << " s." << std::endl;

    const auto solve_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    std::vector<T> coefficients;
    fp_gram_solve_radial(factor, rhs, coefficients);
    if (enable_timing)
        std::cout << "Timing: finite-Peclet radial linear solve (graetz) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_t0).count() << " s." << std::endl;

    T inlet_square_norm = 0;
    for (std::size_t i = 0; i < g[0].size(); ++i)
    {
        d[g[0][i]].coeff_fp = coefficients[i];
        inlet_square_norm += coefficients[i] * rhs[i];
    }
    if (projection_square_norm)
        *projection_square_norm = inlet_square_norm;
    for (unsigned k = 0; k < K; ++k)
        if (!finite(d[k].coeff_fp))
            throw std::runtime_error("fp_graetz_coefficients_radial: unwritten coefficient");
}

template void fp_stratified_coefficients_radial(const std::vector<double> &, const std::vector<double> &, unsigned, std::vector<SeriesTermData<double>> &, unsigned, WallCondition, const FPGramRadialOptions<double> &, RhsMethod, unsigned, bool, double *);
template void fp_stratified_coefficients_radial(const std::vector<long double> &, const std::vector<long double> &, unsigned, std::vector<SeriesTermData<long double>> &, unsigned, WallCondition, const FPGramRadialOptions<long double> &, RhsMethod, unsigned, bool, long double *);
template void fp_graetz_coefficients_radial(unsigned, std::vector<SeriesTermData<double>> &, WallCondition, const FPGramRadialOptions<double> &, unsigned, RhsMethod, unsigned, bool, double *);
template void fp_graetz_coefficients_radial(unsigned, std::vector<SeriesTermData<long double>> &, WallCondition, const FPGramRadialOptions<long double> &, unsigned, RhsMethod, unsigned, bool, long double *);
