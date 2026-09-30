// Finite-Peclet stratified coefficients: thin orchestrator over the shared
// inlet-blind Gram backends (finite_peclet_gram_*.h) and the shared
// inlet-specific RHS builders (finite_peclet_rhs.h). See finite_peclet_rhs.h
// for the design contract this file must respect: it defines no Cap/cap-rule/
// full-disk/inlet-profile logic of its own.

#include "finite_peclet_coefficients_gauss_jacobi.h"
#include "finite_peclet_gram_ultraspherical.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
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
            throw std::invalid_argument("finite-Peclet Gauss-Jacobi coefficients: max_K exceeds series data");
        unsigned high = 0;
        for (unsigned k = 0; k < K; ++k)
        {
            const auto &t = d[k];
            if (!finite(t.root_fp) || !finite(t.btilde_fp) || !finite(t.rate_fp) || t.root_fp < T(0) || t.btilde_fp < T(0) || t.rate_fp < T(0))
                throw std::runtime_error("finite-Peclet Gauss-Jacobi coefficients: invalid mode data");
            high = std::max(high, t.n);
        }
        std::vector<std::vector<unsigned>> g(K ? high + 1 : 0);
        for (unsigned k = 0; k < K; ++k)
            g[d[k].n].push_back(k);
        for (auto &v : g)
            for (std::size_t i = 1; i < v.size(); ++i)
                if (!(d[v[i]].m > d[v[i - 1]].m))
                    throw std::runtime_error("finite-Peclet Gauss-Jacobi coefficients: non-increasing radial index");
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
void fp_stratified_coefficients_gauss_jacobi(const std::vector<T> &p, const std::vector<T> &v, unsigned K, std::vector<SeriesTermData<T>> &d, unsigned cap_quad_margin,
    GramMethod gram_method, WallCondition wall_condition, const FPGramGaussJacobiOptions<T> &gram_options,
    RhsMethod rhs_method, unsigned rhs_representer_margin, bool enable_timing, T *projection_square_norm)
{
    if (!K)
        return;
    // Representer requires the Gram-assembled radial samples, which only the
    // GaussJacobiQR factor retains (the ultraspherical factor holds no such
    // samples). No fallback: this is checked BEFORE Pass 1 runs, not
    // discovered mid-solve.
    if (rhs_method == RhsMethod::Representer && gram_method != GramMethod::GaussJacobiQR)
        throw std::invalid_argument("finite-Peclet Gauss-Jacobi coefficients: RhsMethod::Representer "
                                    "requires GramMethod::GaussJacobiQR (the ultraspherical factor "
                                    "retains no samples to contract against)");
    auto g = groups(K, d);
    const std::size_t num_blocks = g.size();
    std::exception_ptr failure;

    // Pass 1: Gram matrix assembly and factorization for every angular block.
    // Gram-first (rather than the previous RHS-first ordering) so the n=0
    // block's exact constant-mode column, when present, is available before
    // the RHS pass runs -- see finite_peclet_gram_gauss_jacobi.h's
    // constant_mode_gram_column and finite_peclet_rhs.h's
    // fp_rhs_full_disk_from_gram_column.
    std::vector<FPGramGaussJacobiFactor<T>> gj_factors;
    std::vector<FPGramUltrasphericalFactor<T>> us_factors;
    if (gram_method == GramMethod::GaussJacobiQR)
        gj_factors.resize(num_blocks);
    else if (gram_method == GramMethod::Ultraspherical)
        us_factors.resize(num_blocks);
    else
        throw std::invalid_argument("finite-Peclet Gauss-Jacobi coefficients: unhandled Gram method");

    // Local copy: the caller's options must not be mutated. Only the
    // Representer RHS backend needs the Gram assembly's samples retained.
    FPGramGaussJacobiOptions<T> gram_options_local = gram_options;
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
            if (gram_method == GramMethod::GaussJacobiQR)
                gj_factors[ni] = fp_gram_factor_gauss_jacobi(unsigned(ni), modes, gram_options_local);
            else
            {
                FPGramUltrasphericalOptions<T> us_options;
                us_options.ultraspherical_factor = gram_options.oversampling_factor;
                us_options.ultraspherical_margin = gram_options.oversampling_margin;
                us_options.enable_order_check = gram_options.enable_order_check;
                us_options.minimum_factor = gram_options.minimum_factor;
                us_factors[ni] = fp_gram_factor_ultraspherical(unsigned(ni), modes, wall_condition, us_options);
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
        std::cout << "Timing: finite-Peclet Gram matrix assembly (stratified, "
                  << num_blocks << " angular blocks) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - gram_t0).count() << " s." << std::endl;

    // Pass 2: RHS load-vector assembly for every angular block. The n=0 block
    // uses the Gram-assembled exact constant-mode column when GaussJacobiQR
    // recorded one (Neumann wall); every other case falls back to the
    // independent full-disk quadrature (DirectQuadrature) or the constant
    // representer of section 6.3 (Representer).
    //
    // The cap and full-disk quadrature rule caches (finite_peclet_rhs.h) are
    // built ONCE here, from the WHOLE retained mode set (every block, not
    // just one), and shared read-only across every block's call below --
    // pure waste under Representer, which needs no per-mode quadrature rule
    // at all, so they are built only on the DirectQuadrature path.
    const std::vector<SeriesTermData<T>> all_active_modes(d.begin(), d.begin() + K);
    FPCapRuleCache<T> cap_rule_cache, full_disk_rule_cache;
    if (rhs_method == RhsMethod::DirectQuadrature)
    {
        cap_rule_cache = fp_rhs_build_rule_cache(all_active_modes, cap_quad_margin);
        full_disk_rule_cache = fp_rhs_build_full_disk_rule_cache(all_active_modes, cap_quad_margin);
    }
    FPRepresenterOptions<T> representer_options;
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
            if (ni == 0 && gram_method == GramMethod::GaussJacobiQR && gj_factors[0].has_constant_mode)
                full_disk_column = fp_rhs_full_disk_from_gram_column(gj_factors[0].constant_mode_gram_column);
            switch (rhs_method)
            {
            case RhsMethod::DirectQuadrature:
            {
                const auto modes = block_modes(d, g[ni]);
                all_rhs[ni] = fp_rhs_stratified_inlet(unsigned(ni), modes, p, v, cap_quad_margin, full_disk_column,
                                                      &cap_rule_cache, &full_disk_rule_cache);
                break;
            }
            case RhsMethod::Representer:
                all_rhs[ni] = fp_rhs_stratified_inlet_representer(unsigned(ni), gj_factors[ni], p, v,
                                                                  representer_options, full_disk_column);
                std::vector<T>().swap(gj_factors[ni].radial_samples);
                gj_factors[ni].samples_retained = false;
                break;
            default:
                throw std::invalid_argument("finite-Peclet Gauss-Jacobi coefficients: unhandled RHS method");
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
        std::cout << "Timing: finite-Peclet RHS vector assembly (stratified, "
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
            if (gram_method == GramMethod::GaussJacobiQR)
                fp_gram_solve_gauss_jacobi(gj_factors[ni], all_rhs[ni], results[ni]);
            else
                fp_gram_solve_ultraspherical(us_factors[ni], all_rhs[ni], results[ni]);
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
        std::cout << "Timing: finite-Peclet linear solve (stratified, "
                  << num_blocks << " angular blocks) took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_t0).count() << " s." << std::endl;

    // chat^T b, accumulated on the way out. Equals chat^T W chat = || f_K ||^2
    // because the coefficients just solved W chat = b, so the quadratic form of
    // the Gram matrix costs one multiply-add per mode and no assembly at all.
    T inlet_square_norm = 0;
    for (std::size_t n = 0; n < num_blocks; ++n)
    {
        if (results[n].size() != g[n].size())
            throw std::runtime_error("finite-Peclet Gauss-Jacobi coefficients: unwritten group");
        for (std::size_t i = 0; i < g[n].size(); ++i)
        {
            d[g[n][i]].coeff_fp = results[n][i];
            inlet_square_norm += results[n][i] * all_rhs[n][i];
        }
    }
    if (projection_square_norm)
        *projection_square_norm = inlet_square_norm;
}

template void fp_stratified_coefficients_gauss_jacobi(const std::vector<double> &, const std::vector<double> &, unsigned, std::vector<SeriesTermData<double>> &, unsigned, GramMethod, WallCondition, const FPGramGaussJacobiOptions<double> &, RhsMethod, unsigned, bool, double *);
template void fp_stratified_coefficients_gauss_jacobi(const std::vector<long double> &, const std::vector<long double> &, unsigned, std::vector<SeriesTermData<long double>> &, unsigned, GramMethod, WallCondition, const FPGramGaussJacobiOptions<long double> &, RhsMethod, unsigned, bool, long double *);
