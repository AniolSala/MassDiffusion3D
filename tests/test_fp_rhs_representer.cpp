// Regression tests for the representer-based finite-Peclet RHS backend
// (plans/RHS_REPRESENTER_PLAN.md section 6 / section 8).

#include "tinytest.h"

#include "finite_peclet_rhs_representer.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_gram_gauss_jacobi.h"
#include "shifted_jacobi_basis.h"
#include "series_term_struct.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

namespace
{
// Synthetic angular block, independent of any root-finding machinery --
// mirrors test_fp_gram_backends.cpp's own local helper of the same name (a
// separate copy: this file must not depend on that translation unit).
std::vector<SeriesTermData<double>> synthetic_block(unsigned n, unsigned K, double kappa)
{
    std::vector<SeriesTermData<double>> modes(K);
    for (unsigned m = 0; m < K; ++m)
    {
        const double beta = 4.0 * (m + 1) + 2.0 * n + 2.0;
        const double beta2 = beta * beta;
        const double Lambda = (n == 0 && m == 0) ? 0.0 : 2.0 * beta2 / (1.0 + std::sqrt(1.0 + 4.0 * kappa * beta2));
        modes[m].n = n;
        modes[m].m = m;
        modes[m].root_fp = std::sqrt(Lambda);
        modes[m].btilde_fp = modes[m].root_fp * (1.0 + kappa * Lambda);
        modes[m].rate_fp = Lambda;
    }
    return modes;
}

// Canonical Gauss-Jacobi (alpha=1, beta=n) nodes/weights with EXACTLY
// node_count == requested_nodes, via a throwaway single-mode block (the rule
// depends only on (n, node_count), never on the mode's own root/btilde).
std::pair<std::vector<double>, std::vector<double>> jacobi_1n_rule(unsigned n, unsigned requested_nodes)
{
    std::vector<SeriesTermData<double>> modes(1);
    modes[0].n = n;
    modes[0].m = 0;
    modes[0].root_fp = 1.0;
    modes[0].btilde_fp = 1.0;
    modes[0].rate_fp = 1.0;
    FPGramGaussJacobiOptions<double> options;
    options.oversampling_factor = 1u;
    options.oversampling_margin = requested_nodes - 1u;
    options.minimum_factor = 1u;
    options.retain_samples = true;
    const auto factor = fp_gram_factor_gauss_jacobi(n, modes, options);
    return {factor.quadrature_nodes, factor.quadrature_weights};
}

// Independent, PUBLIC-API-only reference builder of psi at an EXPLICIT order
// (fp_cap_representer only exposes the calibrated/exact order rule, not a
// caller-chosen order) -- used to probe order sensitivity directly. Built
// from the same public primitives (make_fp_gauss_jacobi_rule,
// fp_rhs_build_cap_kernel, shifted_jacobi_table, shifted_jacobi_norm) the
// production implementation uses, exactly as test_fp_rhs.cpp's own
// reference_stratified_rhs re-derives fp_rhs_stratified_inlet from public
// primitives.
std::vector<double> local_build_psi(unsigned n, double a, unsigned J, const std::vector<double> &gram_nodes, unsigned order)
{
    const auto rule = make_fp_gauss_jacobi_rule<double>(order);
    const auto kernel = fp_rhs_build_cap_kernel<double>(n, a, rule);
    if (kernel.center)
        return std::vector<double>();
    if (kernel.empty || kernel.zero)
        return std::vector<double>(gram_nodes.size(), 0.0);

    const std::size_t Ns = kernel.s.size();
    const auto cap_table = shifted_jacobi_table<double>(J, n, kernel.s);
    std::vector<double> M(J, 0.0);
    for (unsigned j = 0; j < J; ++j)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < Ns; ++i)
            sum += kernel.kernel[i] * cap_table[static_cast<std::size_t>(j) * Ns + i];
        M[j] = kernel.prefactor * sum;
    }

    const std::size_t Q = gram_nodes.size();
    const auto psi_table = shifted_jacobi_table<double>(J, n, gram_nodes);
    std::vector<double> psi(Q, 0.0);
    for (std::size_t q = 0; q < Q; ++q)
    {
        double sum = 0.0;
        for (unsigned j = 0; j < J; ++j)
            sum += (M[j] / shifted_jacobi_norm<double>(j, n)) * psi_table[static_cast<std::size_t>(j) * Q + q];
        psi[q] = sum;
    }
    return psi;
}
}

// ─── T-RP-01: order rule exact for n >= 1 ──────────────────────────────────

TEST_CASE(fp_representer_order_is_exact_for_n_geq_1)
{
    FPRepresenterOptions<double> options;
    for (unsigned n : {1u, 2u, 3u, 20u, 60u, 99u})
        for (unsigned J : {40u, 140u})
            for (double a : {0.1, 0.4, 0.82})
            {
                const auto gram_nodes = jacobi_1n_rule(n, J).first;
                REQUIRE(gram_nodes.size() == J);
                const unsigned order = fp_representer_order(n, J, a, options);
                const auto psi1 = local_build_psi(n, a, J, gram_nodes, order);
                const auto psi2 = local_build_psi(n, a, J, gram_nodes, 3u * order);
                REQUIRE(psi1.size() == psi2.size());
                double max_abs = 0.0;
                for (double v : psi2)
                    max_abs = std::max(max_abs, std::abs(v));
                for (std::size_t q = 0; q < psi1.size(); ++q)
                {
                    const double rel = std::abs(psi1[q] - psi2[q]) / std::max(1.0, max_abs);
                    // Worst measured 7.2e-13 (n=3, J=140, a=0.4) with the
                    // plain (non-compensated) summation this independent
                    // reference builder deliberately uses; the production
                    // implementation itself uses compensated summation and is
                    // pinned far tighter by fp_representer_order_n0_shallow_cap_calibration.
                    REQUIRE(rel < 5e-12);
                }
            }
}

// ─── T-RP-02: n == 0 shallow-cap calibration ───────────────────────────────

TEST_CASE(fp_representer_order_n0_shallow_cap_calibration)
{
    // Table from plan section 6.2 (K=60, J=N=140).
    struct Row { double a; unsigned expected_order; };
    const std::vector<Row> rows{{0.82, 118u}, {0.40, 125u}, {0.10, 170u}, {0.05, 230u}, {0.02, 410u}};
    FPRepresenterOptions<double> options; // defaults: margin = 40
    const unsigned J = 140u;
    const auto gram_nodes = jacobi_1n_rule(0u, J).first;

    for (const auto &row : rows)
    {
        const unsigned order = fp_representer_order(0u, J, row.a, options);
        REQUIRE(order == row.expected_order);

        const auto psi = local_build_psi(0u, row.a, J, gram_nodes, order);
        const auto psi_ref = local_build_psi(0u, row.a, J, gram_nodes, 20u * order);
        double max_diff = 0.0, max_abs = 0.0;
        for (std::size_t q = 0; q < psi.size(); ++q)
        {
            max_diff = std::max(max_diff, std::abs(psi[q] - psi_ref[q]));
            max_abs = std::max(max_abs, std::abs(psi_ref[q]));
        }
        REQUIRE(max_diff / max_abs < 1e-11);
    }
}

// ─── T-RP-03: shallow-cap guard fires ──────────────────────────────────────
//
// The n=0 order formula clamps a at 0.01 (see finite_peclet_rhs_representer.h
// section 6.2): below that floor the calibrated order under-resolves the
// TRUE a, which is exactly the regime the mandatory doubling guard exists to
// catch. representer_margin=0 removes the only other source of headroom.

TEST_CASE(fp_cap_representer_shallow_guard_fires_below_clamp_floor)
{
    FPRepresenterOptions<double> options;
    options.representer_margin = 0u;
    const unsigned J = 40u;
    const auto gram_nodes = jacobi_1n_rule(0u, J).first;

    bool threw = false;
    try
    {
        fp_cap_representer<double>(0u, 0.001, J, gram_nodes, options);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    REQUIRE(threw);
}

// ─── T-RP-04: degenerate caps ───────────────────────────────────────────────

TEST_CASE(fp_cap_representer_degenerate_cases)
{
    FPRepresenterOptions<double> options;
    const unsigned J = 40u;

    const auto gram_nodes_n2 = jacobi_1n_rule(2u, J).first;
    const auto psi_empty_cap = fp_cap_representer<double>(2u, 1.0, J, gram_nodes_n2, options); // a >= 1
    REQUIRE(psi_empty_cap.size() == gram_nodes_n2.size());
    for (double v : psi_empty_cap)
        REQUIRE(v == 0.0);

    const auto psi_zero = fp_cap_representer<double>(2u, 0.0, J, gram_nodes_n2, options); // n=2 even, a=0
    REQUIRE(psi_zero.size() == gram_nodes_n2.size());
    for (double v : psi_zero)
        REQUIRE(v == 0.0);

    const auto gram_nodes_n0 = jacobi_1n_rule(0u, J).first;
    const auto psi_center = fp_cap_representer<double>(0u, 0.0, J, gram_nodes_n0, options); // n=0, a=0
    REQUIRE(psi_center.empty());
}

// ─── T-RP-05: constant full-disk representer ───────────────────────────────

TEST_CASE(fp_representer_contract_constant_matches_full_disk_projection)
{
    constexpr double kSqrtTwoPi = 2.50662827463100050242;
    for (unsigned K : {20u, 60u, 100u})
    {
        const auto modes = synthetic_block(0u, K, 1e-2);
        FPGramGaussJacobiOptions<double> gram_options;
        gram_options.retain_samples = true;
        const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, gram_options);

        const std::vector<double> psi(factor.node_count, kSqrtTwoPi / 2.0);
        const auto contracted = fp_representer_contract(factor, psi);
        const auto reference = fp_rhs_full_disk_projection(modes, 400u);
        REQUIRE(contracted.size() == reference.size());
        for (std::size_t i = 0; i < contracted.size(); ++i)
        {
            const double rel = std::abs(contracted[i] - reference[i]) / std::max(1.0, std::abs(reference[i]));
            REQUIRE(rel < 1e-9);
        }
    }
}

// ─── T-RP-06: signed caps ───────────────────────────────────────────────────

TEST_CASE(fp_rhs_stratified_inlet_representer_signed_cap_branches_match_direct)
{
    const std::vector<double> zi{-0.35};
    const std::vector<double> ui{1.0, 0.4};
    FPRepresenterOptions<double> options;

    for (unsigned n : {0u, 1u, 2u}) // n=0 (full-minus-pos), odd (+pos), even>0 (-pos)
    {
        const auto modes = synthetic_block(n, 12u, 1e-2);
        FPGramGaussJacobiOptions<double> gram_options;
        gram_options.retain_samples = true;
        const auto factor = fp_gram_factor_gauss_jacobi(n, modes, gram_options);

        const auto rhs_repr = fp_rhs_stratified_inlet_representer(n, factor, zi, ui, options);
        const auto rhs_direct = fp_rhs_stratified_inlet(n, modes, zi, ui, 100u);
        REQUIRE(rhs_repr.size() == rhs_direct.size());
        for (std::size_t i = 0; i < rhs_repr.size(); ++i)
        {
            const double rel = std::abs(rhs_repr[i] - rhs_direct[i]) / std::max(1.0, std::abs(rhs_direct[i]));
            REQUIRE(rel < 1e-8);
        }
    }
}

// ─── T-RP-07: zero radial evaluations (source-level) ───────────────────────

TEST_CASE(fp_rhs_representer_never_calls_fp_radial_factor_source_grep)
{
    // Search for an actual call/reference site ("fp_radial_factor(" or
    // "fp_radial_factor<"), not the bare identifier: this file's own header
    // comment documents the design contract using that name in prose.
    std::ifstream in(std::string(TEST_DATA_DIR) + "/src/finite_peclet_rhs_representer.cpp");
    REQUIRE(in.good());
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    REQUIRE(text.find("fp_radial_factor(") == std::string::npos);
    REQUIRE(text.find("fp_radial_factor<") == std::string::npos);
}

// ─── T-RP-08: throws without samples ────────────────────────────────────────

TEST_CASE(fp_representer_contract_throws_without_retained_samples)
{
    const auto modes = synthetic_block(0u, 10u, 1e-2);
    FPGramGaussJacobiOptions<double> gram_options; // retain_samples defaults to false
    const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, gram_options);
    REQUIRE(!factor.samples_retained);

    const std::vector<double> psi(factor.node_count, 1.0);
    bool threw = false;
    try
    {
        fp_representer_contract(factor, psi);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    REQUIRE(threw);
}

// ─── T-RP-09: conditioning diagnostic ───────────────────────────────────────

TEST_CASE(fp_representer_contract_cancellation_ratio_stays_bounded)
{
    FPRepresenterOptions<double> options;
    for (unsigned n : {0u, 20u, 60u, 95u})
    {
        const auto modes = synthetic_block(n, 45u, 1e-2);
        FPGramGaussJacobiOptions<double> gram_options;
        gram_options.retain_samples = true;
        gram_options.oversampling_factor = 3u;
        gram_options.oversampling_margin = 60u;
        const auto factor = fp_gram_factor_gauss_jacobi(n, modes, gram_options);

        const auto psi = fp_cap_representer<double>(n, 0.82, factor.node_count, factor.quadrature_nodes, options);
        const auto b = fp_representer_contract(factor, psi);

        for (unsigned m = 0; m < factor.mode_count; ++m)
        {
            double abs_sum = 0.0;
            for (unsigned q = 0; q < factor.node_count; ++q)
                abs_sum += std::abs(factor.quadrature_weights[q]
                    * factor.radial_samples[static_cast<std::size_t>(q) * factor.mode_count + m] * psi[q]);
            const double ratio = abs_sum / std::max(1e-300, std::abs(b[m]));
            REQUIRE(ratio < 1e5);
        }
    }
}
