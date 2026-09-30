// Gaussian finite-Peclet coefficients.
// 1 validate and group modes; 2 cache one angular block at Gaussian nodes;
// 3 assemble its fixed-weight Gram matrix; 4 project inlet data; 5 solve the
// equilibrated SPD system.  Cap nodes are separate local geometry, never a
// global finite-Peclet radial table.

#include "finite_peclet_coefficients_gaussian.h"

#include "finite_peclet_radial.h"
#include "finite_peclet_gram_gauss_jacobi.h"
#include "finite_peclet_gram_ultraspherical.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_rhs_representer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

namespace
{
constexpr long double kSqrtTwoPi = 2.50662827463100050242L;
constexpr long double kInvSqrtTwoPi = 0.39894228040143267794L;
constexpr long double kInvSqrtPi = 0.56418958354775628695L;

// Neumaier summation preserves small Gram and projection contributions.
template <typename Ttype> class CompensatedSum {
public:
    void add(const Ttype &value) noexcept {
        const Ttype next = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) correction_ += (sum_ - next) + value;
        else correction_ += (value - next) + sum_;
        sum_ = next;
    }
    Ttype result() const noexcept { return sum_ + correction_; }
private: Ttype sum_ = static_cast<Ttype>(0), correction_ = static_cast<Ttype>(0);
};

template <typename Ttype>
void validate_gaussian_rule(const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes,
                            const char *caller)
{
    if (weights.empty() || nodes.empty()) throw std::invalid_argument(std::string(caller) + ": empty Gaussian rule");
    if (weights.size() != nodes.size()) throw std::invalid_argument(std::string(caller) + ": Gaussian size mismatch");
    bool increasing = true, decreasing = true;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
        if (!std::isfinite(static_cast<long double>(weights[q])) || !std::isfinite(static_cast<long double>(nodes[q])))
            throw std::runtime_error(std::string(caller) + ": non-finite Gaussian value");
        if (!(weights[q] > static_cast<Ttype>(0))) throw std::runtime_error(std::string(caller) + ": non-positive Gaussian weight");
        if (!(nodes[q] > static_cast<Ttype>(0) && nodes[q] < static_cast<Ttype>(1))) throw std::runtime_error(std::string(caller) + ": Gaussian node outside (0,1)");
        if (q) { increasing = increasing && nodes[q] > nodes[q - 1]; decreasing = decreasing && nodes[q] < nodes[q - 1]; }
    }
    // Repository Gaussian tables are stored from one to zero.  Quadrature sums
    // are permutation-invariant, so accept either strict table order but reject
    // genuinely unsorted input.
    if (!increasing && !decreasing) throw std::runtime_error(std::string(caller) + ": Gaussian nodes are not strictly ordered");
}

template <typename Ttype>
std::vector<std::vector<unsigned>> group_active_modes_by_angular_index(unsigned max_K, const std::vector<SeriesTermData<Ttype>> &data)
{
    if (max_K > data.size()) throw std::invalid_argument("finite-Peclet Gaussian coefficients: max_K exceeds series data");
    if (!max_K) return {};
    unsigned highest_n = 0;
    for (unsigned k = 0; k < max_K; ++k) {
        const auto &term = data[k];
        if (!std::isfinite(static_cast<long double>(term.root_fp)) || !(term.root_fp >= static_cast<Ttype>(0)) ||
            !std::isfinite(static_cast<long double>(term.btilde_fp)) || !(term.btilde_fp >= static_cast<Ttype>(0)) ||
            !std::isfinite(static_cast<long double>(term.rate_fp)) || !(term.rate_fp >= static_cast<Ttype>(0)))
            throw std::runtime_error("finite-Peclet Gaussian coefficients: invalid finite-Peclet mode data");
        highest_n = std::max(highest_n, term.n);
    }
    std::vector<std::vector<unsigned>> groups(highest_n + 1);
    for (unsigned k = 0; k < max_K; ++k) groups[data[k].n].push_back(k);
    for (const auto &group : groups) for (std::size_t i = 1; i < group.size(); ++i) {
        const auto &previous = data[group[i - 1]]; const auto &current = data[group[i]];
        if (!(current.m > previous.m)) throw std::runtime_error("finite-Peclet Gaussian coefficients: non-increasing or duplicate (n,m)");
    }
    return groups;
}

// Owns temporary values for one angular family; radial_values is row-major.
template <typename Ttype> struct GaussianCoefficientBlock {
    unsigned angular_index = 0; std::size_t mode_count = 0, node_count = 0;
    std::vector<unsigned> flat_indices; std::vector<Ttype> radial_values;
    std::vector<std::vector<Ttype>> gram_matrix; std::vector<Ttype> right_hand_side, coefficients, full_disk_projection;
};

template <typename Ttype>
void build_radial_values(const std::vector<SeriesTermData<Ttype>> &data, const std::vector<unsigned> &indices,
                         const std::vector<Ttype> &nodes, bool parallel_modes, GaussianCoefficientBlock<Ttype> &block)
{
    block.flat_indices = indices; block.mode_count = indices.size(); block.node_count = nodes.size();
    if (block.mode_count > block.node_count) throw std::runtime_error("finite-Peclet Gaussian coefficients: angular block has more modes than Gaussian nodes");
    block.radial_values.assign(block.mode_count * block.node_count, static_cast<Ttype>(0));
    std::exception_ptr failure;
    const auto work = [&](std::size_t local) {
        try { const auto &term = data[indices[local]]; for (std::size_t q = 0; q < nodes.size(); ++q) {
            const Ttype value = psinm_r_fp(term.n, term.root_fp, term.btilde_fp, nodes[q]);
            if (!std::isfinite(static_cast<long double>(value))) { std::ostringstream message; message << "non-finite radial value n=" << term.n << " m=" << term.m << " root_fp=" << term.root_fp << " btilde_fp=" << term.btilde_fp << " node=" << q << " value=" << nodes[q]; throw std::runtime_error(message.str()); }
            block.radial_values[local * block.node_count + q] = value;
        }} catch (...) {
#pragma omp critical
            { if (!failure) failure = std::current_exception(); }
        }
    };
    if (parallel_modes) {
#pragma omp parallel for schedule(static)
        for (long long local = 0; local < static_cast<long long>(block.mode_count); ++local) work(static_cast<std::size_t>(local));
    } else for (std::size_t local = 0; local < block.mode_count; ++local) work(local);
    if (failure) std::rethrow_exception(failure);
}

template <typename Ttype>
void build_gaussian_gram_matrix(const std::vector<Ttype> &weights, GaussianCoefficientBlock<Ttype> &block)
{
    block.gram_matrix.assign(block.mode_count, std::vector<Ttype>(block.mode_count, static_cast<Ttype>(0)));
    for (std::size_t i = 0; i < block.mode_count; ++i) for (std::size_t j = i; j < block.mode_count; ++j) {
        CompensatedSum<Ttype> sum; for (std::size_t q = 0; q < block.node_count; ++q) sum.add(weights[q] * block.radial_values[i * block.node_count + q] * block.radial_values[j * block.node_count + q]);
        const Ttype entry = sum.result(); if (!std::isfinite(static_cast<long double>(entry))) throw std::runtime_error("finite-Peclet Gaussian coefficients: non-finite Gram entry");
        block.gram_matrix[i][j] = block.gram_matrix[j][i] = entry;
    }
    for (std::size_t i = 0; i < block.mode_count; ++i) if (!(block.gram_matrix[i][i] > static_cast<Ttype>(0))) throw std::runtime_error("finite-Peclet Gaussian coefficients: non-positive Gram diagonal");
}

template <typename Ttype>
void solve_equilibrated_gram_system(std::vector<std::vector<Ttype>> matrix, std::vector<Ttype> rhs, std::vector<Ttype> &coefficients)
{
    const std::size_t count = matrix.size(); if (rhs.size() != count) throw std::invalid_argument("finite-Peclet Gaussian coefficients: Gram RHS size mismatch");
    std::vector<Ttype> diagonal(count);
    std::vector<std::vector<Ttype>> lower(count, std::vector<Ttype>(count, static_cast<Ttype>(0)));
    for (std::size_t i = 0; i < count; ++i) { if (matrix[i].size() != count || !(matrix[i][i] > static_cast<Ttype>(0))) throw std::runtime_error("finite-Peclet Gaussian coefficients: invalid Gram matrix"); diagonal[i] = std::sqrt(matrix[i][i]); }
    for (std::size_t i = 0; i < count; ++i) { rhs[i] /= diagonal[i]; for (std::size_t j = 0; j < count; ++j) matrix[i][j] /= diagonal[i] * diagonal[j]; }
    for (std::size_t i = 0; i < count; ++i) for (std::size_t j = 0; j <= i; ++j) { Ttype value = matrix[i][j]; for (std::size_t k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k]; if (i == j) { if (!(value > static_cast<Ttype>(0))) throw std::runtime_error("finite-Peclet Gaussian coefficients: Gram Cholesky failed (insufficient resolution or numerical dependence)"); lower[i][j] = std::sqrt(value); } else lower[i][j] = value / lower[j][j]; }
    for (std::size_t i = 0; i < count; ++i) { for (std::size_t k = 0; k < i; ++k) rhs[i] -= lower[i][k] * rhs[k]; rhs[i] /= lower[i][i]; }
    for (std::size_t i = count; i-- > 0;) { for (std::size_t k = i + 1; k < count; ++k) rhs[i] -= lower[k][i] * rhs[k]; rhs[i] /= lower[i][i]; }
    coefficients.resize(count); for (std::size_t i = 0; i < count; ++i) { coefficients[i] = rhs[i] / diagonal[i]; if (!std::isfinite(static_cast<long double>(coefficients[i]))) throw std::runtime_error("finite-Peclet Gaussian coefficients: non-finite coefficient"); }
}

template <typename Ttype>
void build_full_disk_projection(const std::vector<Ttype> &weights, GaussianCoefficientBlock<Ttype> &block)
{
    block.full_disk_projection.assign(block.mode_count, static_cast<Ttype>(0)); if (block.angular_index != 0) return;
    for (std::size_t i = 0; i < block.mode_count; ++i) { CompensatedSum<Ttype> sum; for (std::size_t q = 0; q < block.node_count; ++q) sum.add(weights[q] * block.radial_values[i * block.node_count + q]); block.full_disk_projection[i] = static_cast<Ttype>(kSqrtTwoPi) * sum.result(); }
}

template <typename Ttype> struct PositiveCapQuadratureData { Ttype absolute_interface{}; std::vector<Ttype> radial_nodes, angular_coordinates, weighted_jacobian; };
template <typename Ttype>
PositiveCapQuadratureData<Ttype> build_positive_cap(const Ttype &absolute_interface, const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes)
{
    if (!(absolute_interface >= static_cast<Ttype>(0) && absolute_interface < static_cast<Ttype>(1))) throw std::invalid_argument("finite-Peclet Gaussian coefficients: invalid interface");
    PositiveCapQuadratureData<Ttype> cap; cap.absolute_interface = absolute_interface; const Ttype complement = static_cast<Ttype>(1) - absolute_interface * absolute_interface, limit = std::sqrt(complement), jacobian = static_cast<Ttype>(2) * complement * complement;
    for (std::size_t q = 0; q < nodes.size(); ++q) { const Ttype transverse = nodes[q] * limit; cap.radial_nodes.push_back(std::sqrt(transverse * transverse + absolute_interface * absolute_interface)); cap.angular_coordinates.push_back(std::atan2(transverse, absolute_interface)); cap.weighted_jacobian.push_back(weights[q] * jacobian); }
    return cap;
}
template <typename Ttype>
void build_cap_angular_weights(unsigned n, const PositiveCapQuadratureData<Ttype> &cap, std::vector<Ttype> &angular_weights)
{
    angular_weights.resize(cap.radial_nodes.size()); for (std::size_t q = 0; q < angular_weights.size(); ++q) angular_weights[q] = cap.weighted_jacobian[q] * (n == 0 ? cap.angular_coordinates[q] * static_cast<Ttype>(kInvSqrtTwoPi) : std::sin(static_cast<Ttype>(n) * cap.angular_coordinates[q]) * static_cast<Ttype>(kInvSqrtPi) / static_cast<Ttype>(n));
}
template <typename Ttype>
Ttype positive_cap_projection(const SeriesTermData<Ttype> &term, const PositiveCapQuadratureData<Ttype> &cap, const std::vector<Ttype> &angular_weights)
{
    CompensatedSum<Ttype> sum; for (std::size_t q = 0; q < cap.radial_nodes.size(); ++q) { const Ttype value = psinm_r_fp(term.n, term.root_fp, term.btilde_fp, cap.radial_nodes[q]); if (!std::isfinite(static_cast<long double>(value))) throw std::runtime_error("finite-Peclet Gaussian coefficients: non-finite cap radial value"); sum.add(value * angular_weights[q]); } return sum.result();
}
// The lower cap z < -a is reflected from z > a.  Thus n=0 uses full-minus-cap,
// while n>0 obtains (-1)^(n+1) times the upper cap.
template <typename Ttype>
Ttype signed_cap_projection(unsigned n, const Ttype &interface, const Ttype &positive, const Ttype &full)
{ if (!(interface < static_cast<Ttype>(0))) return positive; if (n == 0) return full - positive; return (n % 2) ? positive : -positive; }

template <typename Ttype>
void solve_graetz_block(std::vector<SeriesTermData<Ttype>> &data, const std::vector<unsigned> &indices,
    GramMethod gram_method, WallCondition wall_condition, const FPGramGaussJacobiOptions<Ttype> &gram_options, unsigned cap_quad_margin,
    RhsMethod rhs_method, unsigned rhs_representer_margin, bool enable_timing, Ttype *projection_square_norm)
{
    // Representer requires the Gram-assembled radial samples, which only the
    // GaussJacobiQR factor retains. No fallback: checked before the Gram pass
    // runs, mirroring the stratified driver's own pre-Pass-1 check.
    if (rhs_method == RhsMethod::Representer && gram_method != GramMethod::GaussJacobiQR)
        throw std::invalid_argument("fp_graetz_coefficients_gaussian: RhsMethod::Representer requires "
                                    "GramMethod::GaussJacobiQR (the ultraspherical factor retains no "
                                    "samples to contract against)");
    std::vector<SeriesTermData<Ttype>> modes; modes.reserve(indices.size()); for (unsigned id : indices) modes.push_back(data[id]);
    // cap_quad_margin is forwarded unmodified to finite_peclet_rhs.h, which
    // sizes each mode's own quadrature order from that mode's own bare root
    // (fp_rhs_required_order) -- see finite_peclet_coefficients_gauss_jacobi.h
    // for why this is no longer sized off the block's mode count. Consulted
    // only on the DirectQuadrature path; Representer needs no per-mode rule.
    // Gram first, then RHS, then solve (matches the stratified driver's pass
    // ordering, finite_peclet_coefficients_gauss_jacobi.cpp): the RHS needs to
    // know whether the Gram assembly recorded an exact constant-mode column
    // before it can choose between fp_rhs_full_disk_from_gram_column (exact,
    // GaussJacobiQR + Neumann + Lambda=0 mode present) and the independent
    // quadrature / constant representer, everywhere else. Graetz has a single
    // n=0 block and no parallel loop over blocks, so these timings are plain
    // wall-clock, not a parallel sum.
    std::vector<Ttype> rhs;
    std::vector<Ttype> coefficients;
    const auto gram_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    switch (gram_method)
    {
    case GramMethod::GaussJacobiQR:
    {
        FPGramGaussJacobiOptions<Ttype> gram_options_local = gram_options;
        gram_options_local.retain_samples = (rhs_method == RhsMethod::Representer);
        const auto factor = fp_gram_factor_gauss_jacobi(0u, modes, gram_options_local);
        if (enable_timing)
            std::cout << "Timing: finite-Peclet Gram matrix assembly (graetz) took "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - gram_t0).count() << " s." << std::endl;
        const auto rhs_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        // This is a mathematical identity selection, not a backend fallback: the
        // exact Gram-column formula is used whenever it applies, otherwise the
        // independent quadrature / representer is used -- there is still no
        // retry across Gram backends.
        if (factor.has_constant_mode)
            rhs = fp_rhs_full_disk_from_gram_column(factor.constant_mode_gram_column);
        else if (rhs_method == RhsMethod::DirectQuadrature)
            rhs = fp_rhs_uniform_inlet(0u, modes, cap_quad_margin);
        else
        {
            FPRepresenterOptions<Ttype> representer_options;
            representer_options.representer_margin = rhs_representer_margin;
            rhs = fp_rhs_uniform_inlet_representer(0u, factor, representer_options);
        }
        if (enable_timing)
            std::cout << "Timing: finite-Peclet RHS vector assembly (graetz) took "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_t0).count() << " s." << std::endl;
        fp_gram_solve_gauss_jacobi(factor, rhs, coefficients);
        break;
    }
    case GramMethod::Ultraspherical:
    {
        FPGramUltrasphericalOptions<Ttype> us_options;
        us_options.ultraspherical_factor = gram_options.oversampling_factor;
        us_options.ultraspherical_margin = gram_options.oversampling_margin;
        us_options.enable_order_check = gram_options.enable_order_check;
        us_options.minimum_factor = gram_options.minimum_factor;
        const auto factor = fp_gram_factor_ultraspherical(0u, modes, wall_condition, us_options);
        if (enable_timing)
            std::cout << "Timing: finite-Peclet Gram matrix assembly (graetz) took "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - gram_t0).count() << " s." << std::endl;
        const auto rhs_t0 = enable_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        rhs = fp_rhs_uniform_inlet(0u, modes, cap_quad_margin);
        if (enable_timing)
            std::cout << "Timing: finite-Peclet RHS vector assembly (graetz) took "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_t0).count() << " s." << std::endl;
        fp_gram_solve_ultraspherical(factor, rhs, coefficients);
        break;
    }
    default:
        throw std::invalid_argument("fp_graetz_coefficients_gaussian: unhandled Gram method");
    }
    // chat^T b == chat^T W chat == || f_K ||^2, free now that the solve has run.
    Ttype inlet_square_norm = 0;
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        data[indices[i]].coeff_fp = coefficients[i];
        inlet_square_norm += coefficients[i] * rhs[i];
    }
    if (projection_square_norm) *projection_square_norm = inlet_square_norm;
}
}

template <typename Ttype>
void fp_graetz_coefficients_gaussian(unsigned max_K, std::vector<SeriesTermData<Ttype>> &data,
    GramMethod gram_method, WallCondition wall_condition, const FPGramGaussJacobiOptions<Ttype> &gram_options, unsigned cap_quad_margin,
    RhsMethod rhs_method, unsigned rhs_representer_margin, bool enable_timing, Ttype *projection_square_norm)
{
    if (!max_K) return; auto groups = group_active_modes_by_angular_index(max_K, data); for (unsigned k = 0; k < max_K; ++k) data[k].coeff_fp = std::numeric_limits<Ttype>::quiet_NaN();
    if (groups.empty() || groups[0].empty()) throw std::runtime_error("fp_graetz_coefficients_gaussian: missing n=0 group");
    for (std::size_t n = 1; n < groups.size(); ++n) for (unsigned index : groups[n]) data[index].coeff_fp = static_cast<Ttype>(0);
    solve_graetz_block(data, groups[0], gram_method, wall_condition, gram_options, cap_quad_margin, rhs_method, rhs_representer_margin, enable_timing, projection_square_norm);
    for (unsigned k = 0; k < max_K; ++k) if (!std::isfinite(static_cast<long double>(data[k].coeff_fp))) throw std::runtime_error("fp_graetz_coefficients_gaussian: unwritten coefficient");
}

template <typename Ttype>
void fp_stratified_coefficients_gaussian(const std::vector<Ttype> &interfaces, const std::vector<Ttype> &layers, unsigned max_K, std::vector<SeriesTermData<Ttype>> &data, const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes)
{
    if (!max_K) return; validate_gaussian_rule(weights, nodes, "fp_stratified_coefficients_gaussian"); if (layers.size() != interfaces.size() + 1) throw std::invalid_argument("fp_stratified_coefficients_gaussian: layer size mismatch");
    for (std::size_t i = 0; i < interfaces.size(); ++i) { if (!std::isfinite(static_cast<long double>(interfaces[i])) || !(interfaces[i] > static_cast<Ttype>(-1) && interfaces[i] < static_cast<Ttype>(1)) || (i && !(interfaces[i] > interfaces[i-1]))) throw std::invalid_argument("fp_stratified_coefficients_gaussian: invalid interfaces"); }
    for (const Ttype &value : layers) if (!std::isfinite(static_cast<long double>(value))) throw std::invalid_argument("fp_stratified_coefficients_gaussian: non-finite layer value");
    auto groups = group_active_modes_by_angular_index(max_K, data); for (unsigned k = 0; k < max_K; ++k) data[k].coeff_fp = std::numeric_limits<Ttype>::quiet_NaN();
    struct Jump { Ttype position, value; std::size_t cap; }; std::vector<Jump> jumps; std::vector<PositiveCapQuadratureData<Ttype>> caps;
    for (std::size_t i = 0; i < interfaces.size(); ++i) if (layers[i+1] != layers[i]) { const Ttype absolute = std::abs(interfaces[i]); std::size_t cap = 0; while (cap < caps.size() && caps[cap].absolute_interface != absolute) ++cap; if (cap == caps.size()) caps.push_back(build_positive_cap(absolute, weights, nodes)); jumps.push_back({interfaces[i], layers[i+1]-layers[i], cap}); }
    std::vector<std::vector<Ttype>> results(groups.size()); std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic)
    for (long long raw_n = 0; raw_n < static_cast<long long>(groups.size()); ++raw_n) try {
        const std::size_t n = static_cast<std::size_t>(raw_n); if (groups[n].empty()) continue; GaussianCoefficientBlock<Ttype> block; block.angular_index = static_cast<unsigned>(n); build_radial_values(data, groups[n], nodes, false, block); build_gaussian_gram_matrix(weights, block); build_full_disk_projection(weights, block); block.right_hand_side.resize(block.mode_count);
        for (std::size_t local = 0; local < block.mode_count; ++local) { CompensatedSum<Ttype> rhs; rhs.add(layers.front() * block.full_disk_projection[local]); for (const Jump &jump : jumps) { std::vector<Ttype> angular_weights; build_cap_angular_weights(static_cast<unsigned>(n), caps[jump.cap], angular_weights); const Ttype positive = positive_cap_projection(data[groups[n][local]], caps[jump.cap], angular_weights); rhs.add(jump.value * signed_cap_projection(static_cast<unsigned>(n), jump.position, positive, block.full_disk_projection[local])); } block.right_hand_side[local] = rhs.result(); }
        solve_equilibrated_gram_system(block.gram_matrix, block.right_hand_side, results[n]);
    } catch (...) {
#pragma omp critical
        { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure); for (std::size_t n = 0; n < groups.size(); ++n) for (std::size_t local = 0; local < groups[n].size(); ++local) data[groups[n][local]].coeff_fp = results[n][local];
    for (unsigned k = 0; k < max_K; ++k) if (!std::isfinite(static_cast<long double>(data[k].coeff_fp))) throw std::runtime_error("fp_stratified_coefficients_gaussian: unwritten coefficient");
}

template void fp_graetz_coefficients_gaussian<double>(unsigned, std::vector<SeriesTermData<double>> &, GramMethod, WallCondition, const FPGramGaussJacobiOptions<double> &, unsigned, RhsMethod, unsigned, bool, double *);
template void fp_graetz_coefficients_gaussian<long double>(unsigned, std::vector<SeriesTermData<long double>> &, GramMethod, WallCondition, const FPGramGaussJacobiOptions<long double> &, unsigned, RhsMethod, unsigned, bool, long double *);
template void fp_stratified_coefficients_gaussian<double>(const std::vector<double> &, const std::vector<double> &, unsigned, std::vector<SeriesTermData<double>> &, const std::vector<double> &, const std::vector<double> &);
template void fp_stratified_coefficients_gaussian<long double>(const std::vector<long double> &, const std::vector<long double> &, unsigned, std::vector<SeriesTermData<long double>> &, const std::vector<long double> &, const std::vector<long double> &);
