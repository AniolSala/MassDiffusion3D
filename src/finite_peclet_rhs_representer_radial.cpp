// Inlet-specific finite-Peclet load vectors in L2_r, representer route. See
// finite_peclet_rhs_representer_radial.h for the design contract and for what
// does and does not transfer from the weighted construction: this module NEVER
// calls fp_radial_factor -- avoiding those calls is its entire purpose.

#include "finite_peclet_rhs_representer_radial.h"
#include "shifted_jacobi_basis_radial.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    constexpr long double kSqrtTwoPi = 2.50662827463100050242L;

    template <typename T>
    class Sum
    {
        T s_ = 0, c_ = 0;

    public:
        void add(T x)
        {
            T y = s_ + x;
            c_ += std::abs(s_) >= std::abs(x) ? (s_ - y) + x : (x - y) + s_;
            s_ = y;
        }
        T result() const { return s_ + c_; }
    };
    template <typename T>
    bool finite(T x) { return std::isfinite(static_cast<long double>(x)); }

    template <typename T>
    void validate_profile(const std::vector<T> &p, const std::vector<T> &v)
    {
        if (v.size() != p.size() + 1)
            throw std::invalid_argument("finite-Peclet radial RHS representer: layer size mismatch");
        for (std::size_t i = 0; i < p.size(); ++i)
            if (!finite(p[i]) || !(p[i] > T(-1) && p[i] < T(1)) || (i && !(p[i] > p[i - 1])))
                throw std::invalid_argument("finite-Peclet radial RHS representer: invalid interfaces");
        for (T x : v)
            if (!finite(x))
                throw std::invalid_argument("finite-Peclet radial RHS representer: non-finite layer value");
    }

    // Purely geometric, hence identical to the weighted path's own copy.
    template <typename T>
    T signed_cap(unsigned n, T z, T pos, T full)
    {
        if (!(z < T(0)))
            return pos;
        if (n == 0)
            return full - pos;
        return n % 2 ? pos : -pos;
    }

    // Builds psi at a FIXED order (no order-rule lookup, no doubling guard --
    // that logic lives in fp_cap_representer_radial). is_center signals the
    // (n=0, a=0) degenerate case, for which no Jacobi expansion is built at all.
    template <typename T>
    std::vector<T> build_psi_at_order(unsigned angular_index, const T &a, unsigned J,
                                      const std::vector<T> &gram_nodes, unsigned order,
                                      bool &is_center)
    {
        is_center = false;
        const auto rule = make_fp_gauss_jacobi_rule<T>(order);
        // The (alpha, beta) = (0, 1/2) cap rule and this kernel builder are the
        // UNWEIGHTED cap objects; the rule itself is shared verbatim with the
        // weighted path because its eta^{1/2} weight is cap geometry.
        const auto kernel = fp_rhs_radial_build_cap_kernel<T>(angular_index, a, rule);
        if (kernel.center)
        {
            is_center = true;
            return std::vector<T>();
        }
        if (kernel.empty || kernel.zero)
            return std::vector<T>(gram_nodes.size(), T(0));

        const std::size_t Ns = kernel.s.size();
        const auto cap_table = shifted_jacobi_table_radial<T>(J, angular_index, kernel.s); // J x Ns

        std::vector<T> M(J, T(0));
        for (unsigned j = 0; j < J; ++j)
        {
            Sum<T> sum;
            for (std::size_t i = 0; i < Ns; ++i)
                sum.add(kernel.kernel[i] * cap_table[static_cast<std::size_t>(j) * Ns + i]);
            M[j] = kernel.prefactor * sum.result();
            if (!finite(M[j]))
                throw std::runtime_error("fp_cap_representer_radial: non-finite moment M_j");
        }

        std::vector<T> inv_h(J);
        for (unsigned j = 0; j < J; ++j)
            inv_h[j] = T(1) / shifted_jacobi_norm_radial<T>(j, angular_index);

        const std::size_t Q = gram_nodes.size();
        const auto psi_table = shifted_jacobi_table_radial<T>(J, angular_index, gram_nodes); // J x Q
        std::vector<T> psi(Q, T(0));
        for (std::size_t q = 0; q < Q; ++q)
        {
            Sum<T> sum;
            for (unsigned j = 0; j < J; ++j)
                sum.add((M[j] * inv_h[j]) * psi_table[static_cast<std::size_t>(j) * Q + q]);
            psi[q] = sum.result();
            if (!finite(psi[q]))
                throw std::runtime_error("fp_cap_representer_radial: non-finite representer value");
        }
        return psi;
    }
}

template <typename T>
unsigned fp_representer_order_radial(unsigned angular_index, unsigned coefficient_count, const T &absolute_interface,
                                     const FPRepresenterRadialOptions<T> &options)
{
    if (angular_index >= 1u)
    {
        // Exact: one degree below the weighted rule, the (1-eta) factor having
        // gone. Need 2*order - 1 >= (J-1) + floor((n-1)/2).
        const unsigned half = (angular_index - 1u) / 2u; // floor((n-1)/2)
        const unsigned long long inner = static_cast<unsigned long long>(coefficient_count) + half;
        const unsigned long long order = (inner + 1ull) / 2ull; // ceil(inner/2)
        return static_cast<unsigned>(order) + options.representer_margin;
    }
    // Calibrated; constant provisionally inherited from the weighted path (see
    // the header). The 1/a scaling itself is derived, not fitted.
    const T a_clamped = std::max(absolute_interface, static_cast<T>(0.01L));
    const unsigned long long term1 = (static_cast<unsigned long long>(coefficient_count) + 1ull) / 2ull; // ceil(J/2)
    const long double ratio = 6.0L / static_cast<long double>(a_clamped);
    const unsigned long long term2 = static_cast<unsigned long long>(std::ceil(ratio));
    return static_cast<unsigned>(term1 + term2) + options.representer_margin;
}

template <typename T>
std::vector<T> fp_cap_representer_radial(unsigned angular_index, const T &absolute_interface, unsigned coefficient_count,
                                         const std::vector<T> &gram_nodes, const FPRepresenterRadialOptions<T> &options)
{
    if (!finite(absolute_interface) || absolute_interface < T(0) || absolute_interface > T(1))
        throw std::invalid_argument("fp_cap_representer_radial: invalid absolute interface");
    if (gram_nodes.empty())
        throw std::invalid_argument("fp_cap_representer_radial: empty gram_nodes");

    const unsigned order = fp_representer_order_radial(angular_index, coefficient_count, absolute_interface, options);
    bool is_center = false;
    std::vector<T> psi = build_psi_at_order(angular_index, absolute_interface, coefficient_count, gram_nodes, order, is_center);

    // Mandatory doubling guard: n == 0 only, and only in the shallow-cap
    // regime, where the arctan kernel is analytic-but-not-polynomial and the
    // order rule's constant is a geometric-convergence estimate rather than an
    // exactness statement. On this path the constant is additionally
    // provisional (inherited from the weighted measure), so this guard is what
    // makes that inheritance safe: an insufficient order raises here instead of
    // silently returning a wrong load vector.
    if (angular_index == 0u && !is_center && absolute_interface < options.shallow_cap_threshold)
    {
        bool is_center2 = false;
        const std::vector<T> psi2 =
            build_psi_at_order(angular_index, absolute_interface, coefficient_count, gram_nodes, 2u * order, is_center2);
        T max_diff = 0, max_abs = 0;
        for (std::size_t q = 0; q < psi.size(); ++q)
        {
            max_diff = std::max(max_diff, std::abs(psi[q] - psi2[q]));
            max_abs = std::max(max_abs, std::abs(psi2[q]));
        }
        const T rel = max_diff / (max_abs > T(0) ? max_abs : T(1));
        if (!(rel <= options.shallow_cap_tolerance))
            throw std::runtime_error(
                "fp_cap_representer_radial: n=0 shallow-cap representer failed to converge under order "
                "doubling (relative difference " + std::to_string(static_cast<long double>(rel))
                + " exceeds shallow_cap_tolerance " + std::to_string(static_cast<long double>(options.shallow_cap_tolerance))
                + "); increase representer_margin, or re-calibrate the n=0 order constant for this measure");
    }
    return psi;
}

template <typename T>
std::vector<T> fp_representer_contract_radial(const FPGramRadialFactor<T> &factor, const std::vector<T> &psi)
{
    if (!factor.samples_retained)
        throw std::runtime_error("fp_representer_contract_radial: factor has no retained samples "
                                 "(build the Gram factor with retain_samples = true)");
    const unsigned K = factor.mode_count, N = factor.node_count;
    if (psi.size() != N)
        throw std::invalid_argument("fp_representer_contract_radial: psi size mismatch with Gram node count");
    if (factor.quadrature_weights.size() != N || factor.radial_samples.size() != static_cast<std::size_t>(N) * K)
        throw std::invalid_argument("fp_representer_contract_radial: malformed factor samples");

    std::vector<T> out(K, T(0));
    for (unsigned q = 0; q < N; ++q)
    {
        const T wp = factor.quadrature_weights[q] * psi[q];
        const T *row = &factor.radial_samples[static_cast<std::size_t>(q) * K];
        for (unsigned m = 0; m < K; ++m)
            out[m] += wp * row[m];
    }
    for (unsigned m = 0; m < K; ++m)
        if (!finite(out[m]))
            throw std::runtime_error("fp_representer_contract_radial: non-finite load-vector entry");
    return out;
}

template <typename T>
std::vector<T> fp_rhs_radial_stratified_inlet_representer(unsigned angular_index, const FPGramRadialFactor<T> &factor,
                                                          const std::vector<T> &interface_positions, const std::vector<T> &layer_values,
                                                          const FPRepresenterRadialOptions<T> &options, const std::vector<T> &full_disk_column)
{
    if (factor.angular_index != angular_index)
        throw std::invalid_argument("fp_rhs_radial_stratified_inlet_representer: factor angular_index mismatch");
    validate_profile(interface_positions, layer_values);
    const unsigned K = factor.mode_count;

    struct Jump
    {
        T z, val;
        std::size_t cap;
    };
    std::vector<T> abses;
    std::vector<Jump> jumps;
    for (std::size_t i = 0; i < interface_positions.size(); ++i)
        if (layer_values[i + 1] != layer_values[i])
        {
            T a = std::abs(interface_positions[i]);
            auto it = std::find(abses.begin(), abses.end(), a);
            std::size_t ci;
            if (it == abses.end())
            {
                ci = abses.size();
                abses.push_back(a);
            }
            else
                ci = it - abses.begin();
            jumps.push_back({interface_positions[i], layer_values[i + 1] - layer_values[i], ci});
        }

    // Full-disk term: exact Gram-column identity when supplied, otherwise the
    // constant representer -- two exact identities, not a fallback. The
    // constant is sqrt(2 pi)/2 on BOTH paths: contracting it against this
    // measure gives sqrt(2 pi)/2 * int_0^1 G ds, and the true r-weighted disk
    // term is sqrt(2 pi) * int_0^1 R_{0m} r dr = sqrt(2 pi)/2 * int_0^1 G ds.
    std::vector<T> full;
    if (angular_index == 0u)
    {
        if (!full_disk_column.empty())
        {
            if (full_disk_column.size() != K)
                throw std::invalid_argument("fp_rhs_radial_stratified_inlet_representer: full_disk_column size mismatch");
            full = full_disk_column;
        }
        else
        {
            const T psi_const = static_cast<T>(kSqrtTwoPi) / T(2);
            const std::vector<T> psi_full(factor.node_count, psi_const);
            full = fp_representer_contract_radial(factor, psi_full);
        }
    }
    else
        full = std::vector<T>(K, T(0));

    // One representer build (plus one K*N contraction) per DISTINCT |z_i| --
    // never one per mode, and never repeated for repeated interface magnitude.
    std::vector<std::vector<T>> cap_contrib(abses.size());
    for (std::size_t ci = 0; ci < abses.size(); ++ci)
    {
        const auto psi = fp_cap_representer_radial<T>(angular_index, abses[ci], factor.node_count, factor.quadrature_nodes, options);
        cap_contrib[ci] = psi.empty() ? std::vector<T>() : fp_representer_contract_radial(factor, psi);
    }

    std::vector<T> rhs(K, T(0));
    for (unsigned m = 0; m < K; ++m)
    {
        Sum<T> s;
        s.add(layer_values.front() * full[m]);
        for (const auto &j : jumps)
        {
            const T pos = cap_contrib[j.cap].empty() ? full[m] / T(2) : cap_contrib[j.cap][m];
            s.add(j.val * signed_cap(angular_index, j.z, pos, full[m]));
        }
        rhs[m] = s.result();
        if (!finite(rhs[m]))
            throw std::runtime_error("fp_rhs_radial_stratified_inlet_representer: non-finite load-vector entry");
    }
    return rhs;
}

template <typename T>
std::vector<T> fp_rhs_radial_uniform_inlet_representer(unsigned angular_index, const FPGramRadialFactor<T> &factor,
                                                       const FPRepresenterRadialOptions<T> &options, const std::vector<T> &full_disk_column)
{
    (void)options;
    if (factor.angular_index != angular_index)
        throw std::invalid_argument("fp_rhs_radial_uniform_inlet_representer: factor angular_index mismatch");
    const unsigned K = factor.mode_count;
    if (angular_index != 0u)
        return std::vector<T>(K, T(0));
    if (!full_disk_column.empty())
    {
        if (full_disk_column.size() != K)
            throw std::invalid_argument("fp_rhs_radial_uniform_inlet_representer: full_disk_column size mismatch");
        return full_disk_column;
    }
    const T psi_const = static_cast<T>(kSqrtTwoPi) / T(2);
    const std::vector<T> psi_full(factor.node_count, psi_const);
    return fp_representer_contract_radial(factor, psi_full);
}

template unsigned fp_representer_order_radial(unsigned, unsigned, const double &, const FPRepresenterRadialOptions<double> &);
template unsigned fp_representer_order_radial(unsigned, unsigned, const long double &, const FPRepresenterRadialOptions<long double> &);
template std::vector<double> fp_cap_representer_radial(unsigned, const double &, unsigned, const std::vector<double> &, const FPRepresenterRadialOptions<double> &);
template std::vector<long double> fp_cap_representer_radial(unsigned, const long double &, unsigned, const std::vector<long double> &, const FPRepresenterRadialOptions<long double> &);
template std::vector<double> fp_representer_contract_radial(const FPGramRadialFactor<double> &, const std::vector<double> &);
template std::vector<long double> fp_representer_contract_radial(const FPGramRadialFactor<long double> &, const std::vector<long double> &);
template std::vector<double> fp_rhs_radial_stratified_inlet_representer(unsigned, const FPGramRadialFactor<double> &, const std::vector<double> &, const std::vector<double> &, const FPRepresenterRadialOptions<double> &, const std::vector<double> &);
template std::vector<long double> fp_rhs_radial_stratified_inlet_representer(unsigned, const FPGramRadialFactor<long double> &, const std::vector<long double> &, const std::vector<long double> &, const FPRepresenterRadialOptions<long double> &, const std::vector<long double> &);
template std::vector<double> fp_rhs_radial_uniform_inlet_representer(unsigned, const FPGramRadialFactor<double> &, const FPRepresenterRadialOptions<double> &, const std::vector<double> &);
template std::vector<long double> fp_rhs_radial_uniform_inlet_representer(unsigned, const FPGramRadialFactor<long double> &, const FPRepresenterRadialOptions<long double> &, const std::vector<long double> &);
