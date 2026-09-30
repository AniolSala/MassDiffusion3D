// Inlet/boundary-condition-specific finite-Peclet load vectors in L2_r. See
// finite_peclet_rhs_radial.h for the design contract and for the derivation of
// exactly where omega drops out of Appendix B's cap reduction.

#include "finite_peclet_rhs_radial.h"

#include "finite_peclet_radial.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
    constexpr long double kSqrtTwoPi = 2.50662827463100050242L;
    constexpr long double kInvSqrtTwoPi = 0.39894228040143267794L;
    constexpr long double kInvSqrtPi = 0.56418958354775628695L;

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
            throw std::invalid_argument("finite-Peclet radial RHS: layer size mismatch");
        for (std::size_t i = 0; i < p.size(); ++i)
            if (!finite(p[i]) || !(p[i] > T(-1) && p[i] < T(1)) || (i && !(p[i] > p[i - 1])))
                throw std::invalid_argument("finite-Peclet radial RHS: invalid interfaces");
        for (T x : v)
            if (!finite(x))
                throw std::invalid_argument("finite-Peclet radial RHS: non-finite layer value");
    }

    template <typename T>
    T layer(const SeriesTermData<T> &m, const FPCapKernel<T> &c, T half)
    {
        if (m.n != c.angular_index)
            throw std::invalid_argument("finite-Peclet radial RHS: angular index mismatch");
        if (c.empty || c.zero)
            return T(0);
        if (c.center)
            return half;
        Sum<T> sum;
        for (std::size_t i = 0; i < c.s.size(); ++i)
        {
            T g = fp_radial_factor(m.n, m.root_fp, m.btilde_fp, c.s[i]);
            if (!finite(g))
                throw std::runtime_error("finite-Peclet radial RHS: non-finite radial factor");
            sum.add(c.kernel[i] * g);
        }
        T result = c.prefactor * sum.result();
        if (!finite(result))
            throw std::runtime_error("finite-Peclet radial RHS: non-finite layer integral");
        return result;
    }

    // The lower cap z < -a is the reflection of z > a: at n = 0 it is
    // full-minus-cap, and at n > 0 it is (-1)^(n+1) times the upper cap. Both
    // statements are purely geometric -- they hold for ANY radial weight -- so
    // this is identical to finite_peclet_rhs.cpp's own signed_cap.
    template <typename T>
    T signed_cap(unsigned n, T z, T pos, T full)
    {
        if (!(z < T(0)))
            return pos;
        if (n == 0)
            return full - pos;
        return n % 2 ? pos : -pos;
    }

    // Independent Golub--Welsch implementation for the (alpha=0, beta=0)
    // shifted-Legendre rule used by the r-weighted full-disk projection below.
    // Deliberately NOT shared with finite_peclet_gram_radial.cpp's own copy:
    // this module must never call into a Gram backend (design contract,
    // finite_peclet_rhs_radial.h), exactly as finite_peclet_rhs.cpp keeps its
    // own copy of the (1,0) rule.
    template <class T>
    void legendre_rule_ql(std::vector<T> &d, std::vector<T> &e, std::vector<T> &z)
    {
        const unsigned n = d.size();
        if (!n)
            return;
        e[n - 1] = 0;
        for (unsigned l = 0; l < n; ++l)
        {
            unsigned iter = 0, m;
            do
            {
                for (m = l; m + 1 < n; ++m)
                    if (std::abs(e[m]) <= std::numeric_limits<T>::epsilon() * (std::abs(d[m]) + std::abs(d[m + 1])))
                        break;
                if (m != l)
                {
                    if (++iter == 64)
                        throw std::runtime_error("radial full-disk projection rule: QL did not converge");
                    T g = (d[l + 1] - d[l]) / (2 * e[l]), r = std::hypot(g, T(1));
                    g = d[m] - d[l] + e[l] / (g + (g >= 0 ? std::abs(r) : -std::abs(r)));
                    T s = 1, c = 1, p = 0;
                    for (unsigned ii = m; ii-- > l;)
                    {
                        T f = s * e[ii], b = c * e[ii];
                        if (std::abs(f) >= std::abs(g))
                        {
                            c = g / f;
                            r = std::hypot(c, T(1));
                            e[ii + 1] = f * r;
                            s = T(1) / r;
                            c *= s;
                        }
                        else
                        {
                            s = f / g;
                            r = std::hypot(s, T(1));
                            e[ii + 1] = g * r;
                            c = T(1) / r;
                            s *= c;
                        }
                        g = d[ii + 1] - p;
                        r = (d[ii] - g) * s + 2 * c * b;
                        p = s * r;
                        d[ii + 1] = g + p;
                        g = c * r - b;
                        f = z[ii + 1];
                        z[ii + 1] = s * z[ii] + c * f;
                        z[ii] = c * z[ii] - s * f;
                    }
                    d[l] -= p;
                    e[l] = g;
                    e[m] = 0;
                }
            } while (m != l);
        }
    }

    // Shifted Legendre on [0,1]: measure ds, mass 1, d_k = 1/2,
    // e_{k-1} = k / (2 sqrt(4k^2 - 1)).
    template <class T>
    void legendre_rule(unsigned N, std::vector<T> &x, std::vector<T> &w)
    {
        if (!N)
            throw std::invalid_argument("radial full-disk projection rule: zero nodes");
        using L = long double;
        std::vector<L> d(N, 0.5L), e(N, 0), z(N, 0);
        for (unsigned k = 1; k < N; ++k)
            e[k - 1] = L(k) / (2 * std::sqrt(4 * L(k) * k - 1));
        const L mass = 1;
        z[0] = std::sqrt(mass);
        legendre_rule_ql(d, e, z);
        std::vector<std::pair<L, L>> pairs;
        pairs.reserve(N);
        for (unsigned i = 0; i < N; ++i)
            pairs.emplace_back(d[i], z[i] * z[i]);
        std::sort(pairs.begin(), pairs.end());
        x.resize(N);
        w.resize(N);
        L sum = 0;
        for (unsigned i = 0; i < N; ++i)
        {
            x[i] = static_cast<T>(pairs[i].first);
            w[i] = static_cast<T>(pairs[i].second);
            sum += pairs[i].second;
            if (!(x[i] > T(0) && x[i] < T(1) && w[i] > T(0)))
                throw std::runtime_error("radial full-disk projection rule: invalid node or weight");
        }
        const L tol = 256 * L(N) * std::numeric_limits<T>::epsilon() * mass;
        if (std::abs(sum - mass) > tol)
            throw std::runtime_error("radial full-disk projection rule: mass check failed");
    }

    // Groups `modes` by fp_rhs_required_order(modes[i].root, cap_quad_margin)
    // -- the SAME per-mode, root-driven bucketing the weighted path uses (the
    // bucket rule sizes a 1-D integral of one mode's own radial factor, which
    // the choice of radial weight does not make more oscillatory).
    template <typename T>
    void bucket_modes_by_required_order(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin,
                                        std::vector<unsigned> &bucket_of, std::vector<unsigned> &distinct_buckets)
    {
        bucket_of.resize(modes.size());
        distinct_buckets.clear();
        for (std::size_t i = 0; i < modes.size(); ++i)
        {
            bucket_of[i] = fp_rhs_required_order(modes[i].root, cap_quad_margin);
            if (std::find(distinct_buckets.begin(), distinct_buckets.end(), bucket_of[i]) == distinct_buckets.end())
                distinct_buckets.push_back(bucket_of[i]);
        }
    }

    // Throwing here (rather than silently building a fresh rule) makes a
    // stale/wrong cache a loud, immediate error instead of a silent reversion
    // to the per-call rebuild the cache exists to avoid.
    template <typename T>
    const FPGaussJacobiRule<T> &lookup_cached_rule(const std::vector<std::pair<unsigned, FPGaussJacobiRule<T>>> &cache, unsigned order)
    {
        for (const auto &entry : cache)
            if (entry.first == order)
                return entry.second;
        throw std::runtime_error("finite-Peclet radial RHS: rule cache missing required order "
                                  + std::to_string(order) + " (cache must be built from a superset "
                                  "of the current mode set, with the same cap_quad_margin)");
    }
}

template <typename T>
FPCapKernel<T> fp_rhs_radial_build_cap_kernel(unsigned n, const T &a, const FPGaussJacobiRule<T> &r)
{
    if (!finite(a) || a < T(0) || a > T(1))
        throw std::invalid_argument("fp_rhs_radial_build_cap_kernel: invalid absolute interface");
    FPCapKernel<T> c;
    c.angular_index = n;
    if (a == T(1))
    {
        c.empty = true;
        return c;
    }
    if (n == 0 && a == T(0))
    {
        c.center = true;
        return c;
    }
    if (a == T(0) && n > 0 && n % 2 == 0)
    {
        c.zero = true;
        return c;
    }
    T cc = a * a, Tm = T(1) - cc;
    c.s.resize(r.nodes.size());
    c.kernel.resize(r.nodes.size());
    if (n == 0)
    {
        // L2_omega has prefactor Tm*Tm and kernel factor (1 - eta); dropping
        // omega = Tm(1 - eta) removes exactly one Tm and that factor.
        c.prefactor = T(kInvSqrtTwoPi) * Tm;
        for (std::size_t i = 0; i < r.nodes.size(); ++i)
        {
            T eta = r.nodes[i];
            c.s[i] = cc + Tm * eta;
            c.kernel[i] = r.weights[i] * std::atan2(std::sqrt(Tm * eta), a) / std::sqrt(eta);
        }
    }
    else
    {
        // L2_omega has prefactor Tm*Tm*sqrt(Tm) and kernel factor (1 - eta).
        c.prefactor = T(kInvSqrtPi) / T(n) * Tm * std::sqrt(Tm);
        for (std::size_t i = 0; i < r.nodes.size(); ++i)
        {
            T s = cc + Tm * r.nodes[i], q = 1;
            if (n == 2)
                q = 2 * a;
            else if (n >= 3)
            {
                T qm = 1, q0 = 2 * a;
                for (unsigned j = 2; j <= n - 1; ++j)
                {
                    T qn = 2 * a * q0 - s * qm;
                    qm = q0;
                    q0 = qn;
                }
                q = q0;
            }
            c.s[i] = s;
            c.kernel[i] = r.weights[i] * q;
        }
    }
    if (!finite(c.prefactor))
        throw std::runtime_error("finite-Peclet radial RHS: non-finite cap prefactor");
    return c;
}

// See finite_peclet_rhs_radial.h: builds the (alpha,beta)=(0,0) shifted-Legendre
// cache consumed by fp_rhs_radial_full_disk_projection /
// fp_rhs_radial_uniform_inlet, one rule per distinct fp_rhs_required_order
// present in `modes`. NOT interchangeable with finite_peclet_rhs.h's caches.
template <typename T>
FPCapRuleCache<T> fp_rhs_radial_build_full_disk_rule_cache(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin)
{
    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);
    FPCapRuleCache<T> cache;
    cache.reserve(distinct_buckets.size());
    for (unsigned order : distinct_buckets)
    {
        FPGaussJacobiRule<T> rule;
        legendre_rule<T>(order, rule.nodes, rule.weights);
        cache.emplace_back(order, std::move(rule));
    }
    return cache;
}

// Full-disk projection sqrt(2pi) * int_0^1 R_{0,m}(r) r dr for every mode in an
// n=0 block, used for the baseline / center-cap / negative-interface-at-n=0 RHS
// terms (the "zero column" of U^0, since R_{0,0} == 1), obtained by independent
// quadrature -- NOT by forming or factoring any Gram matrix here. Modes are
// bucketed by their own required order; the rule for each distinct bucket
// present comes from `rule_cache` when supplied, or is built locally, once per
// distinct order, otherwise.
template <typename T>
std::vector<T> fp_rhs_radial_full_disk_projection(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin, const FPCapRuleCache<T> *rule_cache)
{
    FPCapRuleCache<T> local_cache;
    if (!rule_cache)
    {
        local_cache = fp_rhs_radial_build_full_disk_rule_cache(modes, cap_quad_margin);
        rule_cache = &local_cache;
    }

    std::vector<T> results(modes.size());
    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);
    for (unsigned order : distinct_buckets)
    {
        const auto &rule = lookup_cached_rule(*rule_cache, order);
        for (std::size_t m = 0; m < modes.size(); ++m)
        {
            if (bucket_of[m] != order)
                continue;
            // Compensated sum of (w_q/2) * G(s_q) ~= int_0^1 G(s) ds / 2
            //                                      = int_0^1 G(r^2) r dr   (s = r^2).
            Sum<T> sum;
            for (unsigned q = 0; q < order; ++q)
            {
                const T value = fp_radial_factor(0u, modes[m].root_fp, modes[m].btilde_fp, rule.nodes[q]);
                if (!finite(value))
                    throw std::runtime_error("fp_rhs_radial_full_disk_projection: non-finite radial factor");
                sum.add((rule.weights[q] / T(2)) * value);
            }
            const T result = T(kSqrtTwoPi) * sum.result();
            if (!finite(result))
                throw std::runtime_error("fp_rhs_radial_full_disk_projection: non-finite result");
            results[m] = result;
        }
    }
    return results;
}

template <typename T>
std::vector<T> fp_rhs_radial_full_disk_from_gram_column(const std::vector<T> &gram_zero_column)
{
    std::vector<T> result(gram_zero_column.size());
    for (std::size_t i = 0; i < gram_zero_column.size(); ++i)
    {
        result[i] = T(kSqrtTwoPi) * gram_zero_column[i];
        if (!finite(result[i]))
            throw std::runtime_error("fp_rhs_radial_full_disk_from_gram_column: non-finite result");
    }
    return result;
}

template <typename T>
std::vector<T> fp_rhs_radial_uniform_inlet(unsigned angular_index, const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin, const FPCapRuleCache<T> *full_disk_rule_cache)
{
    if (angular_index == 0u)
        return fp_rhs_radial_full_disk_projection(modes, cap_quad_margin, full_disk_rule_cache);
    return std::vector<T>(modes.size(), T(0));
}

template <typename T>
std::vector<T> fp_rhs_radial_stratified_inlet(unsigned angular_index, const std::vector<SeriesTermData<T>> &modes,
                                              const std::vector<T> &interface_positions, const std::vector<T> &layer_values,
                                              unsigned cap_quad_margin, const std::vector<T> &full_disk_column,
                                              const FPCapRuleCache<T> *cap_rule_cache, const FPCapRuleCache<T> *full_disk_rule_cache)
{
    validate_profile(interface_positions, layer_values);
    for (const auto &m : modes)
        if (m.n != angular_index)
            throw std::invalid_argument("fp_rhs_radial_stratified_inlet: mixed angular block");

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

    std::vector<T> full;
    if (angular_index == 0u)
    {
        if (!full_disk_column.empty())
        {
            if (full_disk_column.size() != modes.size())
                throw std::invalid_argument("fp_rhs_radial_stratified_inlet: full_disk_column size mismatch");
            full = full_disk_column;
        }
        else
            full = fp_rhs_radial_full_disk_projection(modes, cap_quad_margin, full_disk_rule_cache);
    }
    else
        full = std::vector<T>(modes.size(), T(0));

    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);

    // The (0,1/2) cap rule is measure-independent (it absorbs the geometric
    // 1/sqrt(eta) singularity, not omega), so finite_peclet_rhs.h's own cache
    // builder is reused verbatim here.
    FPCapRuleCache<T> local_cache;
    if (!cap_rule_cache)
    {
        local_cache = fp_rhs_build_rule_cache(modes, cap_quad_margin);
        cap_rule_cache = &local_cache;
    }

    std::vector<T> rhs(modes.size());
    for (unsigned order : distinct_buckets)
    {
        const auto &rule = lookup_cached_rule(*cap_rule_cache, order);
        std::vector<FPCapKernel<T>> caps;
        caps.reserve(abses.size());
        for (T a : abses)
            caps.push_back(fp_rhs_radial_build_cap_kernel(angular_index, a, rule));

        for (std::size_t local = 0; local < modes.size(); ++local)
        {
            if (bucket_of[local] != order)
                continue;
            Sum<T> s;
            s.add(layer_values.front() * full[local]);
            for (const auto &j : jumps)
            {
                T pos = layer(modes[local], caps[j.cap], full[local] / T(2));
                s.add(j.val * signed_cap(angular_index, j.z, pos, full[local]));
            }
            rhs[local] = s.result();
        }
    }
    return rhs;
}

template FPCapKernel<double> fp_rhs_radial_build_cap_kernel(unsigned, const double &, const FPGaussJacobiRule<double> &);
template FPCapKernel<long double> fp_rhs_radial_build_cap_kernel(unsigned, const long double &, const FPGaussJacobiRule<long double> &);
template FPCapRuleCache<double> fp_rhs_radial_build_full_disk_rule_cache(const std::vector<SeriesTermData<double>> &, unsigned);
template FPCapRuleCache<long double> fp_rhs_radial_build_full_disk_rule_cache(const std::vector<SeriesTermData<long double>> &, unsigned);
template std::vector<double> fp_rhs_radial_full_disk_projection(const std::vector<SeriesTermData<double>> &, unsigned, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_radial_full_disk_projection(const std::vector<SeriesTermData<long double>> &, unsigned, const FPCapRuleCache<long double> *);
template std::vector<double> fp_rhs_radial_full_disk_from_gram_column(const std::vector<double> &);
template std::vector<long double> fp_rhs_radial_full_disk_from_gram_column(const std::vector<long double> &);
template std::vector<double> fp_rhs_radial_uniform_inlet(unsigned, const std::vector<SeriesTermData<double>> &, unsigned, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_radial_uniform_inlet(unsigned, const std::vector<SeriesTermData<long double>> &, unsigned, const FPCapRuleCache<long double> *);
template std::vector<double> fp_rhs_radial_stratified_inlet(unsigned, const std::vector<SeriesTermData<double>> &, const std::vector<double> &, const std::vector<double> &, unsigned, const std::vector<double> &, const FPCapRuleCache<double> *, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_radial_stratified_inlet(unsigned, const std::vector<SeriesTermData<long double>> &, const std::vector<long double> &, const std::vector<long double> &, unsigned, const std::vector<long double> &, const FPCapRuleCache<long double> *, const FPCapRuleCache<long double> *);
