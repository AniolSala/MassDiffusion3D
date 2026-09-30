// Inlet/boundary-condition-specific finite-Peclet load vectors. See
// finite_peclet_rhs.h for the design contract (inlet-specific code lives here,
// never in finite_peclet_gram_*).

#include "finite_peclet_rhs.h"

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

    /* Symmetric tridiagonal QL, also transforming z (IMTQLX). Used to build the
     * (alpha=0, beta=1/2) cap rule -- kept local to this module, independent of
     * the Gram backend's own Golub-Welsch implementation. */
    void tridiagonal_ql(std::vector<long double> &d, std::vector<long double> &e,
                        std::vector<long double> &z)
    {
        const int n = static_cast<int>(d.size());
        if (n == 1)
            return;
        e[n - 1] = 0;
        const long double eps = std::numeric_limits<long double>::epsilon();
        for (int l = 0; l < n; ++l)
        {
            int iterations = 0, m;
            do
            {
                for (m = l; m < n - 1; ++m)
                    if (std::abs(e[m]) <= eps * (std::abs(d[m]) + std::abs(d[m + 1])))
                        break;
                if (m != l)
                {
                    if (++iterations > 64)
                    {
                        std::ostringstream os;
                        os << "Gauss-Jacobi QL failed at index " << l << " for rule size " << n;
                        throw std::runtime_error(os.str());
                    }
                    long double g = (d[l + 1] - d[l]) / (2 * e[l]);
                    long double r = std::hypot(g, 1.0L);
                    g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
                    long double s = 1, c = 1, p = 0;
                    for (int i = m - 1; i >= l; --i)
                    {
                        long double f = s * e[i], b = c * e[i];
                        r = std::hypot(f, g);
                        e[i + 1] = r;
                        if (r == 0)
                        {
                            d[i + 1] -= p;
                            e[m] = 0;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i + 1] - p;
                        r = (d[i] - g) * s + 2 * c * b;
                        p = s * r;
                        d[i + 1] = g + p;
                        g = c * r - b;
                        const long double zi = z[i + 1];
                        z[i + 1] = s * z[i] + c * zi;
                        z[i] = c * z[i] - s * zi;
                    }
                    if (r == 0 && m - 1 >= l)
                        continue;
                    d[l] -= p;
                    e[l] = g;
                    e[m] = 0;
                }
            } while (m != l);
        }
    }

    template <typename T>
    void validate_rule(const FPGaussJacobiRule<T> &r, const char *who)
    {
        if (r.nodes.empty() || r.nodes.size() != r.weights.size())
            throw std::invalid_argument(std::string(who) + ": malformed Gauss-Jacobi rule");
        for (std::size_t i = 0; i < r.nodes.size(); ++i)
        {
            if (!finite(r.nodes[i]) || !finite(r.weights[i]) || !(r.nodes[i] > T(0) && r.nodes[i] < T(1)) || !(r.weights[i] > T(0)))
                throw std::runtime_error(std::string(who) + ": invalid Gauss-Jacobi rule");
            if (i && !(r.nodes[i] > r.nodes[i - 1]))
                throw std::runtime_error(std::string(who) + ": unordered Gauss-Jacobi nodes");
        }
    }
    template <typename T>
    void validate_radial(const std::vector<T> &w, const std::vector<T> &x, const char *who)
    {
        if (w.empty() || w.size() != x.size())
            throw std::invalid_argument(std::string(who) + ": malformed radial Gaussian rule");
        bool inc = true, dec = true;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (!finite(w[i]) || !finite(x[i]) || !(w[i] > T(0)) || !(x[i] > T(0) && x[i] < T(1)))
                throw std::runtime_error(std::string(who) + ": invalid radial Gaussian rule");
            if (i)
            {
                inc &= x[i] > x[i - 1];
                dec &= x[i] < x[i - 1];
            }
        }
        if (!inc && !dec)
            throw std::runtime_error(std::string(who) + ": unordered radial Gaussian nodes");
    }
    template <typename T>
    void validate_profile(const std::vector<T> &p, const std::vector<T> &v)
    {
        if (v.size() != p.size() + 1)
            throw std::invalid_argument("finite-Peclet RHS: layer size mismatch");
        for (std::size_t i = 0; i < p.size(); ++i)
            if (!finite(p[i]) || !(p[i] > T(-1) && p[i] < T(1)) || (i && !(p[i] > p[i - 1])))
                throw std::invalid_argument("finite-Peclet RHS: invalid interfaces");
        for (T x : v)
            if (!finite(x))
                throw std::invalid_argument("finite-Peclet RHS: non-finite layer value");
    }

    template <typename T>
    T layer(const SeriesTermData<T> &m, const FPCapKernel<T> &c, T half)
    {
        if (m.n != c.angular_index)
            throw std::invalid_argument("compute_layer_nm_quadrature: angular index mismatch");
        if (c.empty || c.zero)
            return T(0);
        if (c.center)
            return half;
        Sum<T> sum;
        for (std::size_t i = 0; i < c.s.size(); ++i)
        {
            T g = fp_radial_factor(m.n, m.root_fp, m.btilde_fp, c.s[i]);
            if (!finite(g))
                throw std::runtime_error("finite-Peclet RHS: non-finite radial factor");
            sum.add(c.kernel[i] * g);
        }
        T result = c.prefactor * sum.result();
        if (!finite(result))
            throw std::runtime_error("finite-Peclet RHS: non-finite layer integral");
        return result;
    }
    template <typename T>
    T signed_cap(unsigned n, T z, T pos, T full)
    {
        if (!(z < T(0)))
            return pos;
        if (n == 0)
            return full - pos;
        return n % 2 ? pos : -pos;
    }

    // Independent Golub--Welsch implementation for the (alpha=1, beta=0) Jacobi
    // rule used by the full-disk projection below. Deliberately NOT shared with
    // finite_peclet_gram_gauss_jacobi.cpp's own copy: this module must never
    // call into the Gram backend (design contract, finite_peclet_rhs.h), so the
    // two copies stay independent, exactly like the Gram backend's rule() is
    // independent of this file's tridiagonal_ql() cap rule.
    template <class T>
    void gram_rule_ql(std::vector<T> &d, std::vector<T> &e, std::vector<T> &z)
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
                        throw std::runtime_error("full-disk projection rule: QL did not converge");
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
    template <class T>
    void gram_rule(unsigned n, unsigned N, std::vector<T> &x, std::vector<T> &w)
    {
        if (!N)
            throw std::invalid_argument("full-disk projection rule: zero nodes");
        using L = long double;
        std::vector<L> d(N), e(N), z(N, 0);
        for (unsigned k = 0; k < N; ++k)
        {
            const L q = 2 * L(k) + n + 1;
            d[k] = (1 + (L(n) * n - 1) / (q * (q + 2))) / 2;
        }
        for (unsigned k = 1; k < N; ++k)
        {
            const L q = 2 * L(k) + n + 1;
            e[k - 1] = std::sqrt(L(k) * (k + 1) * (k + n) * (k + n + 1) / ((q - 1) * (q + 1))) / q;
        }
        const L mass = 1 / (L(n + 1) * (n + 2));
        z[0] = std::sqrt(mass);
        gram_rule_ql(d, e, z);
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
                throw std::runtime_error("full-disk projection rule: invalid node or weight");
        }
        const L tol = 256 * L(N) * std::numeric_limits<T>::epsilon() * mass;
        if (std::abs(sum - mass) > tol)
            throw std::runtime_error("full-disk projection rule: mass check failed");
    }

    // Bucket granularity for fp_rhs_required_order: an implementation detail,
    // not part of the public API contract, chosen only to bound how many
    // distinct quadrature rules get built per block (see the header comment
    // on fp_rhs_required_order for why bucketing exists at all).
    //
    // GEOMETRIC, not linear: a single angular block's modes can span a WIDE
    // range of bare roots (e.g. root ~2 to ~400 within one n=0 block at
    // max_root=600, since beta_{0,m} ~ 4m+2). A fixed linear step (the first
    // version of this code used 20) then produces O(range/step) distinct
    // buckets *in that one block alone* -- each needing its own O(order^2)
    // Golub-Welsch rule construction (finite_peclet_rhs.cpp's make_fp_gauss_jacobi_rule),
    // not the O(order) cost of evaluating it. Doubling from a base of 20
    // bounds the number of distinct buckets *anywhere in the whole solve* to
    // O(log2(max_root)) (~6-7 for max_root=600), independent of how many
    // modes populate any one block.
    //
    // Geometric bucketing alone is NOT sufficient, though: the stratified
    // driver calls fp_rhs_stratified_inlet once PER ANGULAR BLOCK (~100
    // blocks at max_root=600), and different blocks routinely need the SAME
    // order (every block's lowest-root mode buckets to 20). Rebuilding every
    // needed order independently in every block multiplies this O(log2)
    // bound by the block count, and measured SLOWER than even the single
    // global rule this whole per-mode scheme replaced (~55-90s vs ~10s at
    // max_root=600). The actual fix is the rule caches below
    // (FPCapRuleCache, fp_rhs_build_rule_cache, fp_rhs_build_full_disk_rule_cache):
    // built ONCE from the whole retained mode set, shared read-only across
    // every block's call. Geometric bucketing still matters -- it bounds how
    // large that one shared cache needs to be.
    constexpr unsigned kCapOrderBaseBucket = 20u;

    // Multiplier applied to the mode's own bare root before sizing its
    // quadrature order: order ~ kCapOrderRootCoefficient * bare_root + margin.
    //
    // The naive expectation (root itself, coefficient=1) badly over-resolves:
    // a direct convergence study (compute_layer_nm_quadrature at a genuine
    // interface, threshold 1e-10) across a grid of (n, m) pairs at
    // max_root~600, kappa=0.01 measured the SMALLEST order that actually
    // converges for a given bare root, and the ratio order_needed/bare_root
    // stayed BOUNDED across the whole (n,m) grid (both low and high indices),
    // never exceeding ~0.48:
    //
    //   n    m    bare_root   order_needed   ratio
    //   0    20   81.3        30             0.37
    //   20   0    42.0        20             0.48   <- worst observed
    //   20   99   437.3       100            0.23
    //   99   0    200.0       30             0.15
    //   99   99   595.9       140            0.23
    //   (plus a low-index sweep, n,m in [0,15]: ratios there run higher
    //   because order_needed floors out at ~10-25 while bare_root itself is
    //   small, but the ABSOLUTE order stays inside the base bucket (20)
    //   regardless -- see kCapOrderBaseBucket above.)
    //
    // 0.6 sits comfortably above the worst measured ratio (0.48), leaving
    // margin against the coarse candidate-order grid the study itself used
    // (it can only bracket the true minimum from above). This does NOT mean
    // "order depends on m, not n" (an earlier, wrong hypothesis) -- bare_root
    // ~ 4m + 2n + 2 already mixes both indices, and that combination remains
    // a fine single predictor once correctly scaled.
    constexpr double kCapOrderRootCoefficient = 0.6;

    template <typename T>
    unsigned required_order_impl(const T &bare_root, unsigned cap_quad_margin)
    {
        const T ceiled = std::ceil(T(kCapOrderRootCoefficient) * bare_root);
        const unsigned long long from_root = ceiled > T(0) ? static_cast<unsigned long long>(ceiled) : 0ull;
        const unsigned long long target = std::max<unsigned long long>(from_root + cap_quad_margin, 1ull);
        unsigned long long bucket = kCapOrderBaseBucket;
        while (bucket < target)
            bucket *= 2;
        return static_cast<unsigned>(bucket);
    }

    // Groups `modes` by fp_rhs_required_order(modes[i].root, cap_quad_margin),
    // returning each mode's bucket order (aligned with `modes`) and the sorted
    // list of distinct bucket orders present.
    template <typename T>
    void bucket_modes_by_required_order(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin,
                                        std::vector<unsigned> &bucket_of, std::vector<unsigned> &distinct_buckets)
    {
        bucket_of.resize(modes.size());
        distinct_buckets.clear();
        for (std::size_t i = 0; i < modes.size(); ++i)
        {
            bucket_of[i] = required_order_impl(modes[i].root, cap_quad_margin);
            if (std::find(distinct_buckets.begin(), distinct_buckets.end(), bucket_of[i]) == distinct_buckets.end())
                distinct_buckets.push_back(bucket_of[i]);
        }
    }

    // Looks up the rule built for exactly `order` in a cache produced by
    // fp_rhs_build_rule_cache / fp_rhs_build_full_disk_rule_cache. Throwing
    // here (rather than silently building a fresh rule) makes a stale/wrong
    // cache -- e.g. one built from a different mode set or margin than the
    // caller is now using -- a loud, immediate error instead of a silent
    // reversion to the per-call rebuild this cache exists to avoid.
    template <typename T>
    const FPGaussJacobiRule<T> &lookup_cached_rule(const std::vector<std::pair<unsigned, FPGaussJacobiRule<T>>> &cache, unsigned order)
    {
        for (const auto &entry : cache)
            if (entry.first == order)
                return entry.second;
        throw std::runtime_error("finite-Peclet RHS: rule cache missing required order "
                                  + std::to_string(order) + " (cache must be built from a superset "
                                  "of the current mode set, with the same cap_quad_margin)");
    }
}

template <typename T>
FPGaussJacobiRule<T> make_fp_gauss_jacobi_rule(unsigned N)
{
    if (!N)
        throw std::invalid_argument("make_fp_gauss_jacobi_rule: number_of_points must be positive");
    const long double alpha = 0, beta = .5L;
    std::vector<long double> d(N), e(N, 0), z(N, 0);
    for (unsigned k = 0; k < N; ++k)
    {
        long double q = 2 * k + alpha + beta;
        d[k] = (1 + (beta * beta - alpha * alpha) / (q * (q + 2))) / 2;
    }
    for (unsigned k = 1; k < N; ++k)
    {
        long double q = 2 * k + alpha + beta;
        e[k - 1] = .5L * 2 / q * std::sqrt(k * (k + alpha) * (k + beta) * (k + alpha + beta) / ((q - 1) * (q + 1)));
    }
    z[0] = std::sqrt(2.L / 3.L);
    tridiagonal_ql(d, e, z);
    std::vector<std::pair<long double, long double>> p;
    for (unsigned i = 0; i < N; ++i)
        p.push_back({d[i], z[i] * z[i]});
    std::sort(p.begin(), p.end());
    FPGaussJacobiRule<T> r;
    r.nodes.resize(N);
    r.weights.resize(N);
    for (unsigned i = 0; i < N; ++i)
    {
        r.nodes[i] = T(p[i].first);
        r.weights[i] = T(p[i].second);
    }
    validate_rule(r, "make_fp_gauss_jacobi_rule");
    Sum<T> s;
    for (T x : r.weights)
        s.add(x);
    if (std::abs(s.result() - T(2) / T(3)) > T(256) * T(N) * std::numeric_limits<T>::epsilon())
        throw std::runtime_error("make_fp_gauss_jacobi_rule: invalid weight mass");
    return r;
}

template <typename T>
FPCapKernel<T> fp_rhs_build_cap_kernel(unsigned n, const T &a, const FPGaussJacobiRule<T> &r)
{
    if (!finite(a) || a < T(0) || a > T(1))
        throw std::invalid_argument("compute_layer_nm_quadrature: invalid absolute interface");
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
        c.prefactor = T(kInvSqrtTwoPi) * Tm * Tm;
        for (std::size_t i = 0; i < r.nodes.size(); ++i)
        {
            T eta = r.nodes[i];
            c.s[i] = cc + Tm * eta;
            c.kernel[i] = r.weights[i] * (T(1) - eta) * std::atan2(std::sqrt(Tm * eta), a) / std::sqrt(eta);
        }
    }
    else
    {
        c.prefactor = T(kInvSqrtPi) / T(n) * Tm * Tm * std::sqrt(Tm);
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
            c.kernel[i] = r.weights[i] * (T(1) - r.nodes[i]) * q;
        }
    }
    if (!finite(c.prefactor))
        throw std::runtime_error("finite-Peclet RHS: non-finite cap prefactor");
    return c;
}

template <typename T>
T compute_layer_nm_quadrature(const SeriesTermData<T> &m, const T &a, const FPGaussJacobiRule<T> &r, const std::vector<T> &w, const std::vector<T> &x)
{
    validate_rule(r, "compute_layer_nm_quadrature");
    validate_radial(w, x, "compute_layer_nm_quadrature");
    if (!finite(m.root_fp) || !finite(m.btilde_fp) || m.root_fp < T(0) || m.btilde_fp < T(0))
        throw std::invalid_argument("compute_layer_nm_quadrature: invalid mode");
    FPCapKernel<T> c = fp_rhs_build_cap_kernel(m.n, a, r);
    T half = 0;
    if (c.center)
    {
        Sum<T> s;
        for (std::size_t q = 0; q < x.size(); ++q)
            s.add(w[q] * psinm_r_fp(m.n, m.root_fp, m.btilde_fp, x[q]));
        half = T(.5L * kSqrtTwoPi) * s.result();
    }
    return layer(m, c, half);
}

template <typename T>
unsigned fp_rhs_required_order(const T &bare_root, unsigned cap_quad_margin)
{
    return required_order_impl(bare_root, cap_quad_margin);
}

// See finite_peclet_rhs.h's "Rule caches" section: builds the cap rule
// ((alpha,beta)=(0,1/2)) cache consumed by fp_rhs_stratified_inlet, one rule
// per distinct fp_rhs_required_order present in `modes`.
template <typename T>
FPCapRuleCache<T> fp_rhs_build_rule_cache(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin)
{
    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);
    FPCapRuleCache<T> cache;
    cache.reserve(distinct_buckets.size());
    for (unsigned order : distinct_buckets)
        cache.emplace_back(order, make_fp_gauss_jacobi_rule<T>(order));
    return cache;
}

// Builds the full-disk rule ((alpha,beta)=(1,0)) cache consumed by
// fp_rhs_full_disk_projection / fp_rhs_uniform_inlet. NOT interchangeable
// with fp_rhs_build_rule_cache above -- see finite_peclet_rhs.h.
template <typename T>
FPCapRuleCache<T> fp_rhs_build_full_disk_rule_cache(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin)
{
    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);
    FPCapRuleCache<T> cache;
    cache.reserve(distinct_buckets.size());
    for (unsigned order : distinct_buckets)
    {
        FPGaussJacobiRule<T> rule;
        gram_rule<T>(0u, order, rule.nodes, rule.weights);
        cache.emplace_back(order, std::move(rule));
    }
    return cache;
}

// Full-disk projection sqrt(2pi) * int_0^1 (1-r^2) R_{0,m}(r) r dr for every
// mode in an n=0 block, used for the baseline / center-cap / negative-
// interface-at-n=0 RHS terms (the "zero column" of W^0, since R_{0,0} == 1),
// obtained by independent quadrature -- NOT by forming or factoring any Gram
// matrix here. Modes are bucketed by their own required order
// (fp_rhs_required_order, sized off each mode's own bare root); the rule for
// each distinct bucket present comes from `rule_cache` when supplied
// (see finite_peclet_rhs.h's "Rule caches" section -- this is what lets the
// SAME order's rule be shared across every angular block's call, not just
// within one call), or is built locally, once per distinct order, otherwise.
// Never one rule per mode: that would trade an O(order) evaluation for an
// O(order^2) rule construction per mode, a net regression.
template <typename T>
std::vector<T> fp_rhs_full_disk_projection(const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin, const FPCapRuleCache<T> *rule_cache)
{
    FPCapRuleCache<T> local_cache;
    if (!rule_cache)
    {
        local_cache = fp_rhs_build_full_disk_rule_cache(modes, cap_quad_margin);
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
            // Compensated sum of (w_q/2) * G(s_q) ~= int_0^1 (1-s) G(s) ds / 2
            //                                       = int_0^1 (1-r^2) G(r^2) r dr   (s = r^2).
            Sum<T> sum;
            for (unsigned q = 0; q < order; ++q)
            {
                const T value = fp_radial_factor(0u, modes[m].root_fp, modes[m].btilde_fp, rule.nodes[q]);
                if (!finite(value))
                    throw std::runtime_error("fp_rhs_full_disk_projection: non-finite radial factor");
                sum.add((rule.weights[q] / T(2)) * value);
            }
            const T result = T(kSqrtTwoPi) * sum.result();
            if (!finite(result))
                throw std::runtime_error("fp_rhs_full_disk_projection: non-finite result");
            results[m] = result;
        }
    }
    return results;
}

template <typename T>
std::vector<T> fp_rhs_full_disk_from_gram_column(const std::vector<T> &gram_zero_column)
{
    std::vector<T> result(gram_zero_column.size());
    for (std::size_t i = 0; i < gram_zero_column.size(); ++i)
    {
        result[i] = T(kSqrtTwoPi) * gram_zero_column[i];
        if (!finite(result[i]))
            throw std::runtime_error("fp_rhs_full_disk_from_gram_column: non-finite result");
    }
    return result;
}

template <typename T>
std::vector<T> fp_rhs_uniform_inlet(unsigned angular_index, const std::vector<SeriesTermData<T>> &modes, unsigned cap_quad_margin, const FPCapRuleCache<T> *full_disk_rule_cache)
{
    if (angular_index == 0u)
        return fp_rhs_full_disk_projection(modes, cap_quad_margin, full_disk_rule_cache);
    return std::vector<T>(modes.size(), T(0));
}

template <typename T>
std::vector<T> fp_rhs_stratified_inlet(unsigned angular_index, const std::vector<SeriesTermData<T>> &modes,
                                       const std::vector<T> &interface_positions, const std::vector<T> &layer_values,
                                       unsigned cap_quad_margin, const std::vector<T> &full_disk_column,
                                       const FPCapRuleCache<T> *cap_rule_cache, const FPCapRuleCache<T> *full_disk_rule_cache)
{
    validate_profile(interface_positions, layer_values);
    for (const auto &m : modes)
        if (m.n != angular_index)
            throw std::invalid_argument("fp_rhs_stratified_inlet: mixed angular block");

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
                throw std::invalid_argument("fp_rhs_stratified_inlet: full_disk_column size mismatch");
            full = full_disk_column;
        }
        else
            full = fp_rhs_full_disk_projection(modes, cap_quad_margin, full_disk_rule_cache);
    }
    else
        full = std::vector<T>(modes.size(), T(0));

    // Bucket modes by their own required order (root-driven, not block size)
    // and build one cap rule (plus one FPCapKernel<T> per distinct interface) per
    // distinct bucket present -- see fp_rhs_full_disk_projection above for
    // why bucketing, and finite_peclet_rhs.h's "Rule caches" section for why
    // (and how) that rule is shared across every angular block's call rather
    // than rebuilt fresh by each one.
    std::vector<unsigned> bucket_of, distinct_buckets;
    bucket_modes_by_required_order(modes, cap_quad_margin, bucket_of, distinct_buckets);

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
            caps.push_back(fp_rhs_build_cap_kernel(angular_index, a, rule));

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

template FPGaussJacobiRule<double> make_fp_gauss_jacobi_rule<double>(unsigned);
template FPGaussJacobiRule<long double> make_fp_gauss_jacobi_rule<long double>(unsigned);
template FPCapKernel<double> fp_rhs_build_cap_kernel(unsigned, const double &, const FPGaussJacobiRule<double> &);
template FPCapKernel<long double> fp_rhs_build_cap_kernel(unsigned, const long double &, const FPGaussJacobiRule<long double> &);
template double compute_layer_nm_quadrature(const SeriesTermData<double> &, const double &, const FPGaussJacobiRule<double> &, const std::vector<double> &, const std::vector<double> &);
template long double compute_layer_nm_quadrature(const SeriesTermData<long double> &, const long double &, const FPGaussJacobiRule<long double> &, const std::vector<long double> &, const std::vector<long double> &);
template unsigned fp_rhs_required_order(const double &, unsigned);
template unsigned fp_rhs_required_order(const long double &, unsigned);
template FPCapRuleCache<double> fp_rhs_build_rule_cache(const std::vector<SeriesTermData<double>> &, unsigned);
template FPCapRuleCache<long double> fp_rhs_build_rule_cache(const std::vector<SeriesTermData<long double>> &, unsigned);
template FPCapRuleCache<double> fp_rhs_build_full_disk_rule_cache(const std::vector<SeriesTermData<double>> &, unsigned);
template FPCapRuleCache<long double> fp_rhs_build_full_disk_rule_cache(const std::vector<SeriesTermData<long double>> &, unsigned);
template std::vector<double> fp_rhs_full_disk_projection(const std::vector<SeriesTermData<double>> &, unsigned, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_full_disk_projection(const std::vector<SeriesTermData<long double>> &, unsigned, const FPCapRuleCache<long double> *);
template std::vector<double> fp_rhs_full_disk_from_gram_column(const std::vector<double> &);
template std::vector<long double> fp_rhs_full_disk_from_gram_column(const std::vector<long double> &);
template std::vector<double> fp_rhs_uniform_inlet(unsigned, const std::vector<SeriesTermData<double>> &, unsigned, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_uniform_inlet(unsigned, const std::vector<SeriesTermData<long double>> &, unsigned, const FPCapRuleCache<long double> *);
template std::vector<double> fp_rhs_stratified_inlet(unsigned, const std::vector<SeriesTermData<double>> &, const std::vector<double> &, const std::vector<double> &, unsigned, const std::vector<double> &, const FPCapRuleCache<double> *, const FPCapRuleCache<double> *);
template std::vector<long double> fp_rhs_stratified_inlet(unsigned, const std::vector<SeriesTermData<long double>> &, const std::vector<long double> &, const std::vector<long double> &, unsigned, const std::vector<long double> &, const FPCapRuleCache<long double> *, const FPCapRuleCache<long double> *);
