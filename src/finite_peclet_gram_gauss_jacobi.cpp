#include "finite_peclet_gram_gauss_jacobi.h"

#include "finite_peclet_radial.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
    // Independent Golub--Welsch implementation.  This is intentionally local to
    // this backend (finite_peclet_rhs.cpp's own QL routine, for a differently
    // weighted rule, is a separate, independent copy -- see that file's header
    // comment).
    template <class T>
    void ql(std::vector<T> &d, std::vector<T> &e, std::vector<T> &z)
    {
        const unsigned n = d.size();
        if (!n)
            return;
        // e already stores the sub/super-diagonal at e[0..n-2].
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
                        throw std::runtime_error("Gauss-Jacobi Gram rule: QL did not converge");
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
    void rule(unsigned n, unsigned N, std::vector<T> &x, std::vector<T> &w)
    {
        if (!N)
            throw std::invalid_argument("Gauss-Jacobi Gram rule: zero nodes");
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
        ql(d, e, z);
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
                throw std::runtime_error("Gauss-Jacobi Gram rule: invalid node or weight");
        }
        const L tol = 256 * L(N) * std::numeric_limits<T>::epsilon() * mass;
        if (std::abs(sum - mass) > tol)
            throw std::runtime_error("Gauss-Jacobi Gram rule: mass check failed");
    }

    template <class T>
    void qr_pivot(std::vector<T> &a, unsigned rows, unsigned cols, std::vector<T> &R, std::vector<unsigned> &p)
    {
        p.resize(cols);
        std::iota(p.begin(), p.end(), 0u);
        std::vector<T> norms(cols);
        for (unsigned j = 0; j < cols; ++j)
        {
            T s = 0;
            for (unsigned i = 0; i < rows; ++i)
                s += a[i * cols + j] * a[i * cols + j];
            norms[j] = std::sqrt(s);
        }
        for (unsigned k = 0; k < cols; ++k)
        {
            unsigned best = k;
            for (unsigned j = k + 1; j < cols; ++j)
                if (norms[j] > norms[best])
                    best = j;
            if (best != k)
            {
                for (unsigned i = 0; i < rows; ++i)
                    std::swap(a[i * cols + k], a[i * cols + best]);
                std::swap(norms[k], norms[best]);
                std::swap(p[k], p[best]);
            }
            T sigma = 0;
            for (unsigned i = k; i < rows; ++i)
                sigma += a[i * cols + k] * a[i * cols + k];
            sigma = std::sqrt(sigma);
            if (!(sigma > T(0)) || !std::isfinite((long double)sigma))
                throw std::runtime_error("Gauss-Jacobi QR: rank deficient block");
            const T alpha = a[k * cols + k] >= 0 ? -sigma : sigma;
            const T v0 = a[k * cols + k] - alpha;
            a[k * cols + k] = alpha;
            if (v0 != T(0))
            {
                for (unsigned j = k + 1; j < cols; ++j)
                {
                    T dot = v0 * a[k * cols + j];
                    for (unsigned i = k + 1; i < rows; ++i)
                        dot += a[i * cols + k] * a[i * cols + j];
                    dot /= -(alpha * v0);
                    a[k * cols + j] -= dot * v0;
                    for (unsigned i = k + 1; i < rows; ++i)
                        a[i * cols + j] -= dot * a[i * cols + k];
                }
            }
            // Trailing column norms are recomputed from scratch rather than
            // downdated (Businger-Golub). This costs O(K^2 N) overall -- the same
            // order as the QR itself -- but avoids the cancellation that makes
            // downdating unreliable. Deliberate: do not "optimize" into a downdate
            // without a dedicated accuracy test.
            for (unsigned j = k + 1; j < cols; ++j)
            {
                T s = 0;
                for (unsigned i = k + 1; i < rows; ++i)
                    s += a[i * cols + j] * a[i * cols + j];
                norms[j] = std::sqrt(s);
            }
        }
        R.assign(cols * cols, T(0));
        for (unsigned i = 0; i < cols; ++i)
            for (unsigned j = i; j < cols; ++j)
                R[i * cols + j] = a[i * cols + j];
    }

    // One radial evaluation per (mode, node): N*K total, NOT N*K^2/2. The
    // previous version evaluated fp_radial_factor(..., modes[j], ...) inside the
    // inner (j <= i) loop, i.e. N*(K + K(K+1)/2) times -- a ~K/2x waste that made
    // enabling enable_order_check (two raw_gram calls) ~2x that again.
    template <class T>
    std::vector<std::vector<T>> raw_gram(unsigned n, const std::vector<SeriesTermData<T>> &modes, unsigned N)
    {
        std::vector<T> x, w;
        rule<T>(n, N, x, w);
        const unsigned K = static_cast<unsigned>(modes.size());
        std::vector<T> values(static_cast<std::size_t>(K) * N);
        for (unsigned i = 0; i < K; ++i)
            for (unsigned q = 0; q < N; ++q)
            {
                const T v = fp_radial_factor(n, modes[i].root_fp, modes[i].btilde_fp, x[q]);
                if (!std::isfinite((long double)v))
                    throw std::runtime_error("Gauss-Jacobi Gram: non-finite radial factor");
                values[static_cast<std::size_t>(i) * N + q] = v;
            }
        std::vector<std::vector<T>> g(K, std::vector<T>(K, T(0)));
        for (unsigned i = 0; i < K; ++i)
            for (unsigned j = 0; j <= i; ++j)
            {
                T acc = 0;
                for (unsigned q = 0; q < N; ++q)
                    acc += w[q] * values[static_cast<std::size_t>(i) * N + q]
                                * values[static_cast<std::size_t>(j) * N + q] / T(2);
                g[i][j] = g[j][i] = acc;
            }
        return g;
    }

    // Above this, the block is under-resolved (see fp_gram_factor_gauss_jacobi's
    // gate). Measured healthy value ~= 6 at the validated node count; the first
    // genuinely failing configuration in the plan's table sits at 1.7e3, and the
    // catastrophic ones at 1e12-1e18. 1e6 sits far above any healthy block and
    // far below any broken one.
    template <class T> constexpr T gram_condition_warn_threshold() { return T(1e6); }
}

// One assembly at a PRESCRIBED node count. fp_gram_factor_gauss_jacobi below
// chooses that count (fp_gram_node_count) and, if the resulting block is still
// under-resolved, re-enters here with more nodes.
template <typename T>
static FPGramGaussJacobiFactor<T> fp_gram_factor_gauss_jacobi_at(unsigned n, const std::vector<SeriesTermData<T>> &modes, const FPGramGaussJacobiOptions<T> &o, unsigned N)
{
    if (modes.empty())
        throw std::invalid_argument("Gauss-Jacobi Gram: empty block");
    if (!o.oversampling_factor)
        throw std::invalid_argument("Gauss-Jacobi Gram: oversampling factor must be positive");
    const unsigned K = modes.size();
    for (const auto &m : modes)
        if (m.n != n)
            throw std::invalid_argument("Gauss-Jacobi Gram: mixed angular block");
    std::vector<T> x, w;
    rule<T>(n, N, x, w);
    std::vector<T> a(N * K), D(K);
    std::vector<T> unscaled_samples;
    if (o.retain_samples)
        unscaled_samples.assign(static_cast<std::size_t>(N) * K, T(0));
    for (unsigned q = 0; q < N; ++q)
        for (unsigned j = 0; j < K; ++j)
        {
            const T v = fp_radial_factor(n, modes[j].root_fp, modes[j].btilde_fp, x[q]);
            if (!std::isfinite((long double)v))
                throw std::runtime_error("Gauss-Jacobi Gram: non-finite radial factor");
            if (o.retain_samples)
                unscaled_samples[static_cast<std::size_t>(q) * K + j] = v;
            a[q * K + j] = std::sqrt(w[q] / T(2)) * v;
            D[j] += a[q * K + j] * a[q * K + j];
        }
    for (unsigned j = 0; j < K; ++j)
    {
        D[j] = std::sqrt(D[j]);
        if (!(D[j] > T(0)) || !std::isfinite((long double)D[j]))
            throw std::runtime_error("Gauss-Jacobi Gram: zero column");
    }
    FPGramGaussJacobiFactor<T> f;
    f.angular_index = n;
    f.mode_count = K;
    f.node_count = N;
    f.column_scaling_D = D;
    if (o.retain_samples)
    {
        f.radial_samples = std::move(unscaled_samples);
        f.quadrature_nodes = x;
        f.quadrature_weights = w;
        f.samples_retained = true;
    }

    // Detect the exact constant mode (Neumann n = 0, Lambda = 0 => G == 1) and
    // record its Gram column BEFORE the column scaling below destroys the raw
    // samples. a[q*K+j] = sqrt(w_q/2)*G_j(s_q), so this dot product is exactly
    // W^0_{j,j0} -- cost: K*N multiply-adds, zero additional Kummer evaluations.
    unsigned j0 = K;
    for (unsigned j = 0; j < K; ++j)
        if (n == 0u && modes[j].rate_fp == T(0)) { j0 = j; break; }
    if (j0 != K)
    {
        f.has_constant_mode = true;
        f.constant_mode_local_index = j0;
        f.constant_mode_gram_column.assign(K, T(0));
        for (unsigned j = 0; j < K; ++j)
        {
            T acc = 0;
            for (unsigned q = 0; q < N; ++q)
                acc += a[q * K + j] * a[q * K + j0];
            f.constant_mode_gram_column[j] = acc;
        }
    }

    for (unsigned j = 0; j < K; ++j)
        for (unsigned q = 0; q < N; ++q)
            a[q * K + j] /= D[j];

    qr_pivot(a, N, K, f.upper_triangular_R, f.pivot_permutation);
    T mn = std::numeric_limits<T>::max(), mx = 0;
    for (unsigned i = 0; i < K; ++i)
    {
        const T v = std::abs(f.upper_triangular_R[i * K + i]);
        if (!(v > T(0)) || !std::isfinite((long double)v))
            throw std::runtime_error("Gauss-Jacobi QR: invalid diagonal");
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    f.diagnostics = {n, K, N, mn, mx, mx / mn, T(0), false};

    if (o.enable_order_check)
    {
        auto g1 = raw_gram(n, modes, N), g2 = raw_gram(n, modes, N + 40);
        T d = 0, s = 0;
        for (unsigned i = 0; i < K; ++i)
            for (unsigned j = 0; j < K; ++j)
            {
                d = std::max(d, std::abs(g1[i][j] - g2[i][j]));
                s = std::max(s, std::abs(g1[i][j]));
            }
        f.diagnostics.order_consistency = d / (s > T(0) ? s : T(1));
        f.diagnostics.order_check_ran = true;
    }
    return f;
}

template <typename T>
FPGramGaussJacobiFactor<T> fp_gram_factor_gauss_jacobi(unsigned n, const std::vector<SeriesTermData<T>> &modes, const FPGramGaussJacobiOptions<T> &o)
{
    if (modes.empty())
        throw std::invalid_argument("Gauss-Jacobi Gram: empty block");
    const unsigned K = static_cast<unsigned>(modes.size());

    // The node rule needs the block's radial scale b_max = max sqrt(Lam): the
    // mode s^n e^{-b s/2} puts its mass at s* = n/b, so a block whose b_max far
    // exceeds n is sampled nowhere near where its modes differ unless the node
    // count grows with b_max. See fp_gram_node_count's header comment.
    T radial_scale = 0;
    if (o.scale_aware_nodes)
        for (const auto &m : modes)
            if (m.root_fp > radial_scale) radial_scale = m.root_fp;

    const unsigned N = fp_gram_node_count(K, o, radial_scale);
    FPGramGaussJacobiFactor<T> f = fp_gram_factor_gauss_jacobi_at(n, modes, o, N);

    // Under-resolution gate. cond(B_hat) is ~6 at the validated node count and
    // explodes by many orders as N approaches K (measured: 6.1 at N = 140 vs
    // 7.9e12 at N = 30, for n = 20, K = 60). A large ratio here means the block
    // is under-sampled and the coefficients would be silently wrong. This is
    // what would have caught the (oversampling_margin = 2) configuration that
    // shipped briefly in benchmarks/benchmark_config.json. Unconditional -- not
    // gated on a verbosity flag.
    if (f.diagnostics.condition_estimate > gram_condition_warn_threshold<T>())
        throw std::runtime_error(
            "Gauss-Jacobi Gram: block is under-resolved (n = " + std::to_string(n)
            + ", K = " + std::to_string(K) + ", nodes = " + std::to_string(f.node_count)
            + ", b_max = " + std::to_string((long double)radial_scale)
            + ", R diagonal ratio " + std::to_string((long double)f.diagnostics.condition_estimate)
            + "); the block's modes peak at s = n/b_max and the rule is not sampling "
              "there -- raise oversampling_factor/oversampling_margin, or lower max_root");
    return f;
}

template <typename T>
void fp_gram_solve_gauss_jacobi(const FPGramGaussJacobiFactor<T> &f, const std::vector<T> &b, std::vector<T> &c)
{
    const unsigned K = f.mode_count;
    if (b.size() != K || f.upper_triangular_R.size() != K * K || f.column_scaling_D.size() != K || f.pivot_permutation.size() != K)
        throw std::invalid_argument("Gauss-Jacobi Gram solve: bad factor or RHS");
    std::vector<T> t(K), y(K), z(K);
    for (unsigned i = 0; i < K; ++i)
        t[i] = b[f.pivot_permutation[i]] / f.column_scaling_D[f.pivot_permutation[i]];
    for (unsigned i = 0; i < K; ++i)
    {
        T v = t[i];
        for (unsigned j = 0; j < i; ++j)
            v -= f.upper_triangular_R[j * K + i] * y[j];
        y[i] = v / f.upper_triangular_R[i * K + i];
    }
    for (unsigned ii = K; ii--;)
    {
        T v = y[ii];
        for (unsigned j = ii + 1; j < K; ++j)
            v -= f.upper_triangular_R[ii * K + j] * z[j];
        z[ii] = v / f.upper_triangular_R[ii * K + ii];
    }
    c.assign(K, T(0));
    for (unsigned i = 0; i < K; ++i)
        c[f.pivot_permutation[i]] = z[i] / f.column_scaling_D[f.pivot_permutation[i]];
}
template <typename T>
void fp_gram_reconstruct_gauss_jacobi(const FPGramGaussJacobiFactor<T> &f, std::vector<std::vector<T>> &g)
{
    const unsigned K = f.mode_count;
    g.assign(K, std::vector<T>(K, T(0)));
    for (unsigned i = 0; i < K; ++i)
        for (unsigned j = 0; j < K; ++j)
        {
            T v = 0;
            for (unsigned k = 0; k < K; ++k)
                v += f.upper_triangular_R[k * K + i] * f.upper_triangular_R[k * K + j];
            g[f.pivot_permutation[i]][f.pivot_permutation[j]] = v * f.column_scaling_D[f.pivot_permutation[i]] * f.column_scaling_D[f.pivot_permutation[j]];
        }
}
template FPGramGaussJacobiFactor<double> fp_gram_factor_gauss_jacobi(unsigned, const std::vector<SeriesTermData<double>> &, const FPGramGaussJacobiOptions<double> &);
template FPGramGaussJacobiFactor<long double> fp_gram_factor_gauss_jacobi(unsigned, const std::vector<SeriesTermData<long double>> &, const FPGramGaussJacobiOptions<long double> &);
template void fp_gram_solve_gauss_jacobi(const FPGramGaussJacobiFactor<double> &, const std::vector<double> &, std::vector<double> &);
template void fp_gram_solve_gauss_jacobi(const FPGramGaussJacobiFactor<long double> &, const std::vector<long double> &, std::vector<long double> &);
template void fp_gram_reconstruct_gauss_jacobi(const FPGramGaussJacobiFactor<double> &, std::vector<std::vector<double>> &);
template void fp_gram_reconstruct_gauss_jacobi(const FPGramGaussJacobiFactor<long double> &, std::vector<std::vector<long double>> &);
