// ---------------------------------------------------------------------------
// qep_computations.cpp — implementation of the exact finite-Péclet QEP solution.
// See qep_computations.h for the formulation, the solution method, and the
// references (Neuhauser et al. 2025, Sec. 2.4 / Appendix B).
//
// Contents:
//   * dense linear algebra helpers (Cholesky, triangular solves, LU, Jacobi)
//   * qep_build_and_solve  — assemble K̃, B per angular index and solve the
//                            symmetric-definite linearised pencil
//   * qep_evaluate_at_points — reconstruct the field from the spectral data
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h>

#include "qep_computations.h"
#include "finite_peclet_quadrature.h"
#include "math_functions.h"

namespace
{

using Mat = std::vector<std::vector<long double>>; // (only for readability in comments)

// --- Cholesky factorisation of a symmetric positive-definite matrix. ---------
// Returns the lower-triangular L with A = L Lᵀ. Throws if A is not numerically
// positive definite (which would signal a mis-assembled pencil).
template <typename Ttype>
void cholesky_lower(const std::vector<std::vector<Ttype>> &A,
                    std::vector<std::vector<Ttype>> &L)
{
    const size_t n = A.size();
    L.assign(n, std::vector<Ttype>(n, static_cast<Ttype>(0)));
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            Ttype sum = A[i][j];
            for (size_t k = 0; k < j; ++k)
                sum -= L[i][k] * L[j][k];

            if (i == j)
            {
                if (!(sum > static_cast<Ttype>(0)))
                    throw std::runtime_error(
                        "qep_computations: Cholesky failed — the linearised pencil "
                        "matrix B is not positive definite (is the beta=0 mode present?).");
                L[i][i] = std::sqrt(sum);
            }
            else
            {
                L[i][j] = sum / L[j][j];
            }
        }
    }
}

// Solve L y = b in place for lower-triangular L (forward substitution).
template <typename Ttype>
void forward_subst(const std::vector<std::vector<Ttype>> &L, std::vector<Ttype> &b)
{
    const size_t n = L.size();
    for (size_t i = 0; i < n; ++i)
    {
        Ttype s = b[i];
        for (size_t k = 0; k < i; ++k) s -= L[i][k] * b[k];
        b[i] = s / L[i][i];
    }
}

// Solve Lᵀ y = b in place for lower-triangular L (back substitution).
template <typename Ttype>
void back_subst_transpose(const std::vector<std::vector<Ttype>> &L, std::vector<Ttype> &b)
{
    const size_t n = L.size();
    for (size_t ii = n; ii-- > 0;)
    {
        Ttype s = b[ii];
        for (size_t k = ii + 1; k < n; ++k) s -= L[k][ii] * b[k];
        b[ii] = s / L[ii][ii];
    }
}

// --- Cyclic Jacobi eigensolver for a real symmetric matrix. ------------------
// On return eval[j] are the eigenvalues and evec[i][j] the i-th component of the
// j-th (orthonormal) eigenvector. Jacobi is used rather than tridiagonal+QL
// because it is short, self-contained, and delivers the eigenvectors with high
// orthogonality for the moderate block sizes involved here.
template <typename Ttype>
void jacobi_symmetric(std::vector<std::vector<Ttype>> A,
                      std::vector<Ttype> &eval,
                      std::vector<std::vector<Ttype>> &evec)
{
    const size_t n = A.size();
    evec.assign(n, std::vector<Ttype>(n, static_cast<Ttype>(0)));
    for (size_t i = 0; i < n; ++i) evec[i][i] = static_cast<Ttype>(1);

    eval.assign(n, static_cast<Ttype>(0));
    if (n == 0) return;
    if (n == 1) { eval[0] = A[0][0]; return; }

    const Ttype eps = std::numeric_limits<Ttype>::epsilon();
    const unsigned max_sweeps = 100;

    for (unsigned sweep = 0; sweep < max_sweeps; ++sweep)
    {
        // Off-diagonal Frobenius norm; stop once it is at round-off level.
        Ttype off = static_cast<Ttype>(0), diag = static_cast<Ttype>(0);
        for (size_t p = 0; p < n; ++p)
        {
            diag += A[p][p] * A[p][p];
            for (size_t q = p + 1; q < n; ++q) off += A[p][q] * A[p][q];
        }
        if (!(off > eps * eps * (diag + static_cast<Ttype>(1)))) break;

        for (size_t p = 0; p < n; ++p)
        {
            for (size_t q = p + 1; q < n; ++q)
            {
                const Ttype apq = A[p][q];
                if (std::fabs(apq) <= eps * std::sqrt(std::fabs(A[p][p] * A[q][q])))
                    continue;

                // Standard Jacobi rotation annihilating A[p][q].
                const Ttype theta = (A[q][q] - A[p][p]) / (static_cast<Ttype>(2) * apq);
                const Ttype sgn = (theta >= static_cast<Ttype>(0)) ? static_cast<Ttype>(1)
                                                                   : static_cast<Ttype>(-1);
                const Ttype t = sgn / (std::fabs(theta) + std::sqrt(theta * theta + static_cast<Ttype>(1)));
                const Ttype c = static_cast<Ttype>(1) / std::sqrt(t * t + static_cast<Ttype>(1));
                const Ttype s = t * c;

                for (size_t k = 0; k < n; ++k)
                {
                    const Ttype akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < n; ++k)
                {
                    const Ttype apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n; ++k)
                {
                    const Ttype vkp = evec[k][p], vkq = evec[k][q];
                    evec[k][p] = c * vkp - s * vkq;
                    evec[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    for (size_t i = 0; i < n; ++i) eval[i] = A[i][i];
}

// --- Dense LU with partial pivoting; solves A x = b. ------------------------
template <typename Ttype>
void lu_solve_dense(std::vector<std::vector<Ttype>> A, std::vector<Ttype> b,
                    std::vector<Ttype> &x)
{
    const size_t n = A.size();
    std::vector<size_t> piv(n);
    std::iota(piv.begin(), piv.end(), 0);

    for (size_t i = 0; i < n; ++i)
    {
        size_t best = i;
        Ttype best_val = std::fabs(A[i][i]);
        for (size_t r = i + 1; r < n; ++r)
        {
            const Ttype v = std::fabs(A[r][i]);
            if (v > best_val) { best_val = v; best = r; }
        }
        if (best != i) { std::swap(A[i], A[best]); std::swap(b[i], b[best]); }

        Ttype pivot = A[i][i];
        if (!(std::fabs(pivot) > static_cast<Ttype>(0)))
            pivot = std::numeric_limits<Ttype>::epsilon(); // singular guard
        for (size_t r = i + 1; r < n; ++r)
        {
            const Ttype f = A[r][i] / pivot;
            if (f == static_cast<Ttype>(0)) continue;
            for (size_t c = i; c < n; ++c) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }

    x.assign(n, static_cast<Ttype>(0));
    for (size_t ii = n; ii-- > 0;)
    {
        Ttype s = b[ii];
        for (size_t k = ii + 1; k < n; ++k) s -= A[ii][k] * x[k];
        x[ii] = s / A[ii][ii];
    }
}

} // namespace

template <typename Ttype>
void qep_build_and_solve(const std::vector<SeriesTermData<Ttype>> &series_data,
                         unsigned max_K,
                         const Ttype &kappa,
                         const Ttype &constant_term,
                         std::vector<QEPBlock<Ttype>> &blocks,
                         unsigned panels)
{
    blocks.clear();
    if (max_K == 0) return;
    if (!(kappa > static_cast<Ttype>(0)))
        throw std::invalid_argument("qep_build_and_solve: kappa must be strictly positive.");

    // Group retained modes by angular index n. Every mode is kept, including the
    // beta = 0 constant mode: it carries the far-field value and its O(kappa)
    // coupling to the decaying modes is the inlet back-diffusion shift.
    unsigned max_n = 0;
    for (unsigned k = 0; k < max_K; ++k) max_n = std::max(max_n, series_data[k].n);

    std::vector<std::vector<unsigned>> groups(max_n + 1);
    for (unsigned k = 0; k < max_K; ++k)
        groups[series_data[k].n].push_back(k);

    std::vector<unsigned> active_n;
    for (unsigned n = 0; n <= max_n; ++n)
        if (!groups[n].empty()) active_n.push_back(n);

    blocks.resize(active_n.size());

    // Shared quadrature nodes for the unweighted overlaps U_ab = ∫₀¹ r R_a R_b dr.
    std::vector<Ttype> qr, qw;
    fp_quad::gl_nodes(panels, qr, qw);

    // Each angular index is an independent block: build, linearise, and solve.
    // Parallel over n (see FINITE_PECLET_PARALLELIZATION.md).
#pragma omp parallel for schedule(dynamic)
    for (size_t gi = 0; gi < active_n.size(); ++gi)
    {
        const unsigned n = active_n[gi];
        const std::vector<unsigned> &g = groups[n];
        const size_t K = g.size();

        QEPBlock<Ttype> &blk = blocks[gi];
        blk.n = n;
        blk.flat_idx.assign(g.begin(), g.end());
        blk.root.resize(K);
        blk.norm.resize(K);

        constexpr long double LD_SQRT_2PI = 2.50662827463100050242L;
        const Ttype tiny = static_cast<Ttype>(1e-12L);

        std::vector<Ttype> beta2(K), c0(K);
        for (size_t a = 0; a < K; ++a)
        {
            const SeriesTermData<Ttype> &t = series_data[g[a]];
            blk.root[a] = t.root;
            blk.norm[a] = std::sqrt(t.norm);          // N_a from the stored N²
            beta2[a]    = t.root * t.root;            // β²_a
            // ĉ_a(0) = C_a(0) · N_a. The bare projection leaves the constant
            // mode's coefficient at zero and stores the flow-weighted mean in
            // constant_term, so restore it here: C_00 = constant_term·√(2π)
            // reproduces C_00·R_00·Φ_0 = constant_term.
            const Ttype C0 = (t.root <= tiny)
                ? constant_term * static_cast<Ttype>(LD_SQRT_2PI)
                : t.coeff;
            c0[a] = C0 * blk.norm[a];
        }

        // Tabulate the normalised radial modes at the quadrature nodes once, then
        // form the Gram matrix K̃_ab = ∫₀¹ r φ_a φ_b dr by dot products. This keeps
        // the number of hypergeometric evaluations at O(K · nodes) instead of
        // O(K² · nodes).
        const size_t NQ = qr.size();
        std::vector<std::vector<Ttype>> phi(K, std::vector<Ttype>(NQ));
        for (size_t a = 0; a < K; ++a)
        {
            const Ttype ba = blk.root[a], Na = blk.norm[a];
            for (size_t q = 0; q < NQ; ++q)
                phi[a][q] = psinm_r(n, ba, qr[q]) / Na;
        }

        std::vector<std::vector<Ttype>> Kt(K, std::vector<Ttype>(K, static_cast<Ttype>(0)));
        for (size_t a = 0; a < K; ++a)
        {
            for (size_t b = a; b < K; ++b)
            {
                Ttype s = static_cast<Ttype>(0);
                for (size_t q = 0; q < NQ; ++q)
                    s += qw[q] * qr[q] * phi[a][q] * phi[b][q];
                Kt[a][b] = s;
                Kt[b][a] = s;
            }
        }

        // --- spectrum shift, λ = μ − σ ------------------------------------
        // With a β = 0 mode present, B is singular and z = [0; e₀] annihilates
        // BOTH pencil matrices: the pencil is singular and unsolvable as it
        // stands. Shifting by σ > 0 replaces B by B + σI in the definite block
        // and makes the pencil regular again. This is exact, not an
        // approximation: λ = μ − σ recovers the eigenvalue.
        //   μ²(κK̃) + μ(I − 2σκK̃) + (σ²κK̃ − σI − B) = 0
        // Definiteness needs B + σI − σ²κK̃ ≻ 0, i.e. σ < 1/(κ·λmax(K̃)); σ is
        // taken well inside that bound (Gershgorin) and capped by the spectrum
        // scale so the cancellation in μ − σ stays harmless.
        Ttype gersh = static_cast<Ttype>(0);
        for (size_t a = 0; a < K; ++a)
        {
            Ttype rs = static_cast<Ttype>(0);
            for (size_t b = 0; b < K; ++b) rs += std::fabs(Kt[a][b]);
            gersh = std::max(gersh, rs);
        }
        if (!(gersh > static_cast<Ttype>(0))) gersh = static_cast<Ttype>(1);

        Ttype b2min = static_cast<Ttype>(0);
        for (size_t a = 0; a < K; ++a)
            if (beta2[a] > tiny && (b2min == static_cast<Ttype>(0) || beta2[a] < b2min))
                b2min = beta2[a];
        if (!(b2min > static_cast<Ttype>(0))) b2min = static_cast<Ttype>(1);

        Ttype sigma = std::min(static_cast<Ttype>(0.25L) / (kappa * gersh),
                               static_cast<Ttype>(0.5L) * b2min);

        const size_t N2 = 2 * K;
        std::vector<std::vector<Ttype>> Bmat, Cmat, L;

        // Build and factorise; if round-off still costs positive definiteness,
        // shrink the shift and retry rather than failing.
        bool factored = false;
        for (unsigned attempt = 0; attempt < 8 && !factored; ++attempt)
        {
            Bmat.assign(N2, std::vector<Ttype>(N2, static_cast<Ttype>(0)));
            Cmat.assign(N2, std::vector<Ttype>(N2, static_cast<Ttype>(0)));
            for (size_t a = 0; a < K; ++a)
            {
                for (size_t b = 0; b < K; ++b)
                {
                    const Ttype kKt = kappa * Kt[a][b];
                    Bmat[a][b]         = kKt;                          // M' = κK̃
                    Bmat[K + a][K + b] = -sigma * sigma * kKt;         // −K' part
                    Cmat[a][b]         = -static_cast<Ttype>(2) * sigma * kKt; // C' part
                    Cmat[a][K + b]     = sigma * sigma * kKt;          // K' part
                    Cmat[K + a][b]     = sigma * sigma * kKt;
                }
                Bmat[K + a][K + a] += beta2[a] + sigma;                // −K' = B+σI−σ²κK̃
                Cmat[a][a]         += static_cast<Ttype>(1);           // C' = I − 2σκK̃
                Cmat[a][K + a]     += -(beta2[a] + sigma);             // K' = σ²κK̃ − σI − B
                Cmat[K + a][a]     += -(beta2[a] + sigma);
            }

            try { cholesky_lower(Bmat, L); factored = true; }
            catch (const std::runtime_error &) { sigma *= static_cast<Ttype>(0.25L); }
        }
        if (!factored)
            throw std::runtime_error("qep_build_and_solve: could not make the shifted pencil "
                                     "positive definite.");

        // Y = L⁻¹ 𝓒 (column by column), then S = L⁻¹ Yᵀ (S is symmetric).
        std::vector<std::vector<Ttype>> Y(N2, std::vector<Ttype>(N2));
        for (size_t c = 0; c < N2; ++c)
        {
            std::vector<Ttype> col(N2);
            for (size_t r = 0; r < N2; ++r) col[r] = Cmat[r][c];
            forward_subst(L, col);
            for (size_t r = 0; r < N2; ++r) Y[r][c] = col[r];
        }
        std::vector<std::vector<Ttype>> S(N2, std::vector<Ttype>(N2));
        for (size_t c = 0; c < N2; ++c)
        {
            std::vector<Ttype> col(N2);
            for (size_t r = 0; r < N2; ++r) col[r] = Y[c][r]; // row c of Y = col c of Yᵀ
            forward_subst(L, col);
            for (size_t r = 0; r < N2; ++r) S[r][c] = col[r];
        }

        std::vector<Ttype> nu;
        std::vector<std::vector<Ttype>> Yv;
        jacobi_symmetric(S, nu, Yv);

        // Recover z = L⁻ᵀ y, then keep the K decaying eigenpairs (λ = −ν > 0).
        std::vector<std::pair<Ttype, size_t>> order;
        order.reserve(N2);
        for (size_t j = 0; j < N2; ++j) order.push_back({nu[j], j});
        std::sort(order.begin(), order.end(),
                  [](const std::pair<Ttype, size_t> &a, const std::pair<Ttype, size_t> &b) {
                      return a.first < b.first;   // most negative ν first = largest λ
                  });

        blk.lambda.assign(K, static_cast<Ttype>(0));
        blk.V.assign(K, std::vector<Ttype>(K, static_cast<Ttype>(0)));

        for (size_t jj = 0; jj < K; ++jj)
        {
            // Take the K most negative ν (equivalently the K largest λ), then
            // store them ascending in λ for readability. ν = −μ and λ = μ − σ,
            // so λ = −ν − σ. Exactly K eigenvalues satisfy λ ≥ 0 (one of them
            // being the constant mode's λ = 0), so this selects the decaying set.
            const size_t src = order[jj].second;
            Ttype lam = -order[jj].first - sigma;
            // Snap the constant mode: λ = 0 is exact, and the shift leaves only
            // round-off behind.
            if (std::fabs(lam) < static_cast<Ttype>(1e-9L) * (static_cast<Ttype>(1) + beta2[K - 1]))
                lam = static_cast<Ttype>(0);
            const size_t dst = K - 1 - jj;

            std::vector<Ttype> z(N2);
            for (size_t r = 0; r < N2; ++r) z[r] = Yv[r][src];
            back_subst_transpose(L, z);

            // Bottom half of z is the eigenvector v; normalise for conditioning.
            Ttype nrm = static_cast<Ttype>(0);
            for (size_t a = 0; a < K; ++a) nrm += z[K + a] * z[K + a];
            nrm = std::sqrt(nrm);
            if (!(nrm > static_cast<Ttype>(0))) nrm = static_cast<Ttype>(1);

            blk.lambda[dst] = lam;
            for (size_t a = 0; a < K; ++a) blk.V[a][dst] = z[K + a] / nrm;
        }

        // Inlet amplitudes: V a = ĉ(0).
        lu_solve_dense(blk.V, c0, blk.amp);
    }
}

template <typename Ttype>
void qep_evaluate_at_points(const std::vector<QEPBlock<Ttype>> &blocks,
                            const std::vector<Ttype> &x_points,
                            const std::vector<Ttype> &r_points,
                            const std::vector<Ttype> &phi_points,
                            std::vector<Ttype> &out)
{
    const size_t NP = x_points.size();
    if (r_points.size() != NP || phi_points.size() != NP)
        throw std::invalid_argument("qep_evaluate_at_points: x, r and phi must have equal length.");

    out.assign(NP, static_cast<Ttype>(0));
    if (NP == 0 || blocks.empty()) return;

    // ---------------------------------------------------------------------
    // Radial tabulation, shared by every plane.
    //
    // The radial modes depend only on r, and evaluation clouds are usually
    // built as rings (many angular points per radius), so the number of
    // DISTINCT radii is far smaller than the number of points -- for a typical
    // disk, tens instead of tens of thousands. Grouping the points by radius
    // and tabulating each mode once per distinct radius therefore removes the
    // dominant cost of evaluation. The table is also independent of x, so it is
    // built once here rather than per plane. This mirrors the psinm cache used
    // by the base evaluation path in CDBaseSolution.
    // ---------------------------------------------------------------------
    std::vector<size_t> r_index(NP, 0);
    std::vector<Ttype> unique_r;
    bool use_cache = false;

    {
        const Ttype max_r = *std::max_element(r_points.begin(), r_points.end());
        const Ttype r_tol = std::max(static_cast<Ttype>(1e-12L),
                                     static_cast<Ttype>(1e-8L) *
                                         std::max(max_r, static_cast<Ttype>(1)));
        std::vector<size_t> order_r(NP);
        std::iota(order_r.begin(), order_r.end(), 0);
        std::sort(order_r.begin(), order_r.end(),
                  [&](size_t a, size_t b) { return r_points[a] < r_points[b]; });

        unique_r.reserve(NP);
        bool grouping_ok = true;
        size_t start = 0;
        while (start < NP)
        {
            const Ttype r0 = r_points[order_r[start]];
            Ttype r_min = r0, r_max = r0;
            size_t end = start + 1;
            while (end < NP)
            {
                const Ttype rv = r_points[order_r[end]];
                if (rv - r0 > r_tol) break;
                r_min = std::min(r_min, rv);
                r_max = std::max(r_max, rv);
                ++end;
            }
            if ((r_max - r_min) > r_tol) { grouping_ok = false; break; }

            const size_t gi = unique_r.size();
            unique_r.push_back((r_min + r_max) * static_cast<Ttype>(0.5));
            for (size_t j = start; j < end; ++j) r_index[order_r[j]] = gi;
            start = end;
        }

        size_t total_modes = 0;
        for (const QEPBlock<Ttype> &b : blocks) total_modes += b.lambda.size();
        const size_t max_cache_bytes = static_cast<size_t>(256) * 1024 * 1024;
        const size_t cache_bytes = grouping_ok
            ? unique_r.size() * total_modes * sizeof(Ttype) : 0;
        use_cache = grouping_ok && unique_r.size() < NP
                    && cache_bytes > 0 && cache_bytes <= max_cache_bytes;
    }

    const size_t U = unique_r.size();
    // rad[bi][a*U + u] = R_{n,a}(unique_r[u])
    std::vector<std::vector<Ttype>> rad(blocks.size());
    if (use_cache)
    {
        for (size_t bi = 0; bi < blocks.size(); ++bi)
            rad[bi].assign(blocks[bi].lambda.size() * U, static_cast<Ttype>(0));

        // Parallel over distinct radii: each u writes its own slots, so the
        // blocks/modes loops inside carry no race.
#pragma omp parallel for schedule(dynamic)
        for (size_t u = 0; u < U; ++u)
        {
            const Ttype rv = unique_r[u];
            for (size_t bi = 0; bi < blocks.size(); ++bi)
            {
                const QEPBlock<Ttype> &blk = blocks[bi];
                for (size_t a = 0; a < blk.lambda.size(); ++a)
                    rad[bi][a * U + u] = psinm_r(blk.n, blk.root[a], rv);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Group points sharing an x value: the modal coefficients C_a(x) cost
    // O(K^2) per block and are identical across a plane, while the per-point
    // sum is only O(K).
    // ---------------------------------------------------------------------
    std::vector<size_t> order(NP);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return x_points[a] < x_points[b]; });

    const Ttype x_tol = static_cast<Ttype>(1e-12L);
    std::vector<std::vector<size_t>> plane;
    std::vector<Ttype> plane_x;
    for (size_t i = 0; i < NP;)
    {
        const Ttype x0 = x_points[order[i]];
        std::vector<size_t> members;
        while (i < NP && std::fabs(x_points[order[i]] - x0) <= x_tol)
        {
            members.push_back(order[i]);
            ++i;
        }
        plane_x.push_back(x0);
        plane.push_back(std::move(members));
    }

    // Modal coefficients C_a(x) of every block at one plane.
    auto assemble = [&](const Ttype xv, std::vector<std::vector<Ttype>> &C) {
        C.assign(blocks.size(), std::vector<Ttype>());
        for (size_t bi = 0; bi < blocks.size(); ++bi)
        {
            const QEPBlock<Ttype> &blk = blocks[bi];
            const size_t K = blk.lambda.size();
            std::vector<Ttype> decay(K);
            for (size_t j = 0; j < K; ++j)
                decay[j] = blk.amp[j] * std::exp(-blk.lambda[j] * xv);
            C[bi].assign(K, static_cast<Ttype>(0));
            for (size_t a = 0; a < K; ++a)
            {
                Ttype s = static_cast<Ttype>(0);
                for (size_t j = 0; j < K; ++j) s += blk.V[a][j] * decay[j];
                C[bi][a] = s / blk.norm[a];
            }
        }
    };

    // Series sum at one point, using the radial table when available.
    auto point_value = [&](const std::vector<std::vector<Ttype>> &C, size_t i) -> Ttype {
        const Ttype rv = r_points[i], pv = phi_points[i];
        Ttype acc = static_cast<Ttype>(0);
        for (size_t bi = 0; bi < blocks.size(); ++bi)
        {
            const QEPBlock<Ttype> &blk = blocks[bi];
            const Ttype ang = sn_phi(blk.n, pv);
            if (ang == static_cast<Ttype>(0)) continue;
            const size_t K = blk.lambda.size();
            Ttype radial = static_cast<Ttype>(0);
            if (use_cache)
            {
                const Ttype *row = rad[bi].data();
                const size_t u = r_index[i];
                for (size_t a = 0; a < K; ++a) radial += C[bi][a] * row[a * U + u];
            }
            else
            {
                for (size_t a = 0; a < K; ++a)
                    radial += C[bi][a] * psinm_r(blk.n, blk.root[a], rv);
            }
            acc += radial * ang;
        }
        return acc;
    };

    // Parallelise over planes when there are several; otherwise over the points
    // of the single plane (see FINITE_PECLET_PARALLELIZATION.md).
    if (plane.size() > 1)
    {
#pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < plane.size(); ++p)
        {
            std::vector<std::vector<Ttype>> C;
            assemble(plane_x[p], C);
            for (size_t idx = 0; idx < plane[p].size(); ++idx)
            {
                const size_t i = plane[p][idx];
                out[i] = point_value(C, i);
            }
        }
    }
    else
    {
        std::vector<std::vector<Ttype>> C;
        assemble(plane_x[0], C);
        const std::vector<size_t> &members = plane[0];
#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < members.size(); ++idx)
        {
            const size_t i = members[idx];
            out[i] = point_value(C, i);
        }
    }
}

// --- Explicit instantiations -------------------------------------------------
template void qep_build_and_solve<double>(const std::vector<SeriesTermData<double>> &, unsigned, const double &, const double &, std::vector<QEPBlock<double>> &, unsigned);
template void qep_build_and_solve<long double>(const std::vector<SeriesTermData<long double>> &, unsigned, const long double &, const long double &, std::vector<QEPBlock<long double>> &, unsigned);
template void qep_evaluate_at_points<double>(const std::vector<QEPBlock<double>> &, const std::vector<double> &, const std::vector<double> &, const std::vector<double> &, std::vector<double> &);
template void qep_evaluate_at_points<long double>(const std::vector<QEPBlock<long double>> &, const std::vector<long double> &, const std::vector<long double> &, const std::vector<long double> &, std::vector<long double> &);
