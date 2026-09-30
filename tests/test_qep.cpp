// ---------------------------------------------------------------------------
// test_qep.cpp — unit tests for the exact finite-Péclet QEP solver
// (src/qep_computations.{h,cpp}).
//
// The tests anchor the solver to:
//   * the QEP residual itself, ( κλ²K̃ + λI − B ) v = 0, verified directly;
//   * the 1x1 case, which must reproduce the closed-form root of
//     κ α λ² + λ − β² = 0 (theory/analytical_solution.tex, eq. seed_quadratic);
//   * the bare limit κ → 0, where λ_a → β²_a.
// ---------------------------------------------------------------------------

#include "tinytest.h"

#include "qep_computations.h"
#include "finite_peclet_norms.h"
#include "series_term_struct.h"

#include <cmath>
#include <vector>

namespace {

// A few genuine Dirichlet (Graetz) eigenvalues, n = 0.
const double GRAETZ_BETA[5] = {2.704364419882532, 6.679031449346628,
                               10.673379538053736, 14.671078462736212,
                               18.669871864451220};

// Build a synthetic series-data vector for angular index n = 0 with the given roots.
// The weighted norms are computed with the same quadrature the solver uses, so the
// test is self-consistent rather than depending on a stored table.
std::vector<SeriesTermData<double>> make_series(unsigned K, double coeff_value)
{
    std::vector<SeriesTermData<double>> sd(K);
    for (unsigned a = 0; a < K; ++a)
    {
        const double b = GRAETZ_BETA[a];
        sd[a].n = 0;
        sd[a].m = a;
        sd[a].root = b;
        // N² = ∫₀¹ (1-r²) R² r dr  == the generalised norm evaluated at κ = 0.
        sd[a].norm = fp_generalized_norm<double>(0, b, b, 0.0, 0.0, 128);
        sd[a].coeff = coeff_value / (a + 1.0);
        sd[a].exp_rate = b * b;
    }
    return sd;
}

} // namespace

// --- 1. The returned eigenpairs must satisfy the QEP to working precision. ---
TEST_CASE(qep_residual_is_zero) {
    const unsigned K = 4;
    const double kappa = 0.04; // Pe = 5
    auto sd = make_series(K, 1.0);

    std::vector<QEPBlock<double>> blocks;
    qep_build_and_solve<double>(sd, K, kappa, 0.0, blocks, 128);
    REQUIRE(blocks.size() == 1);
    const QEPBlock<double> &blk = blocks[0];
    REQUIRE(blk.lambda.size() == K);

    // Rebuild K̃_ab = U_ab/(N_a N_b) and B = diag(β²) independently.
    std::vector<double> N(K), beta2(K);
    for (unsigned a = 0; a < K; ++a) {
        N[a] = std::sqrt(sd[a].norm);
        beta2[a] = sd[a].root * sd[a].root;
    }
    std::vector<std::vector<double>> Kt(K, std::vector<double>(K, 0.0));
    for (unsigned a = 0; a < K; ++a)
        for (unsigned b = 0; b < K; ++b)
            Kt[a][b] = fp_unweighted_overlap<double>(0, sd[a].root, sd[a].root,
                                                     sd[b].root, sd[b].root, 128)
                       / (N[a] * N[b]);

    for (unsigned j = 0; j < K; ++j) {
        const double lam = blk.lambda[j];
        REQUIRE(lam > 0.0);                       // decaying branch only

        // residual_a = Σ_b (κλ²K̃_ab) v_b + λ v_a − β²_a v_a
        double res_max = 0.0, v_max = 0.0;
        for (unsigned a = 0; a < K; ++a) {
            double s = 0.0;
            for (unsigned b = 0; b < K; ++b) s += Kt[a][b] * blk.V[b][j];
            const double res = kappa * lam * lam * s + lam * blk.V[a][j] - beta2[a] * blk.V[a][j];
            res_max = std::max(res_max, std::fabs(res));
            v_max = std::max(v_max, std::fabs(blk.V[a][j]));
        }
        // Scale the residual by the eigenvector magnitude and the problem scale.
        const double scale = v_max * (1.0 + beta2[K - 1]);
        REQUIRE(res_max / scale < 1e-10);
    }
}

// --- 2. The 1x1 block must reproduce the closed-form root of the tex. --------
// κ α λ² + λ − β² = 0  with α = U_aa/N²_a  (eq. seed_quadratic), whose physical
// root is λ = 2β² / (1 + sqrt(1 + 4καβ²))  (eq. seed_root).
TEST_CASE(qep_single_mode_matches_closed_form) {
    const double kappa = 0.04;
    auto sd = make_series(1, 1.0);

    std::vector<QEPBlock<double>> blocks;
    qep_build_and_solve<double>(sd, 1, kappa, 0.0, blocks, 128);
    REQUIRE(blocks.size() == 1);

    const double b = sd[0].root;
    const double beta2 = b * b;
    const double U = fp_unweighted_overlap<double>(0, b, b, b, b, 128);
    const double alpha = U / sd[0].norm;
    const double expected = 2.0 * beta2 / (1.0 + std::sqrt(1.0 + 4.0 * kappa * alpha * beta2));

    REQUIRE_APPROX(blocks[0].lambda[0], expected, 1e-10, 1e-12);
    // Sanity: axial diffusion strictly lowers the rate (tex eq. rate_ordering).
    REQUIRE(blocks[0].lambda[0] < beta2);
}

// --- 3. Bare limit: as κ → 0 the QEP rates collapse onto β². ----------------
TEST_CASE(qep_bare_limit_rates) {
    const unsigned K = 4;
    auto sd = make_series(K, 1.0);

    std::vector<QEPBlock<double>> blocks;
    qep_build_and_solve<double>(sd, K, 1e-10, 0.0, blocks, 128);
    REQUIRE(blocks.size() == 1);

    for (unsigned a = 0; a < K; ++a) {
        const double beta2 = sd[a].root * sd[a].root;
        REQUIRE_APPROX(blocks[0].lambda[a], beta2, 1e-6, 1e-6);
    }
}

// --- 4. Reconstruction at x = 0 returns the inlet coefficients exactly. ------
// The amplitudes solve V a = ĉ(0), so evaluating the modal sum at x = 0 must
// return the bare inlet coefficients it was built from.
TEST_CASE(qep_inlet_reconstruction) {
    const unsigned K = 4;
    const double kappa = 0.04;
    auto sd = make_series(K, 2.5);

    std::vector<QEPBlock<double>> blocks;
    qep_build_and_solve<double>(sd, K, kappa, 0.0, blocks, 128);
    const QEPBlock<double> &blk = blocks[0];

    for (unsigned a = 0; a < K; ++a) {
        double chat = 0.0;                       // ĉ_a(0) = Σ_j V[a][j] amp[j]
        for (unsigned j = 0; j < K; ++j) chat += blk.V[a][j] * blk.amp[j];
        const double C_a = chat / blk.norm[a];   // back to the physical coefficient
        REQUIRE_APPROX(C_a, sd[a].coeff, 1e-9, 1e-11);
    }
}
