#include "tinytest.h"

#include "coefficients_computation_gaussian.h"
#include "io_data.h"
#include "series_term_struct.h"
#include "math_functions.h"

#include "../src/CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#ifndef CDS_ROOTS_FILE
#define CDS_ROOTS_FILE "broots_85x85_cpp.txt"
#endif

#ifndef CDS_NORMS_FILE
#define CDS_NORMS_FILE "norms_85x85_cpp.txt"
#endif

namespace {

std::string data_path(const std::string &rel) {
    return std::string(TEST_DATA_DIR) + "/" + rel;
}

} // namespace

TEST_CASE(parseval_identity_test) {
    // const std::vector<double> z_vec = {-0.1, 0.33};
    // const std::vector<double> values = {0.0, 0.2123, 1.0};
    const std::vector<double> z_vec = {-0.11, 0.33};
    const std::vector<double> values = {1., 0.223423222, 0.0};
    // const std::vector<double> values = {0., 0.776576778, 1.0};

    // Compute expected value (Parseval's identity: \sum_{n,m} c_nm^2 = ||f(0)||^2)
    std::vector<double> z_vec_with_walls(z_vec.size() + 2, 0.0);
    for (unsigned n = 0; n < z_vec.size(); n++)
    {
        z_vec_with_walls[n + 1] = z_vec[n];
    }
    z_vec_with_walls[0] = -1.0;
    z_vec_with_walls[z_vec.size() + 1] = 1.0;
    
    long double total_flux = M_PI_2;
    long double c00 = 0.0;  // This is not included in the coefficients
    long double expected = 0.0;
    for(unsigned n = 0; n < values.size(); n++)
    {
        double z1 = z_vec_with_walls[n];
        double z2 = z_vec_with_walls[n + 1];
        double u_n = values[n];
        double flux_layer = compute_flux_layer(z1, z2);

        c00 += u_n * flux_layer;
        expected += u_n * u_n * flux_layer;
    }
    long double c00_2 = c00 * c00 / total_flux;
    expected -= c00_2;

    std::vector<std::vector<double>> roots;
    std::vector<std::vector<double>> norms;
    std::vector<std::vector<double>> gaussian_data;
    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_ROOTS_FILE), roots);
    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_NORMS_FILE), norms);
    readValuesFromFile(data_path("src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_100.txt"), gaussian_data);

    std::vector<double> gauss_weights;
    std::vector<double> gauss_nodes;
    gauss_weights.reserve(gaussian_data.size());
    gauss_nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data) {
        gauss_weights.push_back(row[0]);
        gauss_nodes.push_back(row[1]);
    }

    // This fixture projects with a fixed 100-point rule and sweeps the whole
    // catalogue rather than truncating on max_root the way the solver does, so
    // the mode set has to be capped by what that rule resolves. Measured with
    // the 200x200 tables: capping both indices at the node count reproduces the
    // historical sum 0.294673 (rel err 7.4e-4), letting m run to 199 aliases the
    // radial integral outright (sum 1.178, rel err 3.0), and letting n run to
    // 199 overshoots (rel err 2.5e-3). The solver is unaffected because
    // set_series_data truncates at max_root, and max_root = 400 keeps m <= 99.
    const unsigned quadrature_cap = static_cast<unsigned>(gauss_nodes.size());
    const unsigned n_max = std::min(static_cast<unsigned>(roots.size()), quadrature_cap) - 1;
    const unsigned m_max = std::min(static_cast<unsigned>(roots[0].size()), quadrature_cap) - 1;

    std::vector<SeriesTermData<double>> series_data;
    series_data.reserve((n_max + 1) * (m_max + 1));
    for (unsigned n = 0; n <= n_max; n++) {
        for (unsigned m = 0; m <= m_max; m++) {
            // Norms below the smallest double are widened to 0.0 by the reader
            // and compute_coefficients_matrix divides by the norm, so a mode
            // that underflowed would poison the sum with nan. None occur under
            // the cap above; this only keeps that failure mode impossible.
            if (!(norms[n][m] > 0.0)) continue;
            SeriesTermData<double> term;
            term.n = n;
            term.m = m;
            term.root = roots[n][m];
            term.norm = norms[n][m];
            term.coeff = 0.0;
            series_data.push_back(term);
        }
    }
    REQUIRE(series_data.size() > (n_max + 1) * (m_max + 1) / 2);

    compute_coefficients_matrix<double>(
        z_vec,
        values,
        gauss_weights,
        gauss_nodes,
        static_cast<unsigned>(series_data.size()),
        series_data);

    long double sum_sq = 0.0L;
    for (const auto &term : series_data) {
        const long double coeff = static_cast<long double>(term.coeff);
        const long double norm = static_cast<long double>(term.norm);
        sum_sq += coeff * coeff * norm;
    }

    const long double rel_tol = 1e-3L;
    const long double abs_tol = 1e-3L;

    if (!tinytest::approx_equal(sum_sq, expected, rel_tol, abs_tol)) {
        std::ostringstream os;
        long double rel_err = std::fabs(expected - sum_sq) / expected;
        os << "parseval identity mismatch sum=" << sum_sq
           << ", expected=" << expected
           << ", rel_error=" << rel_err;
        throw tinytest::Failure(os.str());
    }
}

TEST_CASE(parseval_identity_graetz_test) {
    // For the Graetz problem (T0=1, T_wall=0), the inlet profile is θ(r)=1 everywhere.
    // The 2D L2 norm with weight r(1-r²) over the full cross-section is:
    //   ||θ||² = ∫_0^{2π} ∫_0^1 r(1-r²)·1 dr dφ = 2π·(1/4) = π/2
    //
    // The stored coefficients absorb the √(2π) azimuthal normalisation factor,
    // so Parseval's identity reads:  ∑_m coeff_m² · N_m = π/2
    //
    // No zero-eigenvalue mode exists in the Graetz problem (Dirichlet BC).
    const long double expected = M_PI_2;

    CDGraetzIsothermalSolution<double> sol(1.0, 0.0, 0);
    sol.setup_bare_solution(); // small x retains all 100 modes

    const auto coefficients = sol.get_coefficients(); // shape [1][m] (n=0 only)
    const auto norms = sol.get_norms();               // shape [1][m]

    long double sum_sq = 0.0L;
    for (std::size_t m = 0; m < coefficients[0].size(); ++m) {
        const long double coeff = static_cast<long double>(coefficients[0][m]);
        const long double norm  = static_cast<long double>(norms[0][m]);
        sum_sq += coeff * coeff * norm;
    }

    // The Graetz inlet (θ=1 vs. Dirichlet wall θ=0) is discontinuous, so the
    // series converges slowly and 100 modes leave ~0.18 % truncation error.
    const long double rel_tol = 5e-3L;
    const long double abs_tol = 5e-3L;

    if (!tinytest::approx_equal(sum_sq, expected, rel_tol, abs_tol)) {
        std::ostringstream os;
        long double rel_err = std::fabs(expected - sum_sq) / expected;
        os << "Graetz parseval identity mismatch sum=" << sum_sq
           << ", expected=" << expected
           << ", rel_error=" << rel_err;
        throw tinytest::Failure(os.str());
    }
}
