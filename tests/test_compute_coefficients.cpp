#include "tinytest.h"

#include "coefficients_computation_gaussian.h"
#include "io_data.h"
#include "math_functions.h"
#include "series_term_struct.h"

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

TEST_CASE(test_compute_coefficients_small) {
    const std::vector<double> z_vec = {-0.1, 0.33};
    const std::vector<double> values = {0.0, 0.2123, 1.0};

    std::vector<std::vector<double>> expected;
    readValuesFromFile(data_path("tests/data/coefs_values_zi_neg_5x5.txt"), expected);

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

    const unsigned n_max = 4;
    const unsigned m_max = 4;
    std::vector<std::vector<double>> coefficients(
        n_max + 1,
        std::vector<double>(m_max + 1, 0.0));

    std::vector<SeriesTermData<double>> series_data;
    series_data.reserve((n_max + 1) * (m_max + 1));
    for (unsigned n = 0; n <= n_max; n++) {
        for (unsigned m = 0; m <= m_max; m++) {
            SeriesTermData<double> term;
            term.n = n;
            term.m = m;
            term.root = roots[n][m];
            term.norm = norms[n][m];
            term.coeff = 0.0;
            series_data.push_back(term);
        }
    }

    compute_coefficients_matrix<double>(
        z_vec,
        values,
        gauss_weights,
        gauss_nodes,
        static_cast<unsigned>(series_data.size()),
        series_data);

    for (const auto &term : series_data) {
        if (term.n <= n_max && term.m <= m_max) {
            coefficients[term.n][term.m] = term.coeff;
        }
    }

    const double rel_tol = 1e-7;
    const double abs_tol = 1e-10;

    for (unsigned n = 0; n <= n_max; n++) {
        for (unsigned m = 0; m <= m_max; m++) {
            if (n == 0 && m == 0) {
                continue;
            }
            const double result = coefficients[n][m];
            const double exp_val = expected[n][m];
            if (!tinytest::approx_equal(result, exp_val, rel_tol, abs_tol)) {
                std::ostringstream os;
                os << "coef mismatch (" << n << "," << m << ") result=" << result
                   << ", expected=" << exp_val;
                throw tinytest::Failure(os.str());
            }
        }
    }
}
