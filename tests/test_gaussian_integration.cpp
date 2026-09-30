#include "tinytest.h"

#include "comp_utils.h"
#include "io_data.h"
#include "math_functions.h"
#include "roots_and_norms_calculations.h"

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

void psinm_r_norm(unsigned k, long double b, long double r, long double &val) {
    long double psi_val = 0.0L;
    psinm_r(k, b, r, psi_val);
    val = psi_val * psi_val;
}

} // namespace

TEST_CASE(test_gaussian_integration_matches_norms) {
    std::vector<std::vector<long double>> roots;
    std::vector<std::vector<long double>> norms;
    std::vector<std::vector<long double>> gaussian_data;

    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_ROOTS_FILE), roots);
    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_NORMS_FILE), norms);
    readValuesFromFile(data_path("src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_300.txt"), gaussian_data);

    std::vector<long double> weights;
    std::vector<long double> nodes;
    weights.reserve(gaussian_data.size());
    nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data) {
        weights.push_back(row[0]);
        nodes.push_back(row[1]);
    }

    const unsigned n = 32;
    const unsigned m = 21;
    const long double root = roots[n][m];
    auto integrand = [n, root](long double r, long double &val) { psinm_r_norm(n, root, r, val); };

    long double result = 0.0L;
    gaussian_integration(integrand, weights, nodes, result);

    const long double expected = norms[n][m];
    const long double rel_tol = 1e-8L;
    const long double abs_tol = 1e-12L;

    if (!tinytest::approx_equal(result, expected, rel_tol, abs_tol)) {
        std::ostringstream os;
        os << "norm mismatch for (" << n << "," << m << ") result=" << result
           << ", expected=" << expected;
        throw tinytest::Failure(os.str());
    }
}

TEST_CASE(test_gaussian_integration_product_r_reference_value) {
    std::vector<std::vector<long double>> roots;
    std::vector<std::vector<long double>> gaussian_data;

    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_ROOTS_FILE), roots);
    readValuesFromFile(data_path("src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_300.txt"), gaussian_data);

    std::vector<long double> weights;
    std::vector<long double> nodes;
    weights.reserve(gaussian_data.size());
    nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data) {
        weights.push_back(row[0]);
        nodes.push_back(row[1]);
    }

    const unsigned n = 3;
    const unsigned m1 = 21;
    const unsigned m2 = 37;
    const long double b1 = roots[n][m1];
    const long double b2 = roots[n][m2];

    const long double result = get_integration_product_r(n, b1, b2, weights, nodes);
    const long double expected = 6.0094414315468076564 * 1e-13;
    const long double rel_tol = 1e-8L;
    const long double abs_tol = 1e-36L;

    if (!tinytest::approx_equal(result, expected, rel_tol, abs_tol)) {
        std::ostringstream os;
        os << "integration product mismatch for (n=" << n << ",m1=" << m1 << ",m2=" << m2
           << ") result=" << result << ", expected=" << expected;
        throw tinytest::Failure(os.str());
    }
}
