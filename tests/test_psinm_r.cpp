#include "tinytest.h"

#include "io_data.h"
#include "math_functions.h"

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

namespace {

std::string data_path(const std::string &rel) {
    return std::string(TEST_DATA_DIR) + "/" + rel;
}

} // namespace

TEST_CASE(test_psinm_r_values_small) {
    std::vector<std::vector<double>> roots;
    std::vector<std::vector<double>> ref;

    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" CDS_ROOTS_FILE), roots);
    readValuesFromFile(data_path("tests/data/psinm_r_values.txt"), ref);

    const double r = 0.8;
    const unsigned n_max = 4;
    const unsigned m_max = 4;
    const double rel_tol = 1e-8;
    const double abs_tol = 1e-12;

    for (unsigned n = 0; n <= n_max; n++) {
        for (unsigned m = 0; m <= m_max; m++) {
            double result = 0.0;
            psinm_r(n, roots[n][m], r, result);

            const double expected = ref[n][m];
            if (!tinytest::approx_equal(result, expected, rel_tol, abs_tol)) {
                std::ostringstream os;
                os << "psinm_r(" << n << "," << m << ") = " << result
                   << ", expected " << expected;
                throw tinytest::Failure(os.str());
            }
        }
    }
}
