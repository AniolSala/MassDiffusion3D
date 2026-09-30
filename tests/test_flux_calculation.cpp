#include "tinytest.h"

#include "math_functions.h"

#include <sstream>

TEST_CASE(test_flux_calculation) {
    const double result = compute_flux_layer(-1.0, 1.0);
    const double expected = 1.5707963267948966;
    const double rel_tol = 1e-12;
    const double abs_tol = 1e-12;

    if (!tinytest::approx_equal(result, expected, rel_tol, abs_tol)) {
        std::ostringstream os;
        os << "flux layer mismatch result=" << result
           << ", expected=" << expected;
        throw tinytest::Failure(os.str());
    }
}
