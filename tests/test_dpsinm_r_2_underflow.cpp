#include "tinytest.h"

#include "math_functions.h"

#include <cmath>
#include <sstream>

// Regression test for the unsigned-arithmetic underflow at
// math_functions.cpp dpsinm_r_2: the expression `r^(k-2)` was implemented
// as `std::pow(r, static_cast<Ttype>(k - 2))` with `k` of type unsigned.
// For k in {0, 1} the subtraction `k - 2` wraps to a huge positive
// integer, after which `std::pow(r<1, huge)` underflows to zero and
// silently zeroes the entire formula. The fixed code branches on signed
// k and uses 1/r or 1/(r*r) for k = 1 and k = 0 respectively.
//
// These tests confirm that:
//   (a) the function returns finite, non-zero values for k in {0, 1, 2}
//       at typical r in (0, 1);
//   (b) those values agree with a central finite-difference of psinm_r
//       to within ~1e-5 relative error.
// The current production root finder never calls dpsinm_r_2 (it uses
// ord = 1 in find_root and a finite-difference derivative for Newton),
// so these tests are defensive.

TEST_CASE(test_dpsinm_r_2_finite_nonzero_for_small_k) {
    const long double r_values[] = {0.3L, 0.5L, 0.8L};
    const long double beta_values[] = {5.0L, 10.0L, 20.0L};

    for (unsigned k = 0; k <= 2; ++k) {
        for (long double b : beta_values) {
            for (long double r : r_values) {
                const long double val = dpsinm_r<long double>(2, k, b, r);
                if (!std::isfinite(val)) {
                    std::ostringstream os;
                    os << "dpsinm_r_2 non-finite for k=" << k << " b=" << b
                       << " r=" << r << ": value=" << val;
                    throw tinytest::Failure(os.str());
                }
                // Pre-fix the function returned exactly 0.0 for k in {0,1};
                // post-fix it must be a real second derivative, which is
                // generically far from zero. Allow a tiny window in case
                // a specific (b, r) happens to land near a node, but the
                // pre-fix bug returned identically zero across the grid,
                // so any combination above ~1e-3 proves the fix is alive.
                if (k <= 1 && std::abs(val) < 1e-6L) {
                    std::ostringstream os;
                    os << "dpsinm_r_2 returned near-zero (" << val
                       << ") for k=" << k << " b=" << b << " r=" << r
                       << ", consistent with the unsigned-underflow bug.";
                    throw tinytest::Failure(os.str());
                }
            }
        }
    }
}

TEST_CASE(test_dpsinm_r_2_matches_finite_difference) {
    // Compare against the central finite-difference second derivative of
    // psinm_r at moderate r. h chosen to balance truncation vs roundoff
    // for long-double psinm_r evaluations.
    const long double h = 1e-3L;
    const long double rel_tol = 5e-5L;

    struct Case { unsigned k; long double b; long double r; };
    const Case cases[] = {
        {0, 5.067505500931330531325L, 0.5L},
        {0, 5.067505500931330531325L, 0.8L},
        {1, 6.7L,                       0.5L},
        {1, 6.7L,                       0.8L},
        {2, 9.0L,                       0.5L},
        {2, 9.0L,                       0.8L},
        {3, 12.0L,                      0.5L},
        {5, 18.0L,                      0.6L},
    };

    for (const auto &c : cases) {
        long double psi_p, psi_m, psi_c;
        psinm_r<long double>(c.k, c.b, c.r + h, psi_p);
        psinm_r<long double>(c.k, c.b, c.r - h, psi_m);
        psinm_r<long double>(c.k, c.b, c.r, psi_c);
        const long double fd2 = (psi_p - 2.0L * psi_c + psi_m) / (h * h);
        const long double analytic = dpsinm_r<long double>(2, c.k, c.b, c.r);

        const long double scale = std::max(std::abs(fd2), std::abs(analytic));
        const long double rel = scale > 0.0L
            ? std::abs(analytic - fd2) / scale
            : std::abs(analytic - fd2);
        if (rel > rel_tol) {
            std::ostringstream os;
            os << "dpsinm_r_2(k=" << c.k << ", b=" << c.b << ", r=" << c.r
               << ") = " << analytic
               << " disagrees with finite-difference d^2 psi / dr^2 = " << fd2
               << "; relative diff " << static_cast<double>(rel)
               << " > tol " << static_cast<double>(rel_tol);
            throw tinytest::Failure(os.str());
        }
    }
}
