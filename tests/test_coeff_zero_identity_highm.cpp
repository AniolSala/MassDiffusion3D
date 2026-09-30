#include "tinytest.h"

#define private public
#define protected public
#include "CDStratifiedSolution/CDStratifiedSolution.h"
#undef protected
#undef private

#include <cmath>
#include <sstream>
#include <vector>

// Symmetric inlet zi=[0.0], ui=[1.0, 0.0]. By weighted radial orthogonality
// of R_{0,m} against the constant R_{0,0} under the measure r(1-r^2) dr,
// the projected coefficients C_{0,m} for m>0 must all vanish.
//
// This fixture evaluates at x=0 -- where exp(-beta_m^2 x) = 1 so the engine
// populates every (0, m) mode up to its max_m and the test exercises the full
// dense column.
// WHY THIS SELECTS A 300-NODE RULE. The residual measured here is quadrature
// aliasing in the bare Gaussian projection, not an error in the projection
// formula. The constructor defaults pair max_root = 400 with a 100-node radial
// rule, and at that cutoff the n = 0 block retains exactly 100 modes: the most
// oscillatory retained mode then has as many half-waves over [0,1] as the rule
// has nodes, so the rule can no longer separate it from the constant. Measured
// (long double, this fixture, varying only the cutoff, worst |C_{0,m}| over
// m >= 1):
//
//   retained modes   modes/nodes   worst residual
//         80             0.80          2.2e-13
//         85             0.85          8.3e-10
//         90             0.90          2.1e-07
//         95             0.95          4.6e-05
//        100             1.00          4.9e-03
//
// Round-off until the retained count reaches ~0.8 of the node count, then
// geometric growth -- so the 1e-9 tolerance below first breaks at m = 85, which
// is what this test used to report, while the worst case sat at m = 99. Raising
// the rule at the SAME cutoff removes it entirely: 100 nodes give 4.9e-03,
// 300 nodes 1.4e-17, 400 nodes 1.3e-17, i.e. the long-double floor. The
// tolerance is therefore left at its original 1e-9 (not loosened) and the
// configuration is moved inside the regime the rule supports.
//
// The cap is a property of the bare projection alone. The finite-Peclet path
// does not use this quadrature (it assembles its own Gauss-Jacobi Gram rule and
// rejects an under-resolved block outright, see fp_gram_node_count), and the
// affected coefficients belong to modes decaying as exp(-beta^2 x) with
// beta ~ 400, so they matter only in the inlet plane itself.

TEST_CASE(test_n0_zero_identity_dense_highm) {
    const std::vector<long double> zi = {0.0L};
    const std::vector<long double> ui = {1.0L, 0.0L};

    CDStratifiedSolution<long double> sol(zi, ui, 0);
    // 100 nodes cannot resolve the 100 modes the default cutoff retains; see above.
    sol.set_number_of_gauss_points(300);
    sol.setup_bare_solution();
    // Trigger the dense coefficient computation by evaluating at x=0.
    const std::vector<long double> x0 = {0.0L};
    const std::vector<long double> r0 = {0.0L};
    const std::vector<long double> phi0 = {0.0L};
    sol.get_solution(x0, r0, phi0);

    const auto coeffs = sol.get_coefficients();

    REQUIRE(!coeffs.empty());
    REQUIRE(coeffs[0].size() >= 100);

    const long double abs_tol = 1e-9L;
    long double max_seen = 0.0L;
    unsigned worst_m = 0;
    for (unsigned m = 1; m < coeffs[0].size(); ++m) {
        long double v = std::abs(coeffs[0][m]);
        if (v > max_seen) {
            max_seen = v;
            worst_m = m;
        }
        if (v > abs_tol) {
            std::ostringstream os;
            os << "|C_{0," << m << "}| = " << static_cast<double>(v)
               << " exceeds " << static_cast<double>(abs_tol)
               << " (expected 0 by weighted orthogonality)";
            throw tinytest::Failure(os.str());
        }
    }
    // Sanity: the test must have actually iterated to high m (otherwise it
    // is vacuous).
    if (coeffs[0].size() < 90) {
        std::ostringstream os;
        os << "coefficient column too short (" << coeffs[0].size()
           << "); test is vacuous";
        throw tinytest::Failure(os.str());
    }
    // Echo the worst case so a passing run still records the residual floor.
    std::cout << "  max |C_{0,m}| over m=1.." << (coeffs[0].size() - 1)
              << " = " << static_cast<double>(max_seen)
              << " at m=" << worst_m << std::endl;
}
