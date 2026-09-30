#include "tinytest.h"
#include "finite_peclet_norms_boundary.h"

TEST_CASE(fp_boundary_norm_exact_neumann_zero_mode) {
    REQUIRE_APPROX(fp_generalized_norm_boundary<double>(0, 0.0, 0.04, WallCondition::Neumann),
                   0.25, 1e-14, 1e-14);
}
