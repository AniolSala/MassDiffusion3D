#ifndef FINITE_PECLET_NORMS_BOUNDARY_H
#define FINITE_PECLET_NORMS_BOUNDARY_H

// Quadrature-free finite-Peclet generalised norms, obtained from the derivative
// of the wall characteristic.  The exact Neumann n=0 zero mode is 1/4.

#include <vector>
#include "finite_peclet_roots.h"
#include "series_term_struct.h"

/** Return the positive generalised norm from the boundary-characteristic
 * identity. lambda is a finite nonnegative rate and kappa is nonnegative.
 * Throws if numerical differentiation cannot produce a stable positive norm.
 */
template <typename Ttype>
Ttype fp_generalized_norm_boundary(unsigned angular_index, const Ttype &lambda,
                                   const Ttype &kappa, WallCondition wall_condition);

/** Fill norm_fp for the first max_K finite-Peclet modes without quadrature.
 * root_fp/rate_fp must have been populated. Supports double and long double.
 */
template <typename Ttype>
void fp_compute_norms_boundary(std::vector<SeriesTermData<Ttype>> &series_data,
                               unsigned max_K, const Ttype &kappa,
                               WallCondition wall_condition);

#endif
