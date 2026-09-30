#ifndef FINITE_PECLET_RHS_REPRESENTER_H
#define FINITE_PECLET_RHS_REPRESENTER_H

// ---------------------------------------------------------------------------
// Inlet-specific. Builds the L2_omega representer psi of an interface-cap
// functional, so that every mode's load-vector entry is one weighted dot
// product against the samples the Gram assembly already holds.
//
// This module NEVER calls fp_radial_factor -- avoiding those calls is its
// entire purpose. See plans/RHS_REPRESENTER_PLAN.md section 12.
// ---------------------------------------------------------------------------

#include <vector>
#include "series_term_struct.h"
#include "finite_peclet_rhs.h"
#include "finite_peclet_gram_gauss_jacobi.h"

template <typename Ttype>
struct FPRepresenterOptions
{
    unsigned representer_margin    = 40u;
    Ttype    shallow_cap_threshold = static_cast<Ttype>(0.1L);   // n == 0 guard
    Ttype    shallow_cap_tolerance = static_cast<Ttype>(1e-9L);
};

// Quadrature order for the representer of one cap. Exact for n >= 1; calibrated
// with a run-time doubling guard for n == 0. See section 6.2 of the plan.
template <typename Ttype>
unsigned fp_representer_order(unsigned angular_index, unsigned coefficient_count,
                              const Ttype &absolute_interface,
                              const FPRepresenterOptions<Ttype> &options);

// psi(s_q) for one angular block and one absolute interface, evaluated at
// `gram_nodes` (the block's own Gram quadrature nodes).
//   coefficient_count : J. MUST equal the Gram node count N -- see section 12.
// Degenerate caps: `empty`/`zero` return an all-zero vector; `center`
// (n = 0, a = 0) returns an EMPTY vector and the caller must substitute the
// half full-disk value, exactly as layer() does today.
template <typename Ttype>
std::vector<Ttype> fp_cap_representer(unsigned angular_index, const Ttype &absolute_interface,
                                      unsigned coefficient_count,
                                      const std::vector<Ttype> &gram_nodes,
                                      const FPRepresenterOptions<Ttype> &options);

// Contract a representer against the Gram's retained samples:
//   out[m] = sum_q quadrature_weights[q] * radial_samples[q*K + m] * psi[q].
// Throws if factor.samples_retained is false.
template <typename Ttype>
std::vector<Ttype> fp_representer_contract(const FPGramGaussJacobiFactor<Ttype> &factor,
                                           const std::vector<Ttype> &psi);

// Stratified load vector, representer route. Signature deliberately mirrors
// fp_rhs_stratified_inlet so the two are drop-in comparable in the driver.
//   full_disk_column : when non-empty and angular_index == 0, used verbatim as
//                      the exact full-disk term (fp_rhs_full_disk_from_gram_column).
template <typename Ttype>
std::vector<Ttype> fp_rhs_stratified_inlet_representer(
    unsigned angular_index,
    const FPGramGaussJacobiFactor<Ttype> &factor,
    const std::vector<Ttype> &interface_positions,
    const std::vector<Ttype> &layer_values,
    const FPRepresenterOptions<Ttype> &options,
    const std::vector<Ttype> &full_disk_column = std::vector<Ttype>());

// Uniform inlet (Graetz), representer route: the constant representer of
// section 6.3 for n == 0, an exact zero vector for n > 0.
template <typename Ttype>
std::vector<Ttype> fp_rhs_uniform_inlet_representer(
    unsigned angular_index,
    const FPGramGaussJacobiFactor<Ttype> &factor,
    const FPRepresenterOptions<Ttype> &options,
    const std::vector<Ttype> &full_disk_column = std::vector<Ttype>());

#endif
