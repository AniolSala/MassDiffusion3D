#ifndef FINITE_PECLET_RHS_REPRESENTER_RADIAL_H
#define FINITE_PECLET_RHS_REPRESENTER_RADIAL_H

// ---------------------------------------------------------------------------
// Inlet-specific. Builds the L2_r representer psi of an interface-cap
// functional, so that every mode's load-vector entry is one weighted dot
// product against the samples the radial Gram assembly already holds. This is
// the unweighted counterpart of finite_peclet_rhs_representer.h.
//
// This module NEVER calls fp_radial_factor -- avoiding those calls is its
// entire purpose. Measured on the weighted path, that turns the load-vector
// pass from ~45 s into ~0.06 s at the max_root = 600 production truncation;
// the whole 4x cost gap between the two projections was this one missing
// backend, not the projection.
//
// THE STRUCTURAL FACT THAT MAKES THIS A NEAR-COPY. Dividing each cap kernel by
// its own Gram weight, the factor (1-s) cancels from numerator and denominator
// in the weighted case and is absent from both in the unweighted one, so the
// EXACT representer is the same function for both projections:
//
//     weighted cap kernel        unweighted cap kernel
//     -------------------   =    ---------------------   =  Psi^ex(s; a)
//        (1-s) s^n                       s^n
//
// (theory/theory_computational_implementation, Lemma "the exact representer is
// common to both projections"). The two projections are therefore asking for a
// polynomial approximation of the SAME function and differ only in the inner
// product in which that approximation is taken. Everything that depends on the
// function alone transfers exactly; everything that depends on the measure
// changes. Concretely:
//
//   TRANSFERS EXACTLY   the cap rule and its cache (the eta^{1/2} weight is cap
//                       geometry, not projection); the degenerate branches
//                       (empty / center / zero); the signed-cap reflection; the
//                       1/a form of the n = 0 order rule, whose scaling comes
//                       from the branch points of the SAME arctan factor; the
//                       shallow-cap doubling guard, threshold and tolerance;
//                       and the constant full-disk representer sqrt(2 pi)/2,
//                       which is numerically identical on both paths.
//   CHANGES             the polynomial family, (1,n) -> (0,n), and its norms,
//                       which become the elementary 1/(2j+n+1); the n >= 1
//                       moment order, which drops by one degree because the
//                       kernel lost its (1-eta) factor; and the CONSTANT in the
//                       n = 0 order rule -- see fp_representer_order_radial.
//
// The design contract of finite_peclet_rhs_radial.h applies here unchanged:
// everything inlet-specific lives in this module and in that one, never in a
// Gram backend.
// ---------------------------------------------------------------------------

#include <vector>
#include "series_term_struct.h"
#include "finite_peclet_rhs.h"          // FPGaussJacobiRule / make_fp_gauss_jacobi_rule
#include "finite_peclet_rhs_radial.h"   // fp_rhs_radial_build_cap_kernel
#include "finite_peclet_gram_radial.h"

template <typename Ttype>
struct FPRepresenterRadialOptions
{
    unsigned representer_margin    = 40u;
    Ttype    shallow_cap_threshold = static_cast<Ttype>(0.1L);   // n == 0 guard
    Ttype    shallow_cap_tolerance = static_cast<Ttype>(1e-9L);
};

// Quadrature order for the representer of one cap, in the L2_r measure.
//
// n >= 1: EXACT, and one degree cheaper than the weighted rule. The moment
// integrand is eta^{1/2} times a polynomial of degree floor((n-1)/2) + j,
// against the weighted path's 1 + floor((n-1)/2) + j -- the extra degree there
// being the (1-eta) factor this kernel does not carry. An `order`-point
// Gauss-Jacobi rule is exact to degree 2*order - 1, so the requirement is
//     order >= ceil( (J + floor((n-1)/2)) / 2 ) ,
// one less than the weighted `inner`. Reusing the weighted rule here would be
// conservative and correct; this is simply the tight form.
//
// n == 0: CALIBRATED, and the calibration constant is PROVISIONAL. The
// non-polynomial factor is arctan(sqrt(T*eta)/a)/sqrt(eta), whose nearest
// singularities sit at eta = -a^2/T, just off the interval and approaching the
// origin as a -> 0. Mapping [0,1] to [-1,1] puts them on the Bernstein ellipse
// of parameter rho with ln(rho) -> 2a, and an N-point Gauss rule on a function
// analytic inside that ellipse converges as rho^{-2N}; hence N >~
// ln(1/eps)/(4a), which is where the 1/a term comes from. That derivation
// depends on the FUNCTION, which the lemma above says is shared, so the 1/a
// FORM transfers exactly.
//
// The constant does not. It multiplies rho^{-2N} through the polynomial
// factor, which differs between the two measures, and a change of that
// constant shifts N by ln(C)/(4a) -- not negligible at small a. A direct study
// of the moments measured the unweighted kernel needing consistently FEWER
// nodes than the weighted one at equal accuracy, by 0 to 2 nodes (3-5% in the
// constant), so reusing c_shallow = 6 here is expected to be conservative. That
// is an expectation, not a measurement, and what makes reusing it safe rather
// than reckless is the mandatory doubling guard in fp_cap_representer_radial:
// an insufficient order raises, it does not silently degrade. Re-calibration,
// when it happens, must be done on the COEFFICIENT vector and not on the
// moments or the load vector -- see the header comment there.
template <typename Ttype>
unsigned fp_representer_order_radial(unsigned angular_index, unsigned coefficient_count,
                                     const Ttype &absolute_interface,
                                     const FPRepresenterRadialOptions<Ttype> &options);

// psi(s_q) for one angular block and one absolute interface, evaluated at
// `gram_nodes` (the block's own (0,n) Gram quadrature nodes).
//   coefficient_count : J. MUST equal the Gram node count N, so that the two
//                       quadratures certify the same polynomial degree.
// Degenerate caps: `empty`/`zero` return an all-zero vector; `center`
// (n = 0, a = 0) returns an EMPTY vector and the caller must substitute the
// half full-disk value -- identical to the weighted path, because those
// branches are cap geometry and carry no radial weight.
template <typename Ttype>
std::vector<Ttype> fp_cap_representer_radial(unsigned angular_index, const Ttype &absolute_interface,
                                             unsigned coefficient_count,
                                             const std::vector<Ttype> &gram_nodes,
                                             const FPRepresenterRadialOptions<Ttype> &options);

// Contract a representer against the radial Gram's retained samples:
//   out[m] = sum_q quadrature_weights[q] * radial_samples[q*K + m] * psi[q].
// NO factor of 1/2 here, exactly as on the weighted path: the (0,n) rule's mass
// is int_0^1 s^n ds, so this sum is int_0^1 s^n G_m psi ds, and the 1/2 that
// r dr = ds/2 contributes is already inside the cap functional's own prefactor.
// (The Gram entry U_ij = 1/2 int s^n G_i G_j ds does carry it, which is why the
// two look inconsistent and are not.)
// Throws if factor.samples_retained is false.
template <typename Ttype>
std::vector<Ttype> fp_representer_contract_radial(const FPGramRadialFactor<Ttype> &factor,
                                                  const std::vector<Ttype> &psi);

// Stratified load vector, representer route. Signature deliberately mirrors
// fp_rhs_radial_stratified_inlet so the two are drop-in comparable in the
// driver -- and so the direct route can serve as the oracle in cross-validation.
//   full_disk_column : when non-empty and angular_index == 0, used verbatim as
//                      the exact full-disk term
//                      (fp_rhs_radial_full_disk_from_gram_column).
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_stratified_inlet_representer(
    unsigned angular_index,
    const FPGramRadialFactor<Ttype> &factor,
    const std::vector<Ttype> &interface_positions,
    const std::vector<Ttype> &layer_values,
    const FPRepresenterRadialOptions<Ttype> &options,
    const std::vector<Ttype> &full_disk_column = std::vector<Ttype>());

// Uniform inlet (Graetz), representer route: the constant representer
// sqrt(2*pi)/2 for n == 0, an exact zero vector for n > 0.
template <typename Ttype>
std::vector<Ttype> fp_rhs_radial_uniform_inlet_representer(
    unsigned angular_index,
    const FPGramRadialFactor<Ttype> &factor,
    const FPRepresenterRadialOptions<Ttype> &options,
    const std::vector<Ttype> &full_disk_column = std::vector<Ttype>());

#endif // FINITE_PECLET_RHS_REPRESENTER_RADIAL_H
