#ifndef FINITE_PECLET_COEFFICIENTS_H
#define FINITE_PECLET_COEFFICIENTS_H

// ---------------------------------------------------------------------------
// finite_peclet_coefficients — modified inlet coefficients for the exact
// finite-Peclet solution.
//
// ---------------------------------------------------------------------------
// Why this is a linear solve and not a division
// ---------------------------------------------------------------------------
// theory/analytical_solution.tex Sec. 5.3 derives a closed form (eq.
// coefficients_fp) in which each coefficient is an independent quadrature,
//     Chat_m = int (omega + kappa*Lam_m) f R_m / N2_fp_m .
// That derivation does not hold. The bi-orthogonality it invokes (eq.
// biorthogonality_fp) is TRUE but its weight is PAIR-DEPENDENT,
// omega + kappa*(Lam_m + Lam_k); a projection needs ONE fixed test functional
// per mode that annihilates every other mode simultaneously, and the fixed
// weight omega + kappa*Lam_m does not do that. What it actually leaves is
//
//     int (omega + kappa*Lam_m) R_m R_k r dr  =  -kappa * Lam_k * U_mk   (m != k)
//
// (verified numerically to ratio 1.000000). The neglected term is not small:
// kappa*Lam_m is O(1) for the higher modes, and the resulting inlet error does
// NOT converge under mode refinement (it plateaus near 0.35 at Pe = 5).
//
// Root cause: with kappa > 0 the axial problem is second order, a quadratic
// pencil. Its genuine orthogonality lives in the linearised (doubled) space and
// needs both psi(0) and psi_x(0). On the semi-infinite pipe only psi(0) is
// prescribed, so the decaying modes {R_m} form a perfectly good but NON-
// ORTHOGONAL basis -- and a non-orthogonal basis requires a linear solve.
//
// ---------------------------------------------------------------------------
// What is computed here (method "B")
// ---------------------------------------------------------------------------
// The L2_omega Galerkin projection of the inlet onto the modified basis: find
// Chat minimising || sum_m Chat_m R_m - f ||_{L2_omega}, i.e. solve
//
//     sum_k W_mk Chat_k = b_m ,
//     W_mk = int_0^1 omega(r) R_m(r) R_k(r) r dr        (Gram matrix, SPD)
//     b_m  = int omega(r) f R_m Phi_n dOmega            (same weight as the bare case)
//
// one dense solve per angular index n (modes with different n stay uncoupled
// through Phi_n). Properties:
//   * kappa -> 0 : R_m -> the bare modes, W becomes DIAGONAL, and the system
//     collapses exactly to the classical formula C_m = b_m / N_m^2.
//   * it is the best approximation from span{R_m} in the L2_omega norm, so the
//     inlet error decreases monotonically under mode refinement.
//   * the generalised norm N2_fp (eq. norm_fp) is no longer used to normalise
//     the coefficients; it is retained only as diagnostic output. The relevant
//     normalisation is W_mm, the plain omega-norm of the modified mode.
//
// Numerical stability: W is a Gram matrix of nearly-orthogonal functions and is
// well conditioned in practice (cond ~ 8 at K = 15 up to ~56 at K = 100 for
// Pe = 5), but it is symmetrically equilibrated by its diagonal before
// factorisation, which normalises the diagonal to 1 and removes any scaling
// disparity between modes. The solve is Cholesky (W is SPD by construction)
// with an automatic fall-back to partial-pivot LU should positive-definiteness
// be lost to round-off, so a near-degenerate pair of rates can never silently
// produce garbage.
//
// ---------------------------------------------------------------------------
// The alternative: projecting in L2_r instead
// ---------------------------------------------------------------------------
// finite_peclet_coefficients_radial.h implements the SAME Galerkin solve in
// L2_r (weight r dr) rather than L2_omega (weight omega r dr), selected by
// CDBaseSolution::set_projection_space(ProjectionSpace::Radial) -- the default
// is unchanged. The two differ only in the weight defining the Gram matrix and
// the load vector; after the substitution s = r^2 that is exactly one factor
// (1-s), i.e. the Jacobi parameter alpha of the quadrature measure.
//
// Why one might want it: the "kappa -> 0 makes W diagonal" property noted above
// is the same fact that makes L2_omega the WORSE-conditioned choice at finite
// truncation. The modified modes are a Riesz basis of L2_r but not of
// L2_omega (omega vanishes at the wall, so multiplication by omega is positive
// but not boundedly invertible), and lambda_min of the normalised W decays like
// 1/K while that of the normalised U is flat in K. See
// finite_peclet_coefficients_radial.h for the measured numbers. Both RHS
// backends work there; only the ultraspherical Gram backend does not carry over.
//
// These are the profile-specific free functions the solver classes delegate to;
// each writes coeff_fp. Prerequisite: root_fp/btilde_fp/rate_fp must already be
// populated (fp_modify_roots). Instantiated for double and long double.
// ---------------------------------------------------------------------------

#include <vector>

#include "series_term_struct.h"
#include "finite_peclet_radial_block.h"

// Finite-Peclet coefficients for the isothermal Graetz problem (uniform inlet
// f == 1, Dirichlet walls; only the axisymmetric n = 0 modes are excited).
// @param kappa   Pe^-2 (>= 0).
// @param max_K   number of active series terms.
// @param series_data  per-mode records; coeff_fp is written in place.
// @param panels  composite Gauss-Legendre panels for the radial quadratures.
template <typename Ttype>
void fp_graetz_coefficients(const Ttype &kappa, unsigned max_K,
                            std::vector<SeriesTermData<Ttype>> &series_data,
                            unsigned panels = 64u);

template <typename Ttype>
void fp_graetz_coefficients_from_table(const Ttype &kappa, unsigned max_K,
                                       std::vector<SeriesTermData<Ttype>> &series_data,
                                       const FPRadialTable<Ttype> &table);

// Finite-Peclet coefficients for the stratified inlet (piecewise-constant layers).
// @param zi      interface positions (ascending, in (-1,1)); size = layers-1.
// @param ui      per-layer scaled values; size = layers. ui[0] is the baseline
//                picked up by the constant mode, which is solved for as an
//                ordinary coefficient (not as a separate O(kappa) shift).
// @param kappa   Pe^-2 (>= 0).
// @param max_K   number of active series terms.
// @param series_data  per-mode records; coeff_fp is written in place.
// @param panels  composite Gauss-Legendre panels for the radial quadratures.
template <typename Ttype>
void fp_stratified_coefficients(const std::vector<Ttype> &zi,
                                const std::vector<Ttype> &ui,
                                const Ttype &kappa, unsigned max_K,
                                std::vector<SeriesTermData<Ttype>> &series_data,
                                unsigned panels = 64u);

template <typename Ttype>
void fp_stratified_coefficients_from_table(const std::vector<Ttype> &zi,
                                           const std::vector<Ttype> &ui,
                                           const Ttype &kappa, unsigned max_K,
                                           std::vector<SeriesTermData<Ttype>> &series_data,
                                           const FPRadialTable<Ttype> &table);

#endif // FINITE_PECLET_COEFFICIENTS_H
