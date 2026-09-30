#ifndef FINITE_PECLET_RADIAL_H
#define FINITE_PECLET_RADIAL_H

// ---------------------------------------------------------------------------
// finite_peclet_radial — modified radial eigenfunction for the exact
// finite-Peclet (axial-diffusion) solution.
//
// Theory: theory/analytical_solution.tex, Sec. 5.2 ("Exact normal modes for
// finite Pe"), eqs. (radial_solution_fp), (b_and_btilde_def),
// (char_dirichlet_fp), (char_neumann_fp).
//
// Retaining the axial-diffusion term -kappa*d_xx psi (kappa = Pe^-2) deforms the
// bare radial ODE
//     (1/r)(r R')' - (n^2/r^2) R + beta^2 (1-r^2) R = 0
// into the finite-Peclet radial ODE
//     (1/r)(r R')' - (n^2/r^2) R + [kappa*Lam^2 + Lam*(1-r^2)] R = 0 ,
// which is the SAME confluent-hypergeometric (Kummer) equation with only the
// FIRST Kummer parameter shifted. With
//     b  = sqrt(Lam)                       (unshifted; sets prefactor + argument)
//     bt = b*(1 + kappa*Lam) = b + kappa*b^3   (shifted; sets the Kummer parameter)
// the regular solution is
//     R_nm(r) = r^n e^{-1/2 b r^2} 1F1( (2n+2-bt)/4 , n+1 ; b r^2 ) .
//
// Passing bt == b reproduces the bare radial mode psinm_r(n, b, r) EXACTLY, so
// these primitives also serve the axial-diffusion-free ("bare") solution as the
// special case kappa -> 0. This is the structural fact that lets a single
// evaluation path serve both solutions.
//
// These are pure math primitives: no solver state, mirroring the style of
// math_functions.{h,cpp} (which holds the bare-mode counterparts). Instantiated
// for double and long double.
// ---------------------------------------------------------------------------

// Radial eigenfunction R_nm(r) of the finite-Peclet problem (eq. radial_solution_fp).
// @param n  angular index.
// @param b  = sqrt(Lam), the unshifted parameter (Gaussian prefactor + 1F1 argument).
// @param bt = b*(1+kappa*Lam), the shifted first Kummer parameter carrier.
// @param r  radial coordinate in [0,1].
// @return   R_nm(r). Reduces to psinm_r(n,b,r) when bt == b.
template <typename Ttype>
Ttype psinm_r_fp(const unsigned &n, const Ttype &b, const Ttype &bt, const Ttype &r);

/** Direct finite-Peclet radial factor G_nm(s), with s=r*r in [0,1].
 *  psinm_r_fp(n,b,bt,r) equals r^n times this value. */
template <typename Ttype>
Ttype fp_radial_factor(const unsigned &n, const Ttype &b, const Ttype &bt,
                       const Ttype &squared_radius);

// In-place overload of psinm_r_fp; writes R_nm(r) into `out`.
template <typename Ttype>
void psinm_r_fp(const unsigned &n, const Ttype &b, const Ttype &bt, const Ttype &r, Ttype &out);

// Dirichlet characteristic value R_nm(1) (eq. char_dirichlet_fp, up to the sign
// of the common e^{-b/2} factor). Its zeros in Lam are the Dirichlet eigenvalues.
template <typename Ttype>
Ttype psi_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt);

// Neumann characteristic value (n+1)*R_nm'(1) (eq. char_neumann_fp). Its zeros in
// Lam are the Neumann eigenvalues.
template <typename Ttype>
Ttype dpsi_dr_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt);

/** Actual radial derivative R_nm'(1).
 *
 * dpsi_dr_at_1_fp returns the Neumann characteristic (n+1) R_nm'(1);
 * this function removes that normalization.
 */
template <typename Ttype>
Ttype radial_derivative_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt);

// Convenience: bt = b*(1 + kappa*b^2) from (b, kappa). Note b^2 = Lam, so this is
// exactly bt = b*(1 + kappa*Lam) of eq. (b_and_btilde_def).
template <typename Ttype>
Ttype btilde_from_b(const Ttype &b, const Ttype &kappa);

#endif // FINITE_PECLET_RADIAL_H
