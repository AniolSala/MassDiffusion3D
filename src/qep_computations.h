#ifndef QEP_COMPUTATIONS_H
#define QEP_COMPUTATIONS_H

// ---------------------------------------------------------------------------
// qep_computations — exact finite-Péclet solution via the Quadratic Eigenvalue
// Problem (QEP) posed in the *bare* (Pe-independent) Graetz eigenbasis.
//
// Reference: J. Neuhauser, J.-H. Metsch, D. Gatti, B. Frohnapfel,
//   "Solution of the extended Graetz problem for nonuniform heat flux",
//   Int. J. Heat Mass Transfer 249 (2025) 127198 — Sec. 2.4 and Appendix B.
// Cross-reference: theory/analytical_solution.tex, eq. (coefficients_equation_ord0_full)
// and eq. (seed_quadratic) (the 1x1 case of the QEP solved here).
//
// ---------------------------------------------------------------------------
// Formulation
// ---------------------------------------------------------------------------
// Projecting the O(1) operator (with axial diffusion retained) onto the bare
// radial eigenbasis R_{nm} gives, per angular index n (tex, Sec. 3):
//
//     κ Σ_{m'} U_{n,mm'} C̈_{nm'} = N²_{nm} ( Ċ_{nm} + β²_{nm} C_{nm} ) ,
//
// with κ = Pe⁻², U_{n,mm'} = ∫₀¹ r R_{nm}R_{nm'} dr the UNWEIGHTED overlap and
// N²_{nm} = ⟨R_{nm},R_{nm}⟩_ω the weighted norm. Normalising the basis
// (φ_a := R_a/N_a, ĉ_a := C_a N_a) turns this into exactly Eq. (47) of the paper,
//
//     ĉ' = Λ ĉ + κ K̃ ĉ'' ,   Λ = diag(-β²_a) ,   K̃_{ab} = U_{ab}/(N_a N_b) ,
//
// a system of coupled linear ODEs. K̃ is a Gram matrix in L²(r dr) and hence
// symmetric positive definite; Λ is diagonal negative definite (the constant
// β = 0 mode is excluded, see below).
//
// The ansatz ĉ = v e^{-λx} yields the QUADRATIC EIGENVALUE PROBLEM
//
//     ( κ λ² K̃ + λ I − B ) v = 0 ,      B := diag(β²_a) .              (QEP)
//
// Its 1x1 case is κ α λ² + λ − β² = 0 with α = U_aa/N²_a — precisely the
// closed-form seed eq. (seed_quadratic) of the tex, which fixes the sign
// convention unambiguously.
//
// ---------------------------------------------------------------------------
// Solution method — symmetric-definite linearisation (differs from `main`)
// ---------------------------------------------------------------------------
// Writing (QEP) as Q(λ)v = λ²M v + λC v + K v with M = κK̃ (SPD), C = I (SPD),
// K = −B (negative definite), we use the *symmetric* linearisation
//
//     𝓒 z = ν 𝓑 z ,   ν = −λ ,   z = [ λv ; v ] ,
//     𝓑 = [[M, 0],[0, −K]] = [[κK̃, 0],[0, B]]   (symmetric POSITIVE DEFINITE)
//     𝓒 = [[C, K],[K, 0]]  = [[I, −B],[−B, 0]]  (symmetric)
//
// (verify: row 1 gives λ²Mv+λCv+Kv = 0, row 2 gives the identity λ(−K)v+K(λv)=0).
//
// Because 𝓑 is SPD, this is a symmetric-definite pencil: Cholesky 𝓑 = LLᵀ
// reduces it to the STANDARD symmetric eigenproblem L⁻¹𝓒L⁻ᵀ y = ν y, solved by
// the cyclic Jacobi method. Every eigenvalue is therefore real *by construction*
// — matching the paper's Theorem B.1 (n positive, n negative eigenvalues) — and
// the eigenvectors are obtained directly, orthogonal to working precision.
//
// This is deliberately NOT the route taken on the `main` branch, which inverts
// κK̃ explicitly (ill-conditioned as κ→0), builds a NON-symmetric 2K×2K companion
// matrix, and recovers the spectrum with Hessenberg+QR followed by inverse
// iteration — needing a heuristic |Im λ| < 1e-5 filter and a fallback warning
// when fewer than K real eigenvalues survive. The symmetric route removes the
// explicit inverse, the complex arithmetic, and the filtering heuristics.
//
// ---------------------------------------------------------------------------
// Domain convention (differs from the paper)
// ---------------------------------------------------------------------------
// The paper solves the doubly-infinite pipe z ∈ (−∞,∞) with a jump in the wall
// condition at z = 0, and therefore needs BOTH solvents A⁺ (z<0) and A⁻ (z>0)
// plus a C¹ matching condition (their Eqs. 50, *51). This code follows the
// convention of theory/analytical_solution.tex: the SEMI-INFINITE pipe x ≥ 0
// with Dirichlet inlet data at x = 0. Boundedness as x→∞ discards the growing
// branch (K conditions) and the inlet supplies the remaining K, so only the
// DECAYING half of the spectrum (λ > 0) is retained and the modal amplitudes
// follow from a single linear solve against the inlet projection:
//
//     ĉ(0) = Σ_j a_j v_j  →  V a = ĉ(0) ,   ĉ_a(0) = C_a(0) N_a ,
//     ĉ_a(x) = Σ_j V_{aj} a_j e^{−λ_j x} ,   C_a(x) = ĉ_a(x)/N_a .
//
// The inlet coefficients C_a(0) are the ordinary BARE projections already
// computed by the solver: the QEP reuses the bare eigenbasis unchanged, which is
// the paper's central computational advantage (the eigenfunctions are
// Pe-independent and are computed once).
//
// Limits (used as regression tests): κ→0 gives λ_a→β²_a and V→I, recovering the
// base solution; and at x = 0 the series reproduces the inlet exactly.
//
// ---------------------------------------------------------------------------
// The β = 0 constant mode, and why the spectrum is shifted
// ---------------------------------------------------------------------------
// For zero-flux (Neumann) walls the axisymmetric family contains β₀₁ = 0, hence
// λ = 0 is an exact eigenvalue with eigenvector e₀. That mode carries the
// far-field constant, and it must be retained: the decaying modes feed it at
// O(κ) — from the first row of the QEP, v₀ = −κλ(K̃v)₀ — which is precisely the
// inlet back-diffusion shift of the far-field value. Dropping the mode removes
// not only the constant but its coupling to every other mode, pinning the far
// field to its Pe→∞ value.
//
// Retaining it is not possible in the plain linearisation above: with B
// singular, z = [0; e₀] satisfies BOTH 𝓑z = 0 and 𝓒z = 0, so the pencil is
// SINGULAR (a 0/0 indeterminacy), not merely singular in one matrix. No
// symmetric-definite solver can resolve that, and it cannot be repaired by a
// tolerance.
//
// The cure used here is an exact reformulation, not an approximation: shift the
// spectrum by writing λ = μ − σ with σ > 0. The QEP becomes
//
//     μ²(κK̃) v + μ(I − 2σκK̃) v + (σ²κK̃ − σI − B) v = 0 ,
//
// whose definite block is now B + σI − σ²κK̃ ≻ 0 for σ small enough (σ is chosen
// from a Gershgorin bound on K̃ and capped by the spectrum scale). The pencil is
// regular again, the same Cholesky + Jacobi path applies, and λ = μ − σ recovers
// the original eigenvalue exactly. Setting σ = 0 for a Dirichlet problem, where
// B ≻ 0 already, reproduces the unshifted formulation.
//
// Because the constant mode is now inside the block, its inlet amplitude must be
// supplied: the bare projection stores the flow-weighted mean separately (it sets
// the (0,0) coefficient to zero), so `constant_term` is passed in and used as
// that mode's inlet value. The caller must NOT add the constant term again.
//
// All heavy computation lives in this module; CDBaseSolution only sequences it.
// Instantiated for double and long double.
// ---------------------------------------------------------------------------

#include <vector>

#include "series_term_struct.h"

// Per-angular-index spectral decomposition of the QEP.
// Reconstruction:  C_a(x) = (1/N_a) Σ_j V[a][j] · amp[j] · exp(−lambda[j]·x).
template <typename Ttype>
struct QEPBlock
{
    unsigned n = 0;                       // angular index of this block
    std::vector<unsigned> flat_idx;       // a -> index into series_data (mode identity)
    std::vector<Ttype>    root;           // a -> β_{n,a}   (argument of psinm_r)
    std::vector<Ttype>    norm;           // a -> N_a       (NOT squared)
    std::vector<Ttype>    lambda;         // j -> decay rate λ_j > 0, ascending
    std::vector<std::vector<Ttype>> V;    // [a][j] eigenvector component
    std::vector<Ttype>    amp;            // j -> inlet amplitude a_j
};

// Build and solve one QEP block per angular index n.
//
// Consumes the BARE spectral data of the retained modes (n, m, root = β, norm = N²,
// coeff = C(0)) and produces the decaying spectral decomposition of each block.
// Modes with β ≈ 0 (the constant mode) are excluded.
//
// @param series_data  retained modes, grouped by n (as produced by set_series_data).
// @param max_K        number of active entries in series_data.
// @param kappa        κ = Pe⁻² > 0.
// @param constant_term inlet value of the β = 0 constant mode (the flow-weighted
//                     mean), which the bare projection stores separately. Pass 0
//                     for problems without a zero mode (e.g. Dirichlet walls).
// @param blocks       output, one entry per angular index that has ≥ 1 usable mode.
// @param panels       composite Gauss–Legendre panels for the overlap integrals.
template <typename Ttype>
void qep_build_and_solve(const std::vector<SeriesTermData<Ttype>> &series_data,
                         unsigned max_K,
                         const Ttype &kappa,
                         const Ttype &constant_term,
                         std::vector<QEPBlock<Ttype>> &blocks,
                         unsigned panels = 64u);

// Evaluate the QEP series at an unstructured cloud of points.
// Returns the raw (scaled) series value; the caller adds the constant term and
// un-scales. Points sharing an x value reuse one coefficient assembly.
template <typename Ttype>
void qep_evaluate_at_points(const std::vector<QEPBlock<Ttype>> &blocks,
                            const std::vector<Ttype> &x_points,
                            const std::vector<Ttype> &r_points,
                            const std::vector<Ttype> &phi_points,
                            std::vector<Ttype> &out);

#endif // QEP_COMPUTATIONS_H
