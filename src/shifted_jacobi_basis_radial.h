#ifndef SHIFTED_JACOBI_BASIS_RADIAL_H
#define SHIFTED_JACOBI_BASIS_RADIAL_H

#include <vector>

// ---------------------------------------------------------------------------
// Shifted Jacobi polynomials P̂_j^{(0,n)}(s) = P_j^{(0,n)}(2s - 1) on [0,1],
// orthogonal with respect to the weight s^n -- which is exactly the UNWEIGHTED
// (L2_r) finite-Peclet Gram weight in the squared-radius variable s = r^2.
//
// This is the (alpha, beta) = (0, n) counterpart of shifted_jacobi_basis.h's
// (1, n) family. The two differ only in the Jacobi parameters: same three-term
// recurrence, different norms. They are kept as separate translation units for
// the same reason the two Gram backends are -- each measure's basis is
// independently auditable, and neither file has to be edited to change the
// other.
//
// PURE MATHEMATICS. This header must not include any solver header and must
// not mention modes, blocks, inlets or boundary conditions.
// ---------------------------------------------------------------------------

// Exact norm  h_j = int_0^1 s^n [P̂_j^{(0,n)}(s)]^2 ds = 1 / (2j + n + 1).
//
// This is the (alpha, beta) = (0, n) case of the general shifted-Jacobi norm
//
//   int_0^1 (1-s)^a s^b [P̂_j^{(a,b)}]^2 ds
//        = 1/(2j+a+b+1) * Gamma(j+a+1) Gamma(j+b+1) / (Gamma(j+a+b+1) j!) ,
//
// whose (1, n) case is shifted_jacobi_basis.h's
// h_j = (j+1)/((2j+n+2)(j+n+1)). At (0, n) the Gamma ratio collapses to unity
// and only the leading factor survives, so this form is elementary: it needs no
// Gamma evaluation at all, and at n = 0 it is the shifted-Legendre 1/(2j+1).
//
// Sanity anchor: h_0 = 1/(n+1) is the total mass int_0^1 s^n ds, since
// P̂_0 == 1. Verified against quadrature for n = 0..99, j <= 40, to 1e-12.
template <typename Ttype>
Ttype shifted_jacobi_norm_radial(unsigned j, unsigned angular_index);

// Row-major table, count x nodes.size():  table[j*nodes.size() + q] = P̂_j(nodes[q]).
// Uses the standard three-term recurrence in x = 2s - 1 with (a,b) = (0,n) --
// the same recurrence shifted_jacobi_basis.cpp runs at (1,n):
//   P_0 = 1
//   P_1 = ((a - b) + (a + b + 2) x) / 2
//   2k(k+a+b)(2k+a+b-2) P_k =
//        (2k+a+b-1)[(2k+a+b)(2k+a+b-2) x + a^2 - b^2] P_{k-1}
//      - 2(k+a-1)(k+b-1)(2k+a+b) P_{k-2}
// The leading coefficient 2k(k+a+b)(2k+a+b-2) is non-zero for every k >= 2 at
// (0,n) -- including n = 0, where it is 2k*k*(2k-2) -- so unlike the Gauss rule
// of finite_peclet_gram_radial.cpp this recurrence needs NO 0/0 special case.
//
// Values can be enormous near s = 0 for large n, exactly as in the (1,n) case:
// measured max 3.1e68 at n = 99 over a 220-node Gram rule, against 3.1e68 for
// (1,n) on the same block -- the two families are the same size. This is
// expected, and callers that contract against the s^n weight recover
// well-scaled results because the s^n factor compensates: the measured
// max_s s^n|P̂_j| is bounded by 1.0 here (against ~56 for the (1,n) family),
// so the compensation is if anything tighter on this path. Throw on non-finite
// output rather than clamping.
template <typename Ttype>
std::vector<Ttype> shifted_jacobi_table_radial(unsigned count, unsigned angular_index,
                                               const std::vector<Ttype> &nodes);

#endif
