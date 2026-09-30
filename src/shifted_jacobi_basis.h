#ifndef SHIFTED_JACOBI_BASIS_H
#define SHIFTED_JACOBI_BASIS_H

#include <vector>

// ---------------------------------------------------------------------------
// Shifted Jacobi polynomials P̂_j^{(1,n)}(s) = P_j^{(1,n)}(2s - 1) on [0,1],
// orthogonal with respect to the weight (1-s)*s^n -- which is exactly the
// finite-Peclet Gram weight in the squared-radius variable s = r^2.
//
// PURE MATHEMATICS. This header must not include any solver header and must
// not mention modes, blocks, inlets or boundary conditions.
// ---------------------------------------------------------------------------

// Exact norm  h_j = int_0^1 (1-s) s^n [P̂_j^{(1,n)}(s)]^2 ds
//                 = (j+1) / ((2j + n + 2)(j + n + 1)).
//
// Verified against quadrature for n = 0..99, j <= 300, to 1e-15.
// NOTE: there is NO 2^{-(n+2)} factor and NO underflow at large n:
// h_0 = 1/((n+1)(n+2)) ~ 1.07e-4 at n = 95. Any implementation producing
// 2^{-(n+2)}-sized norms has mixed shifted and unshifted normalisations.
template <typename Ttype>
Ttype shifted_jacobi_norm(unsigned j, unsigned angular_index);

// Row-major table, count x nodes.size():  table[j*nodes.size() + q] = P̂_j(nodes[q]).
// Uses the standard three-term recurrence in x = 2s - 1 with (a,b) = (1,n):
//   P_0 = 1
//   P_1 = ((a - b) + (a + b + 2) x) / 2
//   2k(k+a+b)(2k+a+b-2) P_k =
//        (2k+a+b-1)[(2k+a+b)(2k+a+b-2) x + a^2 - b^2] P_{k-1}
//      - 2(k+a-1)(k+b-1)(2k+a+b) P_{k-2}
//
// Values can be enormous near s = 0 for large n (P̂_j(0) = (-1)^j binom(j+n, j);
// measured max 2.0e43 at n = 99, j <= 220). This is expected: callers that
// contract against the (1-s)s^n weight recover well-scaled results because the
// s^n factor compensates exactly. Throw on non-finite output rather than
// clamping.
template <typename Ttype>
std::vector<Ttype> shifted_jacobi_table(unsigned count, unsigned angular_index,
                                        const std::vector<Ttype> &nodes);

#endif
