#ifndef RHS_METHOD_H
#define RHS_METHOD_H

// Backend used to assemble the finite-Peclet load vector b^n.
//
// DirectQuadrature : one Gauss-Jacobi cap sweep per (mode, interface), each
//                    mode's order sized from its own bare root
//                    (finite_peclet_rhs.h). O(sum_m order_m) confluent-
//                    hypergeometric evaluations.
// Representer      : build the L2_omega representer of each interface's cap
//                    functional once per (angular block, interface), then take
//                    one weighted dot product per mode against the samples the
//                    Gram assembly already holds. ZERO additional confluent-
//                    hypergeometric evaluations.
//
// As with GramMethod there is NO fallback between these: a failing selected
// method must be visible to its caller, so each can be assessed alone.
enum class RhsMethod
{
    DirectQuadrature = 0,
    Representer      = 1
};

#endif // RHS_METHOD_H
