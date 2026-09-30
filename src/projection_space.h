#ifndef PROJECTION_SPACE_H
#define PROJECTION_SPACE_H

// Inner product in which the finite-Peclet inlet coefficients are obtained by
// Galerkin projection.  As with GramMethod/RhsMethod there is NO fallback
// between these: a failing selected projection must be visible to its caller.
//
//   Weighted : <f,g> = int_0^1 f g omega(r) r dr,  omega = 1 - r^2.  DEFAULT,
//              and the norm in which the bare modes are orthogonal.
//   Radial   : <f,g> = int_0^1 f g r dr.  The modified modes are a Riesz basis
//              of this space, so the normalised Gram matrix has a condition
//              number bounded independently of the truncation level.
enum class ProjectionSpace
{
    Weighted = 0,
    Radial   = 1
};

#endif // PROJECTION_SPACE_H
