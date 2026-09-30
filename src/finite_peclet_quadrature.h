#ifndef FINITE_PECLET_QUADRATURE_H
#define FINITE_PECLET_QUADRATURE_H

// ---------------------------------------------------------------------------
// finite_peclet_quadrature — self-contained composite Gauss-Legendre rule on
// (0,1), used by the finite-Peclet norm and coefficient quadratures.
//
// Why self-contained (rather than the cached gaussian_weights_eigenvalues_* node
// files): those files bake a SPECIFIC weight function (the bare (1-r^2) measure)
// into their nodes. The finite-Peclet norm uses a DIFFERENT, rate-dependent
// weight (1 - r^2 + 2*kappa*Lam), and the overlap uses the plain measure r dr, so
// this module must integrate against a clean, weight-agnostic Gauss-Legendre rule
// and apply each integrand's weight explicitly.
//
// The rule is `panels` uniform sub-intervals of (0,1), each carrying the 16-point
// Gauss-Legendre rule. 16*panels nodes total. Header-only (inline) so both the
// double and long double translation units share one definition.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fp_quad
{
// 16-point Gauss-Legendre abscissae/weights on [-1, 1] (long double literals).
inline const long double GL16_X[16] = {
    -0.9894009349916499325961541734504L, -0.9445750230732325760779884155346L,
    -0.8656312023878317438804678977123L, -0.7554044083550030338951011948474L,
    -0.6178762444026437484466717640413L, -0.4580167776572273863424194429835L,
    -0.2816035507792589132304605014605L, -0.0950125098376374401853193354250L,
     0.0950125098376374401853193354250L,  0.2816035507792589132304605014605L,
     0.4580167776572273863424194429835L,  0.6178762444026437484466717640413L,
     0.7554044083550030338951011948474L,  0.8656312023878317438804678977123L,
     0.9445750230732325760779884155346L,  0.9894009349916499325961541734504L};
inline const long double GL16_W[16] = {
    0.0271524594117540948517805724560L, 0.0622535239386478928628438369944L,
    0.0951585116824927848099251076022L, 0.1246289712555338720524762821920L,
    0.1495959888165767320815017305474L, 0.1691565193950025381893120790304L,
    0.1826034150449235888667636679692L, 0.1894506104550684962853967232083L,
    0.1894506104550684962853967232083L, 0.1826034150449235888667636679692L,
    0.1691565193950025381893120790304L, 0.1495959888165767320815017305474L,
    0.1246289712555338720524762821920L, 0.0951585116824927848099251076022L,
    0.0622535239386478928628438369944L, 0.0271524594117540948517805724560L};

// Emit 16*panels Gauss-Legendre nodes r and weights wt on (0,1).
// @param panels number of uniform sub-intervals (>= 1). Larger => higher accuracy.
// @param r      output nodes (resized to 16*panels).
// @param wt     output weights (resized to 16*panels); sum(wt) == 1.
template <typename Ttype>
inline void gl_nodes(unsigned panels, std::vector<Ttype> &r, std::vector<Ttype> &wt)
{
    if (panels == 0)
        throw std::invalid_argument("fp_quad::gl_nodes: panels must be positive");
    r.assign(static_cast<size_t>(panels) * 16, static_cast<Ttype>(0));
    wt.assign(static_cast<size_t>(panels) * 16, static_cast<Ttype>(0));
    const Ttype hp = static_cast<Ttype>(1.0L) / static_cast<Ttype>(panels);
    const Ttype hh = static_cast<Ttype>(0.5L) * hp;
    for (unsigned p = 0; p < panels; ++p)
    {
        const Ttype c = static_cast<Ttype>(p) * hp + hh;
        for (unsigned g = 0; g < 16; ++g)
        {
            const size_t idx = static_cast<size_t>(p) * 16 + g;
            r[idx]  = c + hh * static_cast<Ttype>(GL16_X[g]);
            wt[idx] = hh * static_cast<Ttype>(GL16_W[g]);
        }
    }
}
} // namespace fp_quad

#endif // FINITE_PECLET_QUADRATURE_H
