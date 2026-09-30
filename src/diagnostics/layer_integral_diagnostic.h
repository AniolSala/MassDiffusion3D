#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>

#include "math_functions.h"

// Constants (matching coefficients_computation_gaussian.cpp)
#define DIAG_SQRT_HALF_PI     1.25331413731550024736L
#define DIAG_SQRT_2_PI        2.50662827463100024161L
#define DIAG_INV_SQRT_PI      0.56418958354775628695L
#define DIAG_INV_SQRT_2_PI    0.39894228040143267794L

// All components of a single-layer coefficient integral I(n, z_i, root).
// The full coefficient contribution from layer i is:
//   total = gaussian_part + hypergeometric_part
// where:
//   gaussian_part       – Gauss-quadrature over the half-disc cross-section
//   hypergeometric_part – corrective closed-form via 1F1 (non-zero only when n==0 and z_i<0)
//   closed_form_ref     – analytic reference for n==0 (independent of root/m; useful to
//                         validate that gaussian_part + hypergeometric_part converges to it)
struct LayerIntegralComponents
{
    double gaussian_part;
    double hypergeometric_part;
    double closed_form_ref; // finite only for n==0; NaN otherwise
    double total;
    int    n_nodes;
};

namespace layer_integral_diag {

// ── internal helpers ──────────────────────────────────────────────────────────

// Closed-form for the angular-only integral (n==0) over the spherical cap at z_i.
// This does NOT depend on (root, m) — it is the pure geometric result for
// the constant angular mode s_0(phi) = phi / sqrt(2*pi).
template <typename T>
T closed_form_n0(const T &z_i)
{
    const T z_i_abs = std::fabs(z_i);
    const T y_lim   = std::sqrt(static_cast<T>(1.0L) - z_i * z_i);
    const T phi     = std::acos(z_i_abs);
    const T z_pos   = (phi - z_i_abs * y_lim * (static_cast<T>(5.0L) - static_cast<T>(2.0L) * z_i * z_i)
                                / static_cast<T>(3.0L))
                      * static_cast<T>(DIAG_INV_SQRT_2_PI);
    return (z_i > static_cast<T>(0.0L))
               ? z_pos
               : static_cast<T>(DIAG_SQRT_HALF_PI) - z_pos;
}

// Hypergeometric correction for (n==0, z_i < 0).
template <typename T>
T hypergeometric_correction(const T &root, const T &z_i)
{
    const T nu             = (static_cast<T>(2.0L) - root) * static_cast<T>(0.25L);
    const T z              = root * z_i * z_i;
    const T common_factor  = static_cast<T>(-0.5L) * root * z_i
                             * std::exp(static_cast<T>(-0.5L) * z);
    T F1_1, F1_2;
    hypergeometric1F1(nu,                          1u, z, F1_1);
    hypergeometric1F1(nu + static_cast<T>(1.0L),   2u, z, F1_2);
    const T dpsi_0m = common_factor
                      * (static_cast<T>(2.0L) * F1_1
                         + (root - static_cast<T>(2.0L)) * F1_2);
    return -static_cast<T>(DIAG_SQRT_2_PI) * z_i / (root * root) * dpsi_0m;
}

// Gaussian-quadrature part of the layer integral.
template <typename T>
T gaussian_part(
    const unsigned    &n,
    const T           &root,
    const T           &z_i,
    const std::vector<T> &weights,
    const std::vector<T> &nodes)
{
    if (weights.size() != nodes.size())
        throw std::invalid_argument("weights and nodes must have the same length");

    const unsigned nq            = static_cast<unsigned>(nodes.size());
    const T        one_minus_zsq = static_cast<T>(1.0L) - z_i * z_i;
    const T        y_lim         = std::sqrt(one_minus_zsq);
    const T        layer_factor  = static_cast<T>(2.0L) * one_minus_zsq * one_minus_zsq;

    T result = static_cast<T>(0.0L);

    if (n == 0)
    {
        const T inv_sqrt_2pi = static_cast<T>(DIAG_INV_SQRT_2_PI);
        for (unsigned i = 0; i < nq; ++i)
        {
            const T y     = nodes[i] * y_lim;
            const T r     = std::sqrt(y * y + z_i * z_i);
            const T phi   = std::atan2(y, z_i);
            const T w_jac = weights[i] * layer_factor;
            T psi;
            psinm_r(n, root, r, psi);
            result += psi * (phi * inv_sqrt_2pi) * w_jac;
        }
    }
    else
    {
        const T n_t         = static_cast<T>(n);
        const T sin_prefact = static_cast<T>(DIAG_INV_SQRT_PI) / n_t;
        for (unsigned i = 0; i < nq; ++i)
        {
            const T y     = nodes[i] * y_lim;
            const T r     = std::sqrt(y * y + z_i * z_i);
            const T phi   = std::atan2(y, z_i);
            const T w_jac = weights[i] * layer_factor;
            T psi;
            psinm_r(n, root, r, psi);
            result += psi * (std::sin(n_t * phi) * sin_prefact) * w_jac;
        }
    }
    return result;
}

} // namespace layer_integral_diag

// ── public entry point ────────────────────────────────────────────────────────

// Compute all components of the layer-i coefficient integral for mode (n, m)
// at interface position z_i with radial eigenvalue root, using the supplied
// Gaussian quadrature weights/nodes.
//
// Parameters
//   n, m         – angular/radial mode indices
//   z_i          – interface z-coordinate in (-1, 1)
//   root         – beta_{n,m}, the radial eigenvalue
//   gauss_weights, gauss_nodes – 1-D Gauss quadrature data (same ordering as
//                  the files in src/CDBaseSolution/data/)
//
// Returns LayerIntegralComponents with every contribution split out.
template <typename T = double>
LayerIntegralComponents compute_layer_integral_components(
    const unsigned       n,
    const unsigned       m,
    const double         z_i_d,
    const double         root_d,
    const std::vector<double> &gauss_weights_d,
    const std::vector<double> &gauss_nodes_d)
{
    // Work in template precision internally, accept double from Python
    const T z_i  = static_cast<T>(z_i_d);
    const T root = static_cast<T>(root_d);
    std::vector<T> weights(gauss_weights_d.begin(), gauss_weights_d.end());
    std::vector<T> gNodes(gauss_nodes_d.begin(),   gauss_nodes_d.end());

    LayerIntegralComponents out;
    out.n_nodes = static_cast<int>(gNodes.size());

    out.gaussian_part = static_cast<double>(
        layer_integral_diag::gaussian_part(n, root, z_i, weights, gNodes));

    out.hypergeometric_part = 0.0;
    if (n == 0 && z_i < static_cast<T>(0.0L))
        out.hypergeometric_part = static_cast<double>(
            layer_integral_diag::hypergeometric_correction(root, z_i));

    out.closed_form_ref = (n == 0)
        ? static_cast<double>(layer_integral_diag::closed_form_n0(z_i))
        : std::numeric_limits<double>::quiet_NaN();

    out.total = out.gaussian_part + out.hypergeometric_part;
    return out;
}
