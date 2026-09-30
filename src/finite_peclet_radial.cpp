// ---------------------------------------------------------------------------
// finite_peclet_radial.cpp — implementation of the modified radial eigenfunction
// and its wall (characteristic) values for the exact finite-Peclet solution.
//
// See finite_peclet_radial.h and theory/analytical_solution.tex Sec. 5.2 for the
// mathematical derivation. Adapted (and re-derived) from the reference kernels on
// the `main` branch; verified here against the bare-limit reduction
//     psinm_r_fp(n, b, b, r) == psinm_r(n, b, r)
// in tests/test_fp_radial.cpp.
// ---------------------------------------------------------------------------

#include <cmath>
#include <limits>

#include <boost/math/special_functions.hpp>

#include "finite_peclet_radial.h"
#include "math_functions.h"   // hypergeometric1F1 overloads (shared with the bare mode)

namespace
{
// Integer power by squaring; local copy of the helper in math_functions.cpp
// (which is file-local there). Used for the r^n prefactor.
template <typename Ttype>
Ttype fp_pow_uint(Ttype base, unsigned exponent)
{
    Ttype result = static_cast<Ttype>(1.0L);
    while (exponent > 0)
    {
        if (exponent & 1U)
            result *= base;
        base *= base;
        exponent >>= 1U;
    }
    return result;
}
} // namespace

template <typename Ttype>
Ttype btilde_from_b(const Ttype &b, const Ttype &kappa)
{
    return b * (static_cast<Ttype>(1.0L) + kappa * b * b);
}

template <typename Ttype>
Ttype fp_radial_factor(const unsigned &n, const Ttype &b, const Ttype &bt,
                       const Ttype &squared_radius)
{
    const Ttype z = b * squared_radius;
    // Same overflow guard as psinm_r: 1F1 uses e^z internally, and for r < 1 the
    // overall mode contribution exp(-Lam x) R is negligible at any physical x > 0.
    if (z >= std::log(std::numeric_limits<Ttype>::max()))
        return static_cast<Ttype>(0.0L);

    // First Kummer parameter carries the SHIFTED bt: a = (2n + 2 - bt)/4.
    const Ttype a = static_cast<Ttype>(0.25L) *
                    (static_cast<Ttype>(2 * n) + static_cast<Ttype>(2.0L) - bt);
    const unsigned nu = n + 1;

    Ttype val;
    hypergeometric1F1(a, nu, z, val);
    // Prefactor and argument keep the UNSHIFTED b.
    val *= std::exp(static_cast<Ttype>(-0.5L) * z);
    return val;
}

template <typename Ttype>
Ttype psinm_r_fp(const unsigned &n, const Ttype &b, const Ttype &bt, const Ttype &r)
{
    return fp_pow_uint(r, n) * fp_radial_factor(n, b, bt, r * r);
}

template <typename Ttype>
void psinm_r_fp(const unsigned &n, const Ttype &b, const Ttype &bt, const Ttype &r, Ttype &out)
{
    out = psinm_r_fp(n, b, bt, r);
}

template <typename Ttype>
Ttype psi_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt)
{
    return psinm_r_fp(n, b, bt, static_cast<Ttype>(1.0L));
}

template <typename Ttype>
Ttype dpsi_dr_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt)
{
    if (b >= std::log(std::numeric_limits<Ttype>::max()))
        return static_cast<Ttype>(0.0L);

    // (n+1) R'(1) = e^{-b/2} [ (n - b)(n+1) M1 + 2 a b M2 ]  (eq. char_neumann_fp),
    // with M1 = 1F1(a, nu; b), M2 = 1F1(a+1, nu+1; b), a = (2n+2-bt)/4, nu = n+1.
    const Ttype a = static_cast<Ttype>(0.25L) *
                    (static_cast<Ttype>(2 * n) + static_cast<Ttype>(2.0L) - bt);
    const unsigned nu = n + 1;

    Ttype M1, M2;
    hypergeometric1F1(a, nu, b, M1);
    hypergeometric1F1(a + static_cast<Ttype>(1.0L), nu + 1, b, M2);

    const Ttype term1 = (static_cast<Ttype>(n) - b) * static_cast<Ttype>(nu) * M1;
    const Ttype term2 = static_cast<Ttype>(2.0L) * a * b * M2;
    return std::exp(static_cast<Ttype>(-0.5L) * b) * (term1 + term2);
}

template <typename Ttype>
Ttype radial_derivative_at_1_fp(const unsigned &n, const Ttype &b, const Ttype &bt)
{
    return dpsi_dr_at_1_fp(n, b, bt) / static_cast<Ttype>(n + 1U);
}

// --- Explicit instantiations (double + long double) --------------------------
template double fp_radial_factor<double>(const unsigned &, const double &, const double &, const double &);
template long double fp_radial_factor<long double>(const unsigned &, const long double &, const long double &, const long double &);
template double      psinm_r_fp<double>(const unsigned &, const double &, const double &, const double &);
template long double psinm_r_fp<long double>(const unsigned &, const long double &, const long double &, const long double &);
template void psinm_r_fp<double>(const unsigned &, const double &, const double &, const double &, double &);
template void psinm_r_fp<long double>(const unsigned &, const long double &, const long double &, const long double &, long double &);
template double      psi_at_1_fp<double>(const unsigned &, const double &, const double &);
template long double psi_at_1_fp<long double>(const unsigned &, const long double &, const long double &);
template double      dpsi_dr_at_1_fp<double>(const unsigned &, const double &, const double &);
template long double dpsi_dr_at_1_fp<long double>(const unsigned &, const long double &, const long double &);
template double      radial_derivative_at_1_fp<double>(const unsigned &, const double &, const double &);
template long double radial_derivative_at_1_fp<long double>(const unsigned &, const long double &, const long double &);
template double      btilde_from_b<double>(const double &, const double &);
template long double btilde_from_b<long double>(const long double &, const long double &);
