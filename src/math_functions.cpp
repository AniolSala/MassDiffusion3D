#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
// #include <boost/multiprecision/cpp_bin_float.hpp>
// #include <boost/multiprecision/cpp_dec_float.hpp>

#include <boost/math/special_functions.hpp>

#include "series_term_struct.h"

#define HALF_PI 1.5707963267948966
#define INV_SQRT_PI 0.5641895835477563
#define INV_SQRT_2_PI 0.3989422804014327

template <typename Ttype>
Ttype pow_uint(Ttype base, unsigned exponent)
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

// Define the high-precision type (50 decimal digits of precision)
// using namespace boost::multiprecision;
// using high_precision = cpp_bin_float_50;

double weight_ST(const double& r)
{
    return r * (1. - r * r);
}

void hypergeometric1F1(double mu, unsigned nu, double z, long double &r_val)
{
    r_val = boost::math::hypergeometric_1F1(mu, nu, z);
}

void hypergeometric1F1(double mu, unsigned nu, double z, double &r_val)
{
    r_val = boost::math::hypergeometric_1F1(mu, nu, z);
}

void hypergeometric1F1(long double mu, unsigned nu, long double z, long double &r_val)
{
    r_val = boost::math::hypergeometric_1F1(mu, nu, z);
}

template <typename Ttype>
void psinm_r(const unsigned &k, const Ttype &b, const Ttype &r, Ttype &r_val)
{
    const Ttype z = b * r * r;
    // Guard: 1F1 uses e^z internally; when z exceeds log(max_representable)
    // the intermediate result overflows.  The eigenfunction is zero at every
    // Dirichlet root by construction, and for r < 1 the overall mode
    // contribution exp(-β²x)·ψ is negligible at any physical x > 0.
    if (z >= std::log(std::numeric_limits<Ttype>::max())) { r_val = static_cast<Ttype>(0.0); return; }

    Ttype mu = static_cast<Ttype>(0.25L) * (static_cast<Ttype>(2 * k) - b + static_cast<Ttype>(2.0L));
    unsigned nu = k + 1;

    hypergeometric1F1(mu, nu, z, r_val);

    Ttype pre_factor_u = pow_uint(r, k) * std::exp(static_cast<Ttype>(-0.5L) * z);
    r_val *= pre_factor_u;
}

template <typename Ttype>
Ttype psinm_r(const unsigned& k, const Ttype& b, const Ttype& r)
{
    const Ttype z = b * r * r;
    if (z >= std::log(std::numeric_limits<Ttype>::max())) return static_cast<Ttype>(0.0);

    Ttype mu = static_cast<Ttype>(0.25L) * (static_cast<Ttype>(2 * k) - b + static_cast<Ttype>(2.0L));
    unsigned nu = k + 1;

    Ttype r_val;
    hypergeometric1F1(mu, nu, z, r_val);

    Ttype pre_factor_u = pow_uint(r, k) * std::exp(static_cast<Ttype>(-0.5L) * z);
    r_val *= pre_factor_u;
    return r_val;
}

template <typename Ttype>
Ttype dpsinm_r_1(const unsigned& k, const Ttype& b, const Ttype& r)
{
    if (k == 0 && b <= static_cast<Ttype>(1e-10L))
        return static_cast<Ttype>(0.0L);

    Ttype mu = static_cast<Ttype>(0.25L) * (static_cast<Ttype>(2 * k) - b + static_cast<Ttype>(2.0L));
    unsigned nu = k + 1;
    Ttype z = b * r * r;

    // r_val = boost::math::hypergeometric_1F1(mu, nu, z);
    Ttype hyp_1, hyp_2;
    hypergeometric1F1(mu, nu, z, hyp_1);
    hypergeometric1F1(mu + 1, nu + 1, z, hyp_2);

    if (r >= static_cast<Ttype>(1e-10L))
    {
        hyp_1 *= static_cast<Ttype>(2.0L) * static_cast<Ttype>(nu) * (static_cast<Ttype>(k) / r - b * r);
        hyp_2 *= static_cast<Ttype>(4.0L) * b * mu * r;
        Ttype pre_factor_global = static_cast<Ttype>(0.5L) * std::pow(r, static_cast<Ttype>(k)) * std::exp(static_cast<Ttype>(-0.5L) * z) / static_cast<Ttype>(nu);

        Ttype sol = hyp_1 + hyp_2;
        sol *= pre_factor_global;
        return sol;
    }
    else
    {
        if (k == 1)
        {
            return static_cast<Ttype>(1.0L);
        }
        else
        {
            return static_cast<Ttype>(0.0L);
        }
    }
}

template <typename Ttype>
Ttype dpsinm_r_2(const unsigned& k, const Ttype& b, const Ttype& r)
{
    if (k == 0 && b <= static_cast<Ttype>(1e-10L))
        return static_cast<Ttype>(0.0L);

    Ttype mu = static_cast<Ttype>(0.25L) * (static_cast<Ttype>(2 * k) - b + static_cast<Ttype>(2.0L));
    unsigned nu = k + 1;
    Ttype z = b * r * r;

    // r_val = boost::math::hypergeometric_1F1(mu, nu, z);
    Ttype hyp_1, hyp_2, hyp_3;
    hypergeometric1F1(mu, nu, z, hyp_1);
    hypergeometric1F1(mu + 1, nu + 1, z, hyp_2);
    hypergeometric1F1(mu + 2, nu + 2, z, hyp_3);
        Ttype hyp_23_factor = b * (b - static_cast<Ttype>(2.0L) * static_cast<Ttype>(nu)) * r * r;
        hyp_1 *= static_cast<Ttype>(4.0L) * (static_cast<Ttype>(2.0L) + static_cast<Ttype>(3.0L) * static_cast<Ttype>(k) + static_cast<Ttype>(k * k))
            * (static_cast<Ttype>(k * k) + z * (z - static_cast<Ttype>(1.0L)) - static_cast<Ttype>(k) * (static_cast<Ttype>(1.0L) + static_cast<Ttype>(2.0L) * z));
        hyp_2 *= hyp_23_factor * static_cast<Ttype>(-2.0L) * static_cast<Ttype>(nu + 1)
            * (static_cast<Ttype>(1.0L) + static_cast<Ttype>(2.0L) * static_cast<Ttype>(k) - static_cast<Ttype>(2.0L) * z);
        hyp_3 *= hyp_23_factor * b * (b - static_cast<Ttype>(2.0L) * static_cast<Ttype>(nu + 2)) * r * r;

    // Compute r^(k-2) safely. With `k` unsigned, `k - 2` wraps to a huge
    // positive integer for k in {0, 1}, and `std::pow(r, huge)` collapses
    // to zero, silently zeroing out the formula. Branch on signed k to
    // produce the mathematically correct 1/r or 1/r^2 factor.
    Ttype r_pow_km2;
    if (k >= 2)
    {
        r_pow_km2 = std::pow(r, static_cast<Ttype>(static_cast<int>(k) - 2));
    }
    else if (k == 1)
    {
        r_pow_km2 = static_cast<Ttype>(1.0L) / r;
    }
    else // k == 0
    {
        r_pow_km2 = static_cast<Ttype>(1.0L) / (r * r);
    }

    Ttype pre_factor_global = static_cast<Ttype>(0.25L) * r_pow_km2
                * std::exp(static_cast<Ttype>(-0.5L) * z)
                / (static_cast<Ttype>(nu) * static_cast<Ttype>(nu + 1));

    Ttype sol = pre_factor_global * (hyp_1 + hyp_2 + hyp_3);
    return sol;
}

template <typename Ttype>
Ttype dpsinm_r(const unsigned& n, const unsigned& k, const Ttype& b, const Ttype& r)
{
    if (n == 1)
    {
        return dpsinm_r_1(k, b, r);
    }
    else
    {
        return dpsinm_r_2(k, b, r);
    }
}

// Phi and X functions
template <typename Ttype>
void gnm_x(const Ttype &b, const Ttype &x, Ttype &r_val)
{
    r_val = std::exp(-b * b * x);
}

template <typename Ttype>
Ttype gnm_x(const Ttype& b, const Ttype& x)
{
    return std::exp(-b * b * x);
}

template <typename Ttype>
void sn_phi(const unsigned &k, const Ttype &phi, Ttype &r_val)
{
    if (k == 0)
    {
        // r_val = std::sqrt(.5 * M_1_PI);
        r_val = INV_SQRT_2_PI;
    }
    else
    {
        // r_val = std::sqrt(M_1_PI) * std::cos(k * phi);
        r_val = INV_SQRT_PI * std::cos(k * phi);
    }
}

template <typename Ttype>
Ttype sn_phi(const unsigned& k, const Ttype& phi)
{
    if (k == 0)
    {
        // r_val = std::sqrt(.5 * M_1_PI);
        return INV_SQRT_2_PI;
    }
    else
    {
        // r_val = std::sqrt(M_1_PI) * std::cos(k * phi);
        return INV_SQRT_PI * std::cos(k * phi);
    }
}

template <typename Ttype>
Ttype dsn_phi(const unsigned& n, const unsigned& k, const Ttype& phi)
{
    if (k == 0)
        return 0.0;
    if (k % 2 == 0)
    {
        return pow(-k * k, n / 2) * sn_phi(k, phi);
    }
    else
    {
        return pow(-1, (n + 1) / 2) * pow(k, n) * INV_SQRT_PI * std::sin(k * phi);
    }
}

double compute_flux(const double& z_val)
{
    const double z_clamped = std::clamp(z_val, -1.0, 1.0);
    const double z_abs = std::abs(z_clamped);
    const double z_sq = z_clamped * z_clamped;
    const double y_lim = std::sqrt(1.0 - z_sq);
    double dimless_flux = (z_abs * y_lim * (2.0 * z_sq - 5.0) + 3.0 * std::acos(z_abs)) / 6.0;

    if (z_clamped < 0.0)
    {
        dimless_flux = HALF_PI - dimless_flux;
    }
    return dimless_flux;
}

double compute_flux_layer(const double& z1, const double& z2)
{
    return std::abs(compute_flux(z2) - compute_flux(z1));
}


// Instantiations
template void psinm_r<double>(const unsigned &, const double &, const double &, double &);
template void psinm_r<long double>(const unsigned &, const long double &, const long double &, long double &);
template double psinm_r<double>(const unsigned&, const double&, const double&);
template long double psinm_r<long double>(const unsigned&, const long double&, const long double&);
template double dpsinm_r<double>(const unsigned&, const unsigned&, const double&, const double&);
template long double dpsinm_r<long double>(const unsigned&, const unsigned&, const long double&, const long double&);

template void gnm_x<double>(const double &, const double &, double &);
template void gnm_x<long double>(const long double &, const long double &, long double &);
template double gnm_x<double>(const double&, const double&);
template long double gnm_x<long double>(const long double&, const long double&);

template void sn_phi<double>(const unsigned &, const double &, double &);
template void sn_phi<long double>(const unsigned &, const long double &, long double &);
template double sn_phi<double>(const unsigned&, const double&);
template long double sn_phi<long double>(const unsigned&, const long double&);
template double dsn_phi<double>(const unsigned&, const unsigned&, const double&);
template long double dsn_phi<long double>(const unsigned&, const unsigned&, const long double&);
