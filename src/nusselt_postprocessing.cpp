// ---------------------------------------------------------------------------
// nusselt_postprocessing.cpp — see nusselt_postprocessing.h.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/math/special_functions/bessel.hpp>

#include "nusselt_postprocessing.h"
#include "finite_peclet_roots.h"
#include "finite_peclet_quadrature.h"

template <typename Ttype>
std::vector<Ttype> fp_dirichlet_rates_beyond(const Ttype &kappa, const Ttype &lambda_previous,
                                             const Ttype &lambda_last, unsigned count,
                                             const Ttype &rel_tol, unsigned max_iter)
{
    if (!(kappa > static_cast<Ttype>(0)))
        throw std::invalid_argument("fp_dirichlet_rates_beyond: kappa must be positive (finite-Peclet path).");
    if (!(lambda_previous >= static_cast<Ttype>(0)) || !(lambda_last > lambda_previous))
        throw std::invalid_argument("fp_dirichlet_rates_beyond: need 0 <= lambda_previous < lambda_last.");

    const Ttype quarter = static_cast<Ttype>(0.25L);
    const Ttype b_limit = std::log(std::numeric_limits<Ttype>::max());
    const auto characteristic = [&](const Ttype &lam) {
        if (!(std::sqrt(lam) < b_limit))
            throw std::runtime_error("fp_dirichlet_rates_beyond: rate beyond the range of the Kummer kernel "
                                     "(sqrt(Lam) >= log(max)); use a larger x or long double.");
        const Ttype value = fp_char<Ttype>(0u, lam, kappa, WallCondition::Dirichlet);
        if (!std::isfinite(static_cast<long double>(value)))
            throw std::runtime_error("fp_dirichlet_rates_beyond: non-finite characteristic function.");
        return value;
    };

    std::vector<Ttype> rates;
    rates.reserve(count);
    Ttype min_spacing = lambda_last - lambda_previous;
    Ttype previous_root = lambda_last;
    Ttype lo = lambda_last + quarter * min_spacing;
    Ttype f_lo = characteristic(lo);
    while (rates.size() < count)
    {
        const Ttype hi = lo + quarter * min_spacing;
        const Ttype f_hi = characteristic(hi);
        if (f_hi == static_cast<Ttype>(0) || (f_hi > static_cast<Ttype>(0)) != (f_lo > static_cast<Ttype>(0)))
        {
            const Ttype root = (f_hi == static_cast<Ttype>(0))
                ? hi
                : fp_solve_bracketed<Ttype>(0u, kappa, WallCondition::Dirichlet, lo, hi, rel_tol, max_iter);
            min_spacing = std::min(min_spacing, root - previous_root);
            rates.push_back(root);
            previous_root = root;
            lo = root + quarter * min_spacing;
            f_lo = characteristic(lo);
        }
        else
        {
            lo = hi;
            f_lo = f_hi;
        }
    }
    return rates;
}

template <typename Ttype>
std::vector<Ttype> bessel_j0_zeros(unsigned first_index, unsigned count)
{
    if (first_index == 0u)
        throw std::invalid_argument("bessel_j0_zeros: indices are 1-based.");
    std::vector<Ttype> zeros(count);
    boost::math::cyl_bessel_j_zero(static_cast<Ttype>(0), static_cast<int>(first_index),
                                   count, zeros.begin());
    return zeros;
}

template <typename Ttype>
std::vector<Ttype> slug_flow_rates(const std::vector<Ttype> &bessel_zeros, const Ttype &kappa,
                                   const Ttype &mean_velocity)
{
    if (!(kappa >= static_cast<Ttype>(0)) || !(mean_velocity >= static_cast<Ttype>(0))
        || !(kappa > static_cast<Ttype>(0) || mean_velocity > static_cast<Ttype>(0)))
        throw std::invalid_argument("slug_flow_rates: need kappa >= 0, mean_velocity >= 0, not both zero.");
    std::vector<Ttype> rates(bessel_zeros.size());
    for (std::size_t m = 0; m < bessel_zeros.size(); ++m)
    {
        const Ttype q2 = bessel_zeros[m] * bessel_zeros[m];
        rates[m] = static_cast<Ttype>(2) * q2
            / (mean_velocity + std::sqrt(mean_velocity * mean_velocity + static_cast<Ttype>(4) * kappa * q2));
    }
    return rates;
}

template <typename Ttype>
Ttype spectral_function_sum(const std::vector<Ttype> &rates_ascending, const Ttype &x, const Ttype &rate_shift)
{
    Ttype sum = 0;
    for (auto it = rates_ascending.rbegin(); it != rates_ascending.rend(); ++it)
    {
        const Ttype excess = *it - rate_shift;
        if (excess == static_cast<Ttype>(0))
            sum += static_cast<Ttype>(1);
        else
            sum += std::exp(-excess * x);
    }
    return sum;
}

template <typename Ttype>
std::vector<Ttype> bessel_mode_rates(const std::vector<Ttype> &bessel_zeros, const Ttype &kappa)
{
    if (!(kappa > static_cast<Ttype>(0)))
        throw std::invalid_argument("bessel_mode_rates: kappa must be positive.");
    std::vector<Ttype> rates(bessel_zeros.size());
    for (std::size_t m = 0; m < bessel_zeros.size(); ++m)
    {
        const Ttype q2 = bessel_zeros[m] * bessel_zeros[m];
        const Ttype mean_velocity = static_cast<Ttype>(2) / static_cast<Ttype>(3)
                                  * (static_cast<Ttype>(1) + static_cast<Ttype>(1) / q2);
        rates[m] = static_cast<Ttype>(2) * q2
            / (mean_velocity + std::sqrt(mean_velocity * mean_velocity + static_cast<Ttype>(4) * kappa * q2));
    }
    return rates;
}

template <typename Ttype>
Ttype asymptotic_progression_rate(unsigned m, const Ttype &kappa)
{
    const Ttype pi = static_cast<Ttype>(3.14159265358979323846264338327950288L);
    return pi * (static_cast<Ttype>(m) - static_cast<Ttype>(0.25L)) / std::sqrt(kappa)
         - static_cast<Ttype>(1) / (static_cast<Ttype>(3) * kappa);
}

template <typename Ttype>
unsigned asymptotic_progression_non_positive_terms(const Ttype &kappa)
{
    if (!(kappa > static_cast<Ttype>(0)))
        throw std::invalid_argument("asymptotic_progression_non_positive_terms: kappa must be positive.");
    const Ttype pi = static_cast<Ttype>(3.14159265358979323846264338327950288L);
    // Lam^a_m <= 0  <=>  m <= 1/4 + 1/(3 pi sqrt(kappa)).
    return static_cast<unsigned>(std::floor(static_cast<Ttype>(0.25L)
                                            + static_cast<Ttype>(1) / (static_cast<Ttype>(3) * pi * std::sqrt(kappa))));
}

template <typename Ttype>
Ttype spectral_function_g0(const std::vector<Ttype> &rates, const std::vector<Ttype> &paired_bessel_mode_rates,
                           const Ttype &x, const Ttype &rate_shift)
{
    if (rates.size() != paired_bessel_mode_rates.size())
        throw std::invalid_argument("spectral_function_g0: one Bessel-mode rate per retained rate is required.");
    if (!(x > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(x)))
        throw std::invalid_argument("spectral_function_g0: x must be finite and positive.");
    Ttype sum = 0;
    for (std::size_t k = rates.size(); k-- > 0;)
        sum += std::exp(-(rates[k] - rate_shift) * x) * -std::expm1(-(paired_bessel_mode_rates[k] - rates[k]) * x);
    return sum;
}

template <typename Ttype>
Ttype spectral_function_g_inf(const std::vector<Ttype> &bessel_mode_rates, const Ttype &kappa,
                              const Ttype &x, const Ttype &rate_shift)
{
    if (!(kappa > static_cast<Ttype>(0)))
        throw std::invalid_argument("spectral_function_g_inf: kappa must be positive.");
    if (!(x > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(x)))
        throw std::invalid_argument("spectral_function_g_inf: x must be finite and positive.");
    const Ttype pi = static_cast<Ttype>(3.14159265358979323846264338327950288L);
    const Ttype sqrt_kappa = std::sqrt(kappa);
    const unsigned m0 = asymptotic_progression_non_positive_terms(kappa);
    if (bessel_mode_rates.size() < m0)
        throw std::invalid_argument("spectral_function_g_inf: the sum needs at least the terms with non-positive Lam^a_m.");

    Ttype sum = 0;
    for (std::size_t i = bessel_mode_rates.size(); i-- > 0;)
    {
        const unsigned m = static_cast<unsigned>(i) + 1u;
        const Ttype factor = std::exp(-(bessel_mode_rates[i] - rate_shift) * x);
        if (m <= m0)
            sum += factor;
        else
            sum += factor * -std::expm1(-(asymptotic_progression_rate<Ttype>(m, kappa) - bessel_mode_rates[i]) * x);
    }
    const Ttype closed_form = std::exp(-(asymptotic_progression_rate<Ttype>(m0 + 1u, kappa) - rate_shift) * x)
                            / -std::expm1(-pi * x / sqrt_kappa);
    return sum + closed_form;
}

namespace
{
// Coefficients a_j of I1(k)/I0(k) ~ sum_j a_j k^{-j} (exponentially small terms
// dropped), from the quotient of the two Hankel expansions
//   I_nu(k) ~ e^k/sqrt(2 pi k) sum_j (-1)^j prod_{i=1..j}(4nu^2 - (2i-1)^2) / (j! 8^j k^j).
template <typename Ttype>
std::array<Ttype, 12> bessel_ratio_expansion()
{
    std::array<Ttype, 12> s0{}, s1{}, ratio{};
    for (int nu = 0; nu <= 1; ++nu)
    {
        auto &s = nu == 0 ? s0 : s1;
        s[0] = 1;
        for (unsigned j = 1; j < s.size(); ++j)
        {
            const Ttype odd = static_cast<Ttype>(2 * j - 1);
            s[j] = -s[j - 1] * (static_cast<Ttype>(4 * nu * nu) - odd * odd) / (static_cast<Ttype>(8 * j));
        }
    }
    for (unsigned j = 0; j < ratio.size(); ++j)
    {
        Ttype value = s1[j];
        for (unsigned i = 1; i <= j; ++i)
            value -= s0[i] * ratio[j - i];
        ratio[j] = value;
    }
    return ratio;
}

// f(k) = I1(k)/I0(k) - 1 + 1/(2k) and, for the tail, its derivatives.
template <typename Ttype>
struct BesselRatioRemainder
{
    std::array<Ttype, 12> a = bessel_ratio_expansion<Ttype>();
    const Ttype switch_point = static_cast<Ttype>(40);

    // derivative-th derivative of the asymptotic expansion (a_0 = 1, a_1 = -1/2 cancel).
    Ttype asymptotic(const Ttype &k, unsigned derivative) const
    {
        Ttype sum = 0;
        for (unsigned j = a.size() - 1; j >= 2; --j)
        {
            Ttype factor = 1;
            for (unsigned d = 0; d < derivative; ++d)
                factor *= -static_cast<Ttype>(j + d);
            sum += a[j] * factor * std::pow(k, -static_cast<Ttype>(j + derivative));
        }
        return sum;
    }

    Ttype operator()(const Ttype &k) const
    {
        if (k >= switch_point)
            return asymptotic(k, 0u);
        const Ttype ratio = boost::math::cyl_bessel_i(static_cast<Ttype>(1), k)
                          / boost::math::cyl_bessel_i(static_cast<Ttype>(0), k);
        return ratio - static_cast<Ttype>(1) + static_cast<Ttype>(0.5L) / k;
    }
};
} // namespace

template <typename Ttype>
Ttype bessel_spectral_function_closed_form(const Ttype &xi)
{
    if (!(xi > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(xi)))
        throw std::invalid_argument("bessel_spectral_function_closed_form: xi must be finite and positive.");

    const Ttype pi = static_cast<Ttype>(3.14159265358979323846264338327950288L);
    const BesselRatioRemainder<Ttype> f;
    // At least 200 oscillations and k >= 2000 so that the tail series in 1/(xi L)
    // and the expansion of f are both far inside their accuracy.
    const Ttype upper = std::max(static_cast<Ttype>(200) * 2 * pi / xi, static_cast<Ttype>(2000));
    const Ttype max_width = static_cast<Ttype>(0.25L) * pi / xi;

    Ttype integral = 0;
    Ttype left = 0;
    while (left < upper)
    {
        const Ttype width = std::min({max_width, std::max(static_cast<Ttype>(0.25L), static_cast<Ttype>(0.25L) * left),
                                      upper - left});
        const Ttype half = static_cast<Ttype>(0.5L) * width;
        const Ttype mid = left + half;
        Ttype panel = 0;
        for (unsigned i = 0; i < 16; ++i)
        {
            const Ttype k = mid + half * static_cast<Ttype>(fp_quad::GL16_X[i]);
            panel += static_cast<Ttype>(fp_quad::GL16_W[i]) * f(k) * std::sin(k * xi);
        }
        integral += half * panel;
        left += width;
    }

    // Repeated integration by parts at k = L:
    //   int_L^inf f sin(xi k) dk = f cos/xi - f' sin/xi^2 - f'' cos/xi^3 + f''' sin/xi^4 + ...
    const Ttype c = std::cos(xi * upper), s = std::sin(xi * upper);
    Ttype tail = 0, xi_power = xi;
    for (unsigned d = 0; d < 8; ++d)
    {
        const Ttype derivative = f.asymptotic(upper, d);
        const Ttype trig = (d % 2 == 0) ? c : s;
        const Ttype sign = (d % 4 == 0 || d % 4 == 3) ? static_cast<Ttype>(1) : static_cast<Ttype>(-1);
        tail += sign * derivative * trig / xi_power;
        xi_power *= xi;
    }
    integral += tail;

    return static_cast<Ttype>(1) / (pi * xi) - static_cast<Ttype>(0.25L) + integral / pi;
}

// --- Explicit instantiations (double + long double) --------------------------
template std::vector<double> fp_dirichlet_rates_beyond<double>(const double &, const double &, const double &,
                                                               unsigned, const double &, unsigned);
template std::vector<long double> fp_dirichlet_rates_beyond<long double>(const long double &, const long double &,
                                                                         const long double &, unsigned,
                                                                         const long double &, unsigned);
template std::vector<double> bessel_j0_zeros<double>(unsigned, unsigned);
template std::vector<long double> bessel_j0_zeros<long double>(unsigned, unsigned);
template std::vector<double> slug_flow_rates<double>(const std::vector<double> &, const double &, const double &);
template std::vector<long double> slug_flow_rates<long double>(const std::vector<long double> &,
                                                               const long double &, const long double &);
template double spectral_function_sum<double>(const std::vector<double> &, const double &, const double &);
template long double spectral_function_sum<long double>(const std::vector<long double> &, const long double &,
                                                        const long double &);
template std::vector<double> bessel_mode_rates<double>(const std::vector<double> &, const double &);
template std::vector<long double> bessel_mode_rates<long double>(const std::vector<long double> &, const long double &);
template double asymptotic_progression_rate<double>(unsigned, const double &);
template long double asymptotic_progression_rate<long double>(unsigned, const long double &);
template unsigned asymptotic_progression_non_positive_terms<double>(const double &);
template unsigned asymptotic_progression_non_positive_terms<long double>(const long double &);
template double spectral_function_g0<double>(const std::vector<double> &, const std::vector<double> &,
                                             const double &, const double &);
template long double spectral_function_g0<long double>(const std::vector<long double> &,
                                                       const std::vector<long double> &,
                                                       const long double &, const long double &);
template double spectral_function_g_inf<double>(const std::vector<double> &, const double &, const double &,
                                                const double &);
template long double spectral_function_g_inf<long double>(const std::vector<long double> &, const long double &,
                                                          const long double &, const long double &);
template double bessel_spectral_function_closed_form<double>(const double &);
template long double bessel_spectral_function_closed_form<long double>(const long double &);
