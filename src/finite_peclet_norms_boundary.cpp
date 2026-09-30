// Boundary-characteristic finite-Peclet norms.  Differentiating the radial
// characteristic removes the former composite quadrature from production.

#include "finite_peclet_norms_boundary.h"

#include "finite_peclet_radial.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace
{
template <typename Ttype>
bool is_zero_rate(const Ttype &value)
{
    return std::abs(value) <= static_cast<Ttype>(32) * std::numeric_limits<Ttype>::epsilon();
}

// Five-point derivatives balance truncation and cancellation by checking h and h/2.
template <typename Ttype>
Ttype characteristic_derivative(unsigned n, const Ttype &lambda, const Ttype &kappa,
                                WallCondition wall)
{
    const Ttype scale = std::max(static_cast<Ttype>(1), std::abs(lambda));
    Ttype step = std::pow(std::numeric_limits<Ttype>::epsilon(), static_cast<Ttype>(0.2L)) * scale;
    for (unsigned refinement = 0; refinement < 4; ++refinement)
    {
        if (!(lambda > static_cast<Ttype>(2) * step)) step = lambda / static_cast<Ttype>(4.1L);
        if (!(step > static_cast<Ttype>(0))) break;
        const auto derivative_at = [&](const Ttype &h) {
            return (fp_char(n, lambda - static_cast<Ttype>(2) * h, kappa, wall)
                    - static_cast<Ttype>(8) * fp_char(n, lambda - h, kappa, wall)
                    + static_cast<Ttype>(8) * fp_char(n, lambda + h, kappa, wall)
                    - fp_char(n, lambda + static_cast<Ttype>(2) * h, kappa, wall)) /
                   (static_cast<Ttype>(12) * h);
        };
        const Ttype coarse = derivative_at(step);
        const Ttype fine = derivative_at(step / static_cast<Ttype>(2));
        if (std::isfinite(static_cast<long double>(coarse)) &&
            std::isfinite(static_cast<long double>(fine)) &&
            std::abs(coarse - fine) <= static_cast<Ttype>(1e-5L) *
                std::max(static_cast<Ttype>(1), std::abs(fine)))
            return fine;
        step /= static_cast<Ttype>(2);
    }
    throw std::runtime_error("fp_generalized_norm_boundary: unstable characteristic derivative");
}
}

template <typename Ttype>
Ttype fp_generalized_norm_boundary(unsigned angular_index, const Ttype &lambda,
                                   const Ttype &kappa, WallCondition wall_condition)
{
    if (!(lambda >= static_cast<Ttype>(0)) || !(kappa >= static_cast<Ttype>(0)) ||
        !std::isfinite(static_cast<long double>(lambda)) || !std::isfinite(static_cast<long double>(kappa)))
        throw std::invalid_argument("fp_generalized_norm_boundary: invalid lambda or kappa");
    if (wall_condition == WallCondition::Neumann && angular_index == 0 && is_zero_rate(lambda))
        return static_cast<Ttype>(0.25L);
    if (!(lambda > static_cast<Ttype>(0)))
        throw std::runtime_error("fp_generalized_norm_boundary: zero rate is only valid for Neumann n=0");

    const Ttype b = std::sqrt(lambda);
    const Ttype bt = btilde_from_b(b, kappa);
    const Ttype derivative = characteristic_derivative(angular_index, lambda, kappa, wall_condition);
    const Ttype characteristic = (wall_condition == WallCondition::Dirichlet)
        ? dpsi_dr_at_1_fp(angular_index, b, bt)
        : psi_at_1_fp(angular_index, b, bt);
    const Ttype norm = (wall_condition == WallCondition::Dirichlet)
        ? characteristic * derivative / static_cast<Ttype>(angular_index + 1)
        : -characteristic * derivative / static_cast<Ttype>(angular_index + 1);
    if (!(norm > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(norm)))
        throw std::runtime_error("fp_generalized_norm_boundary: non-positive or non-finite norm");
    return norm;
}

template <typename Ttype>
void fp_compute_norms_boundary(std::vector<SeriesTermData<Ttype>> &series_data,
                               unsigned max_K, const Ttype &kappa, WallCondition wall_condition)
{
    if (max_K > series_data.size()) throw std::invalid_argument("fp_compute_norms_boundary: max_K exceeds series data");
#pragma omp parallel for schedule(dynamic)
    for (long long index = 0; index < static_cast<long long>(max_K); ++index)
        series_data[static_cast<std::size_t>(index)].norm_fp = fp_generalized_norm_boundary(
            series_data[static_cast<std::size_t>(index)].n,
            series_data[static_cast<std::size_t>(index)].rate_fp, kappa, wall_condition);
}

template double fp_generalized_norm_boundary(unsigned, const double &, const double &, WallCondition);
template long double fp_generalized_norm_boundary(unsigned, const long double &, const long double &, WallCondition);
template void fp_compute_norms_boundary(std::vector<SeriesTermData<double>> &, unsigned, const double &, WallCondition);
template void fp_compute_norms_boundary(std::vector<SeriesTermData<long double>> &, unsigned, const long double &, WallCondition);
