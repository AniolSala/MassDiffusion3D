#ifndef NUSSELT_POSTPROCESSING_H
#define NUSSELT_POSTPROCESSING_H

// ---------------------------------------------------------------------------
// nusselt_postprocessing — free functions behind the accelerated evaluations of
// CDBaseSolution::get_nusselt_number (plan nusselt_implementation_plan_update.md,
// Sec. 6b). Nothing here touches a solution object: the functions act on rates,
// Bessel zeros and scalars only. Instantiated for double and long double.
//
// For k >> Pe/pi the axisymmetric Dirichlet modes are Bessel-like and
// c_k R_k'(1) -> -2 sqrt(2 pi) (unit internal inlet), while the rates grow only
// linearly. The wall-gradient sum over the modes beyond the retained block is
// therefore -2 sqrt(2 pi) sum_{m>K} exp(-Lam_m x) up to a remainder that
// vanishes with the retained block; that sum needs rates only.
// ---------------------------------------------------------------------------

#include <vector>

enum class NusseltEvaluation
{
    Plain = 0,          // retained modes only (eq. nusselt_modal)
    SpectralSplit = 1,  // + tail over the exact finite-Peclet rates beyond the block
    SlugSplit = 2,      // + tail over the slug-flow comparison rates beyond the block
    AsymptoticSplit = 3 // G = G_0 + G_inf of eqs. G_split and G_inf_split_asymptotic
};

// Velocity of the slug-flow comparison problem used by NusseltEvaluation::SlugSplit:
// the limit of the mode-averaged profile <1 - r^2>_m = (2/3)(1 + 1/q_m^2). The 1/q_m^2
// term is dropped; it is smaller than the second-order rate error, which decays as 1/m.
template <typename Ttype>
inline Ttype nusselt_slug_mean_velocity() { return static_cast<Ttype>(2) / static_cast<Ttype>(3); }

// Successive axisymmetric Dirichlet rates above lambda_last: roots of
// Lam -> psi_at_1_fp(0, sqrt(Lam), sqrt(Lam)(1 + kappa Lam)), found by an upward
// scan with sign-change detection and fp_solve_bracketed. The scan step is a
// quarter of the smallest rate spacing seen, starting from
// lambda_last - lambda_previous (the spacing grows monotonically towards pi Pe).
// Throws std::runtime_error when a rate leaves the range of the Kummer kernel.
template <typename Ttype>
std::vector<Ttype> fp_dirichlet_rates_beyond(const Ttype &kappa, const Ttype &lambda_previous,
                                             const Ttype &lambda_last, unsigned count,
                                             const Ttype &rel_tol, unsigned max_iter);

// First `count` zeros of J_0 starting at the 1-based index `first_index`.
template <typename Ttype>
std::vector<Ttype> bessel_j0_zeros(unsigned first_index, unsigned count);

// Slug-flow rates  Lam^s = [sqrt(u^2 + 4 kappa q^2) - u] / (2 kappa), written in the
// form 2 q^2 / (u + sqrt(u^2 + 4 kappa q^2)) that is also valid at kappa = 0.
template <typename Ttype>
std::vector<Ttype> slug_flow_rates(const std::vector<Ttype> &bessel_zeros, const Ttype &kappa,
                                   const Ttype &mean_velocity);

// sum_m exp(-(Lam_m - rate_shift) x) over ascending rates, summed from the
// smallest term up. x = +inf gives the number of rates equal to rate_shift.
template <typename Ttype>
Ttype spectral_function_sum(const std::vector<Ttype> &rates_ascending, const Ttype &x,
                            const Ttype &rate_shift = static_cast<Ttype>(0));

// Rates of the Bessel modes J_0(q_m r) (eq. rate_inf of analytical_solution.tex)
// with the mode-averaged velocity <omega>_m = (2/3)(1 + 1/q_m^2):
//   Lam^inf_m = 2 q_m^2 / (<omega>_m + sqrt(<omega>_m^2 + 4 kappa q_m^2)).
template <typename Ttype>
std::vector<Ttype> bessel_mode_rates(const std::vector<Ttype> &bessel_zeros, const Ttype &kappa);

// Lam^a_m = pi (m - 1/4) / sqrt(kappa) - 1/(3 kappa) for m >= 1, the large-m
// form of Lam^inf_m (eq. rate_asymptotic_progression).
template <typename Ttype>
Ttype asymptotic_progression_rate(unsigned m, const Ttype &kappa);

// Number m0 of rates Lam^a_m (m >= 1) that are not positive,
// floor(1/4 + 1/(3 pi sqrt(kappa))); zero for Pe <= 9 pi/4.
template <typename Ttype>
unsigned asymptotic_progression_non_positive_terms(const Ttype &kappa);

// G_0 of eq. G_split over the retained modes, times exp(rate_shift x):
//   sum_k e^{-(Lam_k - s) x} [1 - e^{-(Lam^inf_k - Lam_k) x}],
// with Lam^inf_k the rate of eq. rate_inf paired with the retained rate Lam_k.
// x must be finite and positive.
template <typename Ttype>
Ttype spectral_function_g0(const std::vector<Ttype> &rates, const std::vector<Ttype> &paired_bessel_mode_rates,
                           const Ttype &x, const Ttype &rate_shift);

// G_inf of eq. G_inf_split_asymptotic, times exp(rate_shift x), with its sum
// truncated at M = bessel_mode_rates.size() (Lam^inf_m for m = 1..M):
//   sum_{m<=M} e^{-(Lam^inf_m - s) x} [1 - e^{-(Lam^a_m - Lam^inf_m) x}]
//     + e^{(pi/(4 sqrt(kappa)) + 1/(3 kappa)) x} e^{s x} / (e^{pi x/sqrt(kappa)} - 1).
// For Pe > 9 pi/4 the first m0 = floor(1/4 + Pe/(3 pi)) rates Lam^a_m are not
// positive, and their e^{-Lam^a_m x} grow with x in both the sum and the closed
// form. They are dropped from both, i.e. the closed form is summed from m0 + 1,
//   sum_{m>m0} e^{-Lam^a_m x} = e^{-Lam^a_{m0+1} x} / (1 - e^{-pi x/sqrt(kappa)}),
// which is the same expression without the cancelling growth. Needs M >= m0 and
// x finite and positive.
template <typename Ttype>
Ttype spectral_function_g_inf(const std::vector<Ttype> &bessel_mode_rates, const Ttype &kappa,
                              const Ttype &x, const Ttype &rate_shift);

// Closed form of sum_{m>=1} exp(-q_m xi), q_m the zeros of J_0 (xi > 0):
//   1/(pi xi) - 1/4 + (1/pi) int_0^inf [I1(k)/I0(k) - 1 + 1/(2k)] sin(k xi) dk.
// Composite Gauss-Legendre on [0, L] plus the integration-by-parts tail built
// from the large-k expansion of I1/I0.
template <typename Ttype>
Ttype bessel_spectral_function_closed_form(const Ttype &xi);

#endif // NUSSELT_POSTPROCESSING_H
