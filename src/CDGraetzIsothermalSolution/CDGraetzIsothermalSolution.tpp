#if !defined(CD_GRAETZ_ISOTHERMAL_SOLUTION_TPP)
#define CD_GRAETZ_ISOTHERMAL_SOLUTION_TPP

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "CDGraetzIsothermalSolution.h"
#include "../pseudo_products.h"
#include "../io_data.h"
#include "../math_functions.h"
#include "../finite_peclet_coefficients_gaussian.h"
#include "../finite_peclet_coefficients_radial.h"

template <typename Ttype>
/**
 * Construct a CDGraetzIsothermalSolution for the classical isothermal Graetz problem.
 *
 * The problem: Poiseuille flow in a unit-radius pipe with a uniform inlet profile
 * θ(x=0) = 1 and an isothermal wall θ(r=1) = 0 (Dirichlet BC).
 *
 * Solution:  θ(x,r) = Σ_m C_m exp(-β_m² x) G_m(r)
 *
 * where G_m(r) = psinm_r(0, β_m, r) with Dirichlet eigenvalues β_m satisfying
 * psinm_r(0, β_m, 1) = 0, and the coefficient C_m is given by the closed-form
 * formula of Belhocine (2017), Eq. 56.
 *
 * Physical temperature: T(x,r) = T_wall + θ(x,r) * (T0 - T_wall)
 *
 * @param T0          Inlet (uniform) temperature.
 * @param T_wall      Isothermal wall temperature.
 * @param verbose_level  Verbosity (0 = silent, 1 = summary, 2 = timing).
 */
CDGraetzIsothermalSolution<Ttype>::CDGraetzIsothermalSolution(
    Ttype T0, Ttype T_wall, unsigned short verbose_level)
    // Pass a minimal 2-layer dummy to CDBaseSolution so that set_zi / set_ui
    // do not throw. The normalisation is overridden immediately below.
    : CDBaseSolution<Ttype>(
          {static_cast<Ttype>(0.0)},
          {T0, T_wall},
          0),  // silence base-class output; we call print_info() ourselves
      m_T0(T0), m_T_wall(T_wall)
{
    if (T0 == T_wall)
        throw std::invalid_argument("T0 and T_wall must differ.");

    // Override the base-class normalisation for the Graetz far-field.
    //
    // The base class formula:  T_physical = m_max_ui - θ * (m_max_ui - m_min_ui)
    //
    // We want:  T_physical = T_wall + θ * (T0 - T_wall)
    //
    // Setting m_max_ui = T_wall and m_min_ui = T0 gives exactly this:
    //   T_physical = T_wall - θ * (T_wall - T0) = T_wall + θ * (T0 - T_wall)  ✓
    this->m_max_ui = T_wall;
    this->m_min_ui = T0;

    // Far field (x → ∞): θ → 0 (Dirichlet BC drains the scalar to zero).
    this->m_constant_term = static_cast<Ttype>(0.0);

    this->m_verbose_level = verbose_level;
    if (verbose_level >= 1)
        print_info();
}

template <typename Ttype>
/**
 * Load Dirichlet eigenvalues (roots of psinm_r(0, β, 1) = 0) for the Graetz problem.
 * File format: one row (n=0), columns are the successive eigenvalues β_0, β_1, …
 *
 * The file path is fixed at compile time and identical for every
 * CDGraetzIsothermalSolution<Ttype> instance regardless of T0/T_wall, so it is
 * loaded from disk once per process (function-local static, safe under
 * concurrent first use per the C++11 static-initialization guarantee) and
 * reused by every subsequent instance.
 */
void CDGraetzIsothermalSolution<Ttype>::set_roots(
    std::vector<std::vector<Ttype>> &roots) const
{
    static const std::vector<std::vector<Ttype>> cached_roots = [] {
        const std::filesystem::path roots_path =
            std::filesystem::path(__FILE__).parent_path() / "data" / "graetz_roots.txt";
        std::vector<std::vector<Ttype>> loaded;
        readValuesFromFile(roots_path.string(), loaded);
        return loaded;
    }();
    roots = cached_roots;
}

template <typename Ttype>
/**
 * Load norms N_m² = ∫₀¹ r(1-r²) [G_m(r)]² dr for the Graetz eigenfunctions.
 * File format: one row (n=0), columns are N_0², N_1², …
 * Cached the same way as set_roots() above.
 */
void CDGraetzIsothermalSolution<Ttype>::set_norms(
    std::vector<std::vector<Ttype>> &norms) const
{
    static const std::vector<std::vector<Ttype>> cached_norms = [] {
        const std::filesystem::path norms_path =
            std::filesystem::path(__FILE__).parent_path() / "data" / "graetz_norms.txt";
        std::vector<std::vector<Ttype>> loaded;
        readValuesFromFile(norms_path.string(), loaded);
        return loaded;
    }();
    norms = cached_norms;
}

template <typename Ttype>
/**
 * Load precomputed pseudo-products matrix P_{m1,m2} = ∫₀¹ r(1-r²) G_m1(r) G_m2(r) dr
 * for the Graetz Dirichlet eigenfunctions.  Generated offline by
 * write_graetz_pseudo_products_matrix() in main.cpp.
 * Cached the same way as set_roots() above -- this is the largest of the
 * Graetz data tables (~1.3 MB of text), so caching it matters most.
 */
void CDGraetzIsothermalSolution<Ttype>::set_pseudo_products_matrix(
    std::vector<std::vector<std::vector<Ttype>>> &pseudo_products)
{
    static const std::vector<std::vector<std::vector<Ttype>>> cached_pseudo_products = [] {
        const std::filesystem::path path =
            std::filesystem::path(__FILE__).parent_path() / "data"
            / "graetz_pseudo_products_matrix.txt";
        std::vector<std::vector<std::vector<Ttype>>> loaded;
        read_pseudo_products_matrix(path, loaded);
        return loaded;
    }();
    pseudo_products = cached_pseudo_products;
}

template <typename Ttype>
/**
 * Load precomputed series coefficients c_{0m}^S = √(2π)·C_m from the data file.
 *
 * The values were generated offline by write_graetz_roots_and_norms() in main.cpp
 * using the closed-form Belhocine (2017) Eq. 56 formula at long-double precision:
 *
 *   C_m = [(1/2 - 1/β_m) exp(-β_m/2) ₁F₁(3/2 - β_m/4, 2; β_m)] / N_m²
 *   c_{0m}^S = √(2π) · C_m
 *
 * Loading from file avoids recomputing ₁F₁ at runtime and ensures all three
 * related tables (roots, norms, coefficients) are consistent and produced at
 * the same precision.
 *
 * The file itself is fixed at compile time (independent of T0/T_wall), so the
 * parsed row is cached in a function-local static and reused by every
 * instance in the process; only the assignment into this instance's
 * m_series_data below is per-object.
 */
void CDGraetzIsothermalSolution<Ttype>::compute_coefficients()
{
    static const std::vector<Ttype> cached_coeffs_row = [] {
        const std::filesystem::path coeffs_path =
            std::filesystem::path(__FILE__).parent_path() / "data" / "graetz_coefficients.txt";
        std::vector<std::vector<Ttype>> coeffs_table;
        readValuesFromFile(coeffs_path.string(), coeffs_table);
        if (coeffs_table.empty() || coeffs_table[0].empty())
            throw std::runtime_error("graetz_coefficients.txt is empty or missing.");
        return coeffs_table[0];
    }();

    for (unsigned k = 0; k < this->m_max_K; k++)
    {
        auto &term = this->m_series_data[k];

        if (term.n != 0)
        {
            term.coeff = static_cast<Ttype>(0.0);
            continue;
        }

        if (term.m < cached_coeffs_row.size())
            term.coeff = cached_coeffs_row[term.m];
        else
            term.coeff = static_cast<Ttype>(0.0);
    }

    this->mCoefficientsAreComputed = true;
}

template <typename Ttype>
/**
 * Finite-Péclet coefficients for the isothermal Graetz problem. Delegates to the
 * Gaussian Gram solve. The legacy table argument is retained solely for the
 * virtual interface and is never read; no global finite-Peclet table is built.
 */
void CDGraetzIsothermalSolution<Ttype>::compute_coefficients_fp(
    const FPRadialTable<Ttype> &unused_radial_table)
{
    static_cast<void>(unused_radial_table);
    // L2_r projection (opt-in; see projection_space.h and
    // finite_peclet_coefficients_radial.h). Both RHS backends are available
    // here; the one unsupported combination (ultraspherical Gram) was already
    // rejected by setup_fp_solution, so this branch needs no GramMethod.
    if (this->m_projection_space == ProjectionSpace::Radial)
    {
        FPGramRadialOptions<Ttype> radial_gram_options;
        radial_gram_options.oversampling_factor = this->m_gram_oversampling_factor;
        radial_gram_options.oversampling_margin = this->m_gram_oversampling_margin;
        radial_gram_options.enable_order_check = this->m_gram_order_check;
        radial_gram_options.minimum_factor = this->m_gram_minimum_factor;
        fp_graetz_coefficients_radial<Ttype>(this->m_max_K, this->m_series_data,
                                             this->wall_condition(), radial_gram_options,
                                             this->m_fp_cap_quad_margin,
                                             this->m_rhs_method, this->m_rhs_representer_margin,
                                             this->m_verbose_level >= 3,
                                             &this->m_inlet_projection_square_norm);
        return;
    }
    FPGramGaussJacobiOptions<Ttype> gram_options;
    gram_options.oversampling_factor = this->m_gram_oversampling_factor;
    gram_options.oversampling_margin = this->m_gram_oversampling_margin;
    gram_options.enable_order_check = this->m_gram_order_check;
    gram_options.minimum_factor = this->m_gram_minimum_factor;
    fp_graetz_coefficients_gaussian<Ttype>(this->m_max_K, this->m_series_data,
                                           this->m_gram_method, this->wall_condition(), gram_options,
                                           this->m_fp_cap_quad_margin, this->m_rhs_method,
                                           this->m_rhs_representer_margin, this->m_verbose_level >= 3,
                                           &this->m_inlet_projection_square_norm);
}

template <typename Ttype>
/**
 * Draw a simple ASCII sketch of the Graetz problem (uniform inlet, isothermal wall).
 */
void CDGraetzIsothermalSolution<Ttype>::draw_step_solution_terminal(
    std::ostringstream &os)
{
    const int radius = 10;
    const double aspect = 2.0;

    for (int y = radius; y >= -radius; --y)
    {
        os << "   ";
        bool inside = false;
        for (int x = static_cast<int>(-radius * aspect);
             x <= static_cast<int>(radius * aspect); ++x)
        {
            double dist = std::sqrt(std::pow(x / aspect, 2) + y * y);
            if (std::abs(dist - radius) < 0.5)
            {
                os << "#";
                inside = (dist - radius <= 0.0);
            }
            else
            {
                os << (inside ? "/" : " ");
            }
        }
        os << "\n";
    }
}

template <typename Ttype>
/**
 * Print a summary of the Graetz solution parameters.
 */
void CDGraetzIsothermalSolution<Ttype>::print_info()
{
    std::ostringstream msg;
    msg << "\nCDGraetzIsothermalSolution (classical isothermal Graetz problem)\n";
    msg << "  Inlet temperature  T0     = " << m_T0    << "\n";
    msg << "  Wall  temperature  T_wall = " << m_T_wall << "\n";

    std::ostringstream sketch;
    draw_step_solution_terminal(sketch);
    msg << sketch.str();

    std::cout << msg.str() << std::endl;
}

#endif
