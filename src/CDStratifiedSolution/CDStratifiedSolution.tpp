#if !defined(CD_STRATIFIED_SOLUTION_TPP)
#define CD_STRATIFIED_SOLUTION_TPP

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

#include "CDStratifiedSolution.h"

#include "../coefficients_computation_gaussian.h"
#include "../pseudo_products.h"
#include "../finite_peclet_coefficients_gauss_jacobi.h"
#include "../finite_peclet_coefficients_radial.h"

#ifndef CDS_ROOTS_FILE
#define CDS_ROOTS_FILE "roots_200x200.txt"
#endif

#ifndef CDS_NORMS_FILE
#define CDS_NORMS_FILE "norms_200x200_cpp.txt"
#endif

#ifndef CDS_PSEUDO_PRODUCTS_FILE
#define CDS_PSEUDO_PRODUCTS_FILE "pseudo_products_matrix_100x100x100.txt"
#endif

// set_roots/set_norms/set_pseudo_products_matrix each read from a file path
// fixed at compile time (CDS_ROOTS_FILE / CDS_NORMS_FILE /
// CDS_PSEUDO_PRODUCTS_FILE), identical for every CDStratifiedSolution<Ttype>
// instance regardless of zi/ui. Each caches its parsed table in a function-
// local static, loaded from disk once per process and reused by every
// subsequent instance -- safe under concurrent first use because C++11
// guarantees static local initialization happens exactly once even under
// a race between threads. The pseudo-products table is by far the largest
// (~7.7 MB of text), so caching it matters most.

template <typename Ttype>
void CDStratifiedSolution<Ttype>::set_roots(std::vector<std::vector<Ttype>> &roots) const
{
    static const std::vector<std::vector<Ttype>> cached_roots = [] {
        const std::filesystem::path path =
            std::filesystem::path(__FILE__).parent_path() / "data" / CDS_ROOTS_FILE;
        std::vector<std::vector<Ttype>> loaded;
        readValuesFromFile(path.string(), loaded);
        return loaded;
    }();
    roots = cached_roots;
}

template <typename Ttype>
void CDStratifiedSolution<Ttype>::set_norms(std::vector<std::vector<Ttype>> &norms) const
{
    static const std::vector<std::vector<Ttype>> cached_norms = [] {
        const std::filesystem::path path =
            std::filesystem::path(__FILE__).parent_path() / "data" / CDS_NORMS_FILE;
        std::vector<std::vector<Ttype>> loaded;
        readValuesFromFile(path.string(), loaded);
        return loaded;
    }();
    norms = cached_norms;
}

template <typename Ttype>
void CDStratifiedSolution<Ttype>::set_pseudo_products_matrix(
    std::vector<std::vector<std::vector<Ttype>>> &pseudo_products)
{
    static const std::vector<std::vector<std::vector<Ttype>>> cached_pseudo_products = [] {
        const std::filesystem::path path =
            std::filesystem::path(__FILE__).parent_path() / "data" / CDS_PSEUDO_PRODUCTS_FILE;
        std::vector<std::vector<std::vector<Ttype>>> loaded;
        read_pseudo_products_matrix(path, loaded);
        return loaded;
    }();
    pseudo_products = cached_pseudo_products;
}

template <typename Ttype>
/**
 * Construct a CDStratifiedSolution instance with layer data.
 * Delegates all field initialisation to CDBaseSolution, then prints info.
 * @param zi Interface positions.
 * @param ui Layer density values.
 * @param verbose_level Verbosity level.
 */
CDStratifiedSolution<Ttype>::CDStratifiedSolution(
    const std::vector<Ttype> &zi,
    const std::vector<Ttype> &ui,
    unsigned short verbose_level)
    : CDBaseSolution<Ttype>(zi, ui, verbose_level)
{
    if (this->m_verbose_level >= 1)
    {
        print_info();
    }
}


template <typename Ttype>
/**
 * Compute coefficients for the current series data.
 */
void CDStratifiedSolution<Ttype>::compute_coefficients()
{
    compute_coefficients_matrix(this->m_zi, this->m_ui, this->m_gauss_weights, this->m_gauss_points, this->m_max_K, this->m_series_data);
    this->mCoefficientsAreComputed = true;

    // for(unsigned k = 0; k < this->m_max_K; k++)
    // {
    //     unsigned nn = this->m_series_data[k].n;
    //     unsigned mm = this->m_series_data[k].m;
    //     Ttype coeff = this->m_series_data[k].coeff;
    //     if (nn == 0)
    //     {
    //         std::cout << "n = " << nn << ", m = " << mm <<", coeff = " << coeff << std::endl;
    //     }
    // }
}


template <typename Ttype>
/**
 * Finite-Péclet coefficients for the stratified inlet. Delegates to the
 * exact angular reduction followed by fixed Gauss-Jacobi cap quadrature and a
 * radial Gaussian Gram solve. The legacy table argument remains unused solely
 * to preserve the virtual interface.
 */
void CDStratifiedSolution<Ttype>::compute_coefficients_fp(
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
        fp_stratified_coefficients_radial<Ttype>(this->m_zi, this->m_ui, this->m_max_K,
                                                 this->m_series_data, this->m_fp_cap_quad_margin,
                                                 this->wall_condition(), radial_gram_options,
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
    fp_stratified_coefficients_gauss_jacobi<Ttype>(
        this->m_zi,
        this->m_ui,
        this->m_max_K,
        this->m_series_data,
        this->m_fp_cap_quad_margin,
        this->m_gram_method,
        this->wall_condition(),
        gram_options,
        this->m_rhs_method,
        this->m_rhs_representer_margin,
        this->m_verbose_level >= 3,
        &this->m_inlet_projection_square_norm
    );
}

template <typename Ttype>
/**
 * Print a summary of the current solution.
 */
void CDStratifiedSolution<Ttype>::print_info()
{
    std::ostringstream info_msg;
    std::ostringstream circle_draw;

    info_msg << "\nCDStratifiedSolution object" << " with " << this->m_ui.size() << " layers:\n"
             << std::endl;

    // Draw circle
    draw_step_solution_terminal(circle_draw);
    info_msg << circle_draw.str() << std::endl;

    // Print info about the densities
    std::vector<Ttype> limits(this->m_zi.size() + 2);
    limits[0] = -1.0;
    for (unsigned i = 0; i < this->m_zi.size(); i++)
    {
        limits[i + 1] = this->m_zi[i];
    }
    limits[this->m_zi.size() + 1] = 1.0;
    info_msg << "\nDensities:\n";
    for (unsigned i = 0; i < this->m_ui.size(); i++)
    {
        double z1 = limits[i], z2 = limits[i + 1];
        double ui_unscaled = this->get_unscaled_solution_value(this->m_ui[i]);
        info_msg << "  - Layer " << i << ": density(" << z1 << " < z <= " << z2 << ") = " << ui_unscaled << std::endl;
    }
    info_msg << std::endl;

    std::cout << info_msg.str() << std::endl;
}

template <typename Ttype>
/**
 * Draw an ASCII representation of the step solution.
 * @param circle_draw Output stream for the drawing.
 */
void CDStratifiedSolution<Ttype>::draw_step_solution_terminal(std::ostringstream &circle_draw)
{
    const int n_interfaces = this->m_n_layers - 1;
    const int radius = 10;
    const double aspect_ratio = 2.0;

    // Calculate the y indices of the interfaces
    std::vector<int> y_layers_inds(n_interfaces, 0);
    for (unsigned i = 0; i < n_interfaces; i++)
    {
        double layer_value = radius * this->m_zi[i];
        int layer_value_int = static_cast<int>(layer_value);

        int increment = (layer_value_int < 0) ? -1 : 1;
        y_layers_inds[i] = (std::abs(layer_value - layer_value_int) >= 0.5) ? layer_value_int + increment : layer_value_int;
    }

    unsigned nth_interface = 0;
    bool on_layer = false, within_circle = false;
    for (int y = radius; y >= -radius; --y)
    {
        circle_draw << "   ";
        for (int x = -radius * aspect_ratio; x <= radius * aspect_ratio; ++x)
        {
            double dist = std::sqrt(std::pow(x / aspect_ratio, 2) + y * y);

            if (std::abs(dist - radius) < 0.5)
            {
                circle_draw << "#";
                within_circle = (dist - radius <= 0.0) ? true : false;
            }
            else
            {
                on_layer = false;
                for (unsigned i = 0; i < n_interfaces; i++)
                {
                    if (y_layers_inds[i] == y)
                    {
                        on_layer = true;
                        break;
                    }
                }

                if (on_layer)
                {
                    circle_draw << ((within_circle) ? "-" : " ");
                }
                else
                {
                    if (within_circle)
                    {
                        char char_to_draw = (nth_interface % 2 == 0) ? '/' : '\\';
                        circle_draw << char_to_draw;
                    }
                    else
                    {
                        circle_draw << " ";
                    }
                }
            }
        }

        if (on_layer)
        {
            for (unsigned i = 0; i < n_interfaces; i++)
            {
                if (y_layers_inds[i] == y)
                {
                    circle_draw << " Interface " << n_interfaces - nth_interface << ", z = " << this->m_zi[this->m_zi.size() - 1 - nth_interface] << "; ";
                    nth_interface++;
                }
            }
        }
        circle_draw << std::endl;
    }
}

#endif
