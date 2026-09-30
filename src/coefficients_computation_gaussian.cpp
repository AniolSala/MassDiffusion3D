#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

#include "coefficients_computation_gaussian.h"
// #include "comp_utils.h"
#include "math_functions.h"

#define SQRT_PI          1.77245385090551602730L
#define SQRT_HALF_PI     1.25331413731550024736L
#define SQRT_2_PI        2.50662827463100024161L
#define INV_SQRT_PI      0.56418958354775628695L
#define INV_SQRT_HALF_PI 0.79788456080286535588L
#define INV_SQRT_2_PI    0.39894228040143267794L

template <typename Ttype>
struct LayerQuadratureData
{
    std::vector<Ttype> r_nodes;
    std::vector<Ttype> phi_nodes;
    std::vector<Ttype> weighted_jacobian;
};

template <typename Ttype>
void precompute_layer_quadrature_data(
    const std::vector<Ttype> &z_vec,
    const unsigned &layers_num,
    const std::vector<Ttype> &gauss_weights,
    const std::vector<Ttype> &gauss_nodes,
    std::vector<LayerQuadratureData<Ttype>> &layer_quadrature_data)
{
    if (layers_num <= 1)
    {
        layer_quadrature_data.clear();
        return;
    }

    const unsigned nodes_num = gauss_nodes.size();
    layer_quadrature_data.clear();
    layer_quadrature_data.resize(layers_num - 1);

    for (unsigned layer_idx = 0; layer_idx < layers_num - 1; ++layer_idx)
    {
        const Ttype z_i_t = z_vec[layer_idx];
        const Ttype one_minus_zsq = static_cast<Ttype>(1.0L) - z_i_t * z_i_t;
        const Ttype y_lim = std::sqrt(one_minus_zsq);
        const Ttype layer_factor = static_cast<Ttype>(2.0L) * one_minus_zsq * one_minus_zsq;

        LayerQuadratureData<Ttype> &layer_data = layer_quadrature_data[layer_idx];
        layer_data.r_nodes.resize(nodes_num);
        layer_data.phi_nodes.resize(nodes_num);
        layer_data.weighted_jacobian.resize(nodes_num);

#pragma omp simd
        for (unsigned node_idx = 0; node_idx < nodes_num; ++node_idx)
        {
            const Ttype y = gauss_nodes[node_idx] * y_lim;
            const Ttype r_z_cnt = std::sqrt(y * y + z_i_t * z_i_t);

            layer_data.r_nodes[node_idx] = r_z_cnt;
            layer_data.phi_nodes[node_idx] = std::atan2(y, z_i_t);
            layer_data.weighted_jacobian[node_idx] = gauss_weights[node_idx] * layer_factor;
        }
    }
}

template <typename Ttype>
/**
 * Compute the layer integral contribution using Gaussian quadrature.
 * @param nn Series index n.
 * @param root Radial root for the basis function.
 * @param z_i Interface position for the layer.
 * @param coefficient_nm_2 Output contribution from Gaussian integration.
 * @param gauss_weights Gaussian quadrature weights.
 * @param gauss_nodes Gaussian quadrature nodes.
 */
void compute_coefficient_nm_i_2(
    const unsigned &nn,
    const Ttype &root,
    const LayerQuadratureData<Ttype> &layer_data,
    Ttype &coefficient_nm_2)
{
    coefficient_nm_2 = 0.;

    const unsigned nodes_num = layer_data.r_nodes.size();
    const Ttype *const r_nodes = layer_data.r_nodes.data();
    const Ttype *const phi_nodes = layer_data.phi_nodes.data();
    const Ttype *const weighted_jacobian = layer_data.weighted_jacobian.data();

    if (nn == 0)
    {
        const Ttype inv_sqrt_2_pi = static_cast<Ttype>(INV_SQRT_2_PI);
        for (unsigned i = 0; i < nodes_num; ++i)
        {
            Ttype psi_nm_y;
            psinm_r(nn, root, r_nodes[i], psi_nm_y);
            coefficient_nm_2 += psi_nm_y * (phi_nodes[i] * inv_sqrt_2_pi) * weighted_jacobian[i];
        }
    }
    else
    {
        const Ttype nn_t = static_cast<Ttype>(nn);
        const Ttype sin_prefactor = static_cast<Ttype>(INV_SQRT_PI) / nn_t;
        for (unsigned i = 0; i < nodes_num; ++i)
        {
            Ttype psi_nm_y;
            psinm_r(nn, root, r_nodes[i], psi_nm_y);
            coefficient_nm_2 += psi_nm_y * (std::sin(nn_t * phi_nodes[i]) * sin_prefactor) * weighted_jacobian[i];
        }
    }

    // auto t0 = std::chrono::high_resolution_clock::now();
    // gaussian_integration(integrand, gauss_weights, gauss_nodes, coefficient_nm_2);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    // std::cout << "n = " << nn << ", b = " << root << ", dt_gaussian = " << dt.count() << " (cpp)" << std::endl;
}

template <typename Ttype>
/**
 * Compute the corrective term for n=0 and z_i<0 using hypergeometric functions.
 * @param root Radial root for the basis function.
 * @param z_i Interface position for the layer.
 * @param coefficient_nm_i Output corrective contribution.
 */
void compute_coefficient_nm_i_1(
    const Ttype &root,
    const Ttype &z_i,
    Ttype &coefficient_nm_i)
{
    const Ttype nu = (static_cast<Ttype>(2.0L) - root) * static_cast<Ttype>(0.25L);
    const Ttype z = root * z_i * z_i;
    const Ttype common_factor = static_cast<Ttype>(-0.5L) * root * z_i * std::exp(static_cast<Ttype>(-0.5L) * z);
    Ttype F1_1, F1_2;
    hypergeometric1F1(nu, 1, z, F1_1);
    hypergeometric1F1(nu + static_cast<Ttype>(1.0L), 2, z, F1_2);

    const Ttype dpsi_0m = common_factor * (static_cast<Ttype>(2.0L) * F1_1 + (root - static_cast<Ttype>(2.0L)) * F1_2);
    coefficient_nm_i = -static_cast<Ttype>(SQRT_2_PI) * z_i / (root * root) * dpsi_0m;

    // coefficient_nm_i = .5 * SQRT_HALF_PI;

    // coefficient_nm_i = 0.0;
}

template <typename Ttype>
/**
 * Compute the closed-form contribution for the n=0 base case.
 * @param z_i Interface position for the layer.
 * @param coefficient_nm_i Output contribution for the base case.
 */
void compute_coefficient_nm_i_0(
    const Ttype &z_i,
    Ttype &coefficient_nm_i)
{
    const Ttype z_i_abs = std::fabs(z_i);
    const Ttype y_lim = std::sqrt(static_cast<Ttype>(1.0L) - z_i * z_i);
    const Ttype phi = std::acos(z_i_abs);
    const Ttype z_pos_int = (phi - z_i_abs * y_lim * (static_cast<Ttype>(5.0L) - static_cast<Ttype>(2.0L) * z_i * z_i) / static_cast<Ttype>(3.0L)) * static_cast<Ttype>(INV_SQRT_2_PI);
    coefficient_nm_i = (z_i > static_cast<Ttype>(0.0L)) ? z_pos_int : static_cast<Ttype>(SQRT_HALF_PI) - z_pos_int;
    // if (z_i < 0.0)
    // {
    //     coefficient_nm_i = SQRT_HALF_PI - z_pos_int;
    // }
    // else
    // {
    //     coefficient_nm_i = z_pos_int;
    // }
    // std::cout << "y_lim = " << y_lim << "z_abs" << z_i_abs << "phi = " << phi << "z_pos_int = " << coefficient_nm_i << std::endl;
}

template <typename Ttype>
/**
 * Combine per-layer contributions for a single interface.
 * @param nn Series index n.
 * @param mm Series index m.
 * @param z_i Interface position for the layer.
 * @param root Radial root for the basis function.
 * @param gauss_weights Gaussian quadrature weights.
 * @param gauss_nodes Gaussian quadrature nodes.
 * @param coefficient_nm_i Output per-layer coefficient contribution.
 */
void compute_coefficient_nm_i(
    const unsigned &nn,
    const unsigned &mm,
    const Ttype &z_i,
    const Ttype &root,
    const LayerQuadratureData<Ttype> &layer_data,
    Ttype &coefficient_nm_i)
{
    Ttype coefficient_nm_1 = 0.;
    Ttype coefficient_nm_2 = 0.;
    if (nn == 0 && z_i < 0.0)
        compute_coefficient_nm_i_1(root, z_i, coefficient_nm_1);

    compute_coefficient_nm_i_2(nn, root, layer_data, coefficient_nm_2);
    coefficient_nm_i = coefficient_nm_1 + coefficient_nm_2;
}

template <typename Ttype>
/**
 * Compute coefficient c_{n,m} by summing layer contributions.
 * @param nn Series index n.
 * @param mm Series index m.
 * @param z_vec Interface positions.
 * @param values Density values per layer.
 * @param root Radial root for the basis function.
 * @param gauss_weights Gaussian quadrature weights.
 * @param gauss_nodes Gaussian quadrature nodes.
 * @param coefficient_nm Output coefficient value.
 */
void compute_coefficient_nm(
    const unsigned &nn,
    const unsigned &mm,
    const std::vector<Ttype> &z_vec,
    const std::vector<Ttype> &values,
    const Ttype &root,
    const std::vector<Ttype> &gauss_weights,
    const std::vector<Ttype> &gauss_nodes,
    Ttype &coefficient_nm)
{
    std::vector<LayerQuadratureData<Ttype>> layer_quadrature_data;
    const unsigned layers_num = values.size();
    precompute_layer_quadrature_data(z_vec, layers_num, gauss_weights, gauss_nodes, layer_quadrature_data);

    coefficient_nm = 0.0;
    for (unsigned i = 0; i < layers_num - 1; i++)
    {
        Ttype values_diff = values[i + 1] - values[i];
        if (values_diff == 0.0)
            continue;

        Ttype coefficient_nm_i = 0.0;
        compute_coefficient_nm_i(nn, mm, z_vec[i], root, layer_quadrature_data[i], coefficient_nm_i);
        coefficient_nm += values_diff * coefficient_nm_i;
    }

    // if (nn == 0 && mm == 0)
    // {
    //     coefficient_nm += .25 * values[0] * SQRT_2_PI;
    // }
}

template <typename Ttype>
void compute_coefficient_nm_precomputed(
    const unsigned &nn,
    const unsigned &mm,
    const std::vector<Ttype> &z_vec,
    const std::vector<Ttype> &values,
    const Ttype &root,
    const std::vector<LayerQuadratureData<Ttype>> &layer_quadrature_data,
    Ttype &coefficient_nm)
{
    coefficient_nm = 0.0;
    const unsigned layers_num = values.size();
    if (layers_num <= 1)
        return;

    for (unsigned i = 0; i < layers_num - 1; ++i)
    {
        const Ttype values_diff = values[i + 1] - values[i];
        if (values_diff == 0.0)
            continue;

        Ttype coefficient_nm_i = 0.0;
        compute_coefficient_nm_i(nn, mm, z_vec[i], root, layer_quadrature_data[i], coefficient_nm_i);
        coefficient_nm += values_diff * coefficient_nm_i;
    }
}

template <typename Ttype>
/**
 * Compute coefficients into series_data. Note that the norm
 * (squared) of the radial eigenfunction is included in the
 * coefficient.
 * @param z_vec Interface positions.
 * @param values Density values per layer.
 * @param gauss_weights Gaussian quadrature weights.
 * @param gauss_nodes Gaussian quadrature nodes.
 * @param max_K Number of series_data entries to compute.
 * @param series_data (n,m,root,norm,coeff) entries updated in place.
 */
void compute_coefficients_matrix(
    const std::vector<Ttype> &z_vec,
    const std::vector<Ttype> &values,
    const std::vector<Ttype> &gauss_weights,
    const std::vector<Ttype> &gauss_nodes,
    const unsigned &max_K,
    std::vector<SeriesTermData<Ttype>> &series_data)
{
    if (max_K == 0) return;

    const unsigned layers_num = values.size();
    if (layers_num <= 1) return;

    // 1. Precompute layer quadrature data (r_nodes, phi_nodes, weighted_jacobian)
    std::vector<LayerQuadratureData<Ttype>> layer_quadrature_data;
    precompute_layer_quadrature_data(z_vec, layers_num, gauss_weights, gauss_nodes, layer_quadrature_data);

    const unsigned nodes_num = gauss_nodes.size();
    const unsigned num_interfaces = layers_num - 1;

    // 2. Pre-filter layers with non-zero values_diff
    std::vector<unsigned> active_layers;
    std::vector<Ttype> active_values_diff;
    active_layers.reserve(num_interfaces);
    active_values_diff.reserve(num_interfaces);
    for (unsigned i = 0; i < num_interfaces; ++i)
    {
        const Ttype vd = values[i + 1] - values[i];
        if (vd != 0.0)
        {
            active_layers.push_back(i);
            active_values_diff.push_back(static_cast<Ttype>(vd));
        }
    }
    const unsigned num_active = active_layers.size();
    if (num_active == 0)
    {
        for (unsigned k = 0; k < max_K; ++k)
            series_data[k].coeff = static_cast<Ttype>(0.0);
        return;
    }

    // 3. Group terms by n-index: n_groups[n] = list of k-indices with that n
    unsigned max_n = 0;
    for (unsigned k = 0; k < max_K; ++k)
        max_n = std::max(max_n, series_data[k].n);

    std::vector<std::vector<unsigned>> n_groups(max_n + 1);
    for (unsigned k = 0; k < max_K; ++k)
    {
        if (series_data[k].n == 0 && series_data[k].m == 0)
        {
            series_data[k].coeff = static_cast<Ttype>(0.0);
            continue; // Skip the constant term (0,0)
        }
        n_groups[series_data[k].n].push_back(k);
    }

    // 4. Precompute angular × weight factors for each (n, active_layer, node)
    //    angular_weight[n][active_layer_idx][node] = angular_factor(n, phi) * weighted_jacobian
    //    For n == 0: angular_factor = phi * INV_SQRT_2_PI
    //    For n > 0:  angular_factor = sin(n * phi) * INV_SQRT_PI / n
    std::vector<std::vector<std::vector<Ttype>>> angular_weight(max_n + 1);
    for (unsigned n = 0; n <= max_n; ++n)
    {
        if (n_groups[n].empty()) continue;

        angular_weight[n].resize(num_active);
        const Ttype n_t = static_cast<Ttype>(n);

        for (unsigned ai = 0; ai < num_active; ++ai)
        {
            const unsigned layer_idx = active_layers[ai];
            const LayerQuadratureData<Ttype> &ld = layer_quadrature_data[layer_idx];
            angular_weight[n][ai].resize(nodes_num);

            if (n == 0)
            {
                const Ttype inv_sqrt_2_pi = static_cast<Ttype>(INV_SQRT_2_PI);
                for (unsigned i = 0; i < nodes_num; ++i)
                    angular_weight[n][ai][i] = ld.phi_nodes[i] * inv_sqrt_2_pi * ld.weighted_jacobian[i];
            }
            else
            {
                const Ttype sin_prefactor = static_cast<Ttype>(INV_SQRT_PI) / n_t;
                for (unsigned i = 0; i < nodes_num; ++i)
                    angular_weight[n][ai][i] = std::sin(n_t * ld.phi_nodes[i]) * sin_prefactor * ld.weighted_jacobian[i];
            }
        }
    }

    // 5. Compute coefficients, parallelized over n-groups
#pragma omp parallel for schedule(dynamic)
    for (unsigned n = 0; n <= max_n; ++n)
    {
        const std::vector<unsigned> &group = n_groups[n];
        if (group.empty()) continue;

        // Process each term (n, m) in this group
        for (unsigned gi = 0; gi < group.size(); ++gi)
        {
            const unsigned k = group[gi];
            SeriesTermData<Ttype> &term_data = series_data[k];
            const Ttype root = term_data.root;
            const unsigned nn = term_data.n;

            try
            {
                Ttype coefficient_nm = static_cast<Ttype>(0.0);
                const bool use_asymptotic = false; // approximation k!*J_k(b*r) is incorrect for this problem

                for (unsigned ai = 0; ai < num_active; ++ai)
                {
                    const unsigned layer_idx = active_layers[ai];
                    const Ttype vd = active_values_diff[ai];
                    const LayerQuadratureData<Ttype> &ld = layer_quadrature_data[layer_idx];
                    const Ttype *const aw = angular_weight[nn][ai].data();
                    const Ttype *const r_nodes = ld.r_nodes.data();

                    // n == 0 corrective term (hypergeometric, only for z_i < 0)
                    Ttype coeff_nm_1 = static_cast<Ttype>(0.0);
                    if (nn == 0 && z_vec[layer_idx] < 0.0)
                        compute_coefficient_nm_i_1(root, z_vec[layer_idx], coeff_nm_1);

                    // Gaussian quadrature with precomputed angular×weight
                    Ttype coeff_nm_2 = static_cast<Ttype>(0.0);
                    for (unsigned i = 0; i < nodes_num; ++i)
                    {
                        Ttype psi_val;
                        psinm_r(nn, root, r_nodes[i], psi_val);
                        coeff_nm_2 += psi_val * aw[i];
                    }

                    coefficient_nm += vd * (coeff_nm_1 + coeff_nm_2);
                }

                term_data.coeff = coefficient_nm / term_data.norm;
            }
            catch (const std::exception &e)
            {
                std::cout << "Error in coefficient (n, m) = (" << term_data.n << ", " << term_data.m << "):" << std::endl;
                std::cerr << e.what() << '\n';
                exit(1);
            }
        }
    }
}

template void compute_coefficients_matrix<double>(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &,
    const unsigned &,
    std::vector<SeriesTermData<double>> &);

template void compute_coefficients_matrix<long double>(
    const std::vector<long double> &,
    const std::vector<long double> &,
    const std::vector<long double> &,
    const std::vector<long double> &,
    const unsigned &,
    std::vector<SeriesTermData<long double>> &);

template void compute_coefficient_nm<double>(
    const unsigned &,
    const unsigned &,
    const std::vector<double> &,
    const std::vector<double> &,
    const double &,
    const std::vector<double> &,
    const std::vector<double> &,
    double &);

template void compute_coefficient_nm<long double>(
    const unsigned &,
    const unsigned &,
    const std::vector<long double> &,
    const std::vector<long double> &,
    const long double &,
    const std::vector<long double> &,
    const std::vector<long double> &,
    long double &);
