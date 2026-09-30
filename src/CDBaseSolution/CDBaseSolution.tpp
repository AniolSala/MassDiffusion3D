#if !defined(CD_BASE_SOLUTION_TPP)
#define CD_BASE_SOLUTION_TPP

#ifndef CD_BASE_SOLUTION_H
#include "CDBaseSolution.h"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <cassert>
#include <chrono>
#include <exception>
#include <omp.h>

#include <limits>
#include <array>
#include <map>
#include <mutex>
#include <utility>

#include "../io_data.h"
#include "../math_functions.h"
#include "../roots_and_norms_calculations.h"
#include "../finite_peclet_roots.h"
#include "../finite_peclet_radial.h"
#include "../finite_peclet_rhs.h"

template <typename Ttype>
/**
 * Construct a CDBaseSolution instance with layer data.
 * @param zi Interface positions.
 * @param ui Layer density values.
 * @param verbose_level Verbosity level.
 *
 * NOTE: print_info() is pure virtual and must be called by the derived class
 * constructor after the derived object is fully constructed.
 */
CDBaseSolution<Ttype>::CDBaseSolution(
    const std::vector<Ttype> &zi,
    const std::vector<Ttype> &ui,
    unsigned short verbose_level)
    : m_verbose_level(verbose_level), m_number_of_gauss_points(100), m_max_K(0), m_max_warnings(10), m_current_warning(0), mCoefficientsAreFixed(false), mCoefficientsAreComputed(false),
      m_use_axial_diffusion(false),
      m_peclet(std::numeric_limits<Ttype>::infinity()),
      m_kappa(static_cast<Ttype>(0)),
      m_fp_rel_tol(static_cast<Ttype>(1e-13L)),
      m_fp_max_iter(200u),
      m_fp_quad_panels(64u),
      m_qep_kappa(static_cast<Ttype>(-1)),
      m_inlet_projection_square_norm(std::numeric_limits<Ttype>::quiet_NaN())
{
    m_n_layers = 0;
    m_max_root = static_cast<Ttype>(400);
    set_zi(zi);
    set_ui(ui);

    // Data directory lives alongside this file at src/CDBaseSolution/data/
    std::filesystem::path file_path(__FILE__), data_files_path("data");
    std::filesystem::path module_path = file_path.parent_path();
    m_data_dir = module_path / data_files_path;

    // Load the gaussian weights and points, and compute the constant term
    set_gaussian_weights_and_points();
    set_constant_term();

    // NOTE: Derived class constructors should call print_info() if m_verbose_level >= 1.
}

// Setters
template <typename Ttype>
/**
 * Set interface positions.
 * @param zi Interface positions in ascending order.
 */
void CDBaseSolution<Ttype>::set_zi(std::vector<Ttype> zi)
{
    for (unsigned i = 1; i < zi.size(); i++)
    {
        if (zi[i] - zi[i - 1] <= 0.)
        {
            throw std::invalid_argument("Vector variable `zi` must be given in ascending order!");
        }
    }

    modify_n_layers(zi.size() + 1);
    m_zi = zi;
}

template <typename Ttype>
/**
 * Set layer values and normalize them.
 * @param ui Layer density values.
 */
void CDBaseSolution<Ttype>::set_ui(std::vector<Ttype> ui)
{
    modify_n_layers(ui.size());
    m_ui = ui;

    // Solution is normalized
    m_max_ui = ui[0];
    m_min_ui = ui[0];
    for (unsigned i = 1; i < ui.size(); i++)
    {
        if (ui[i] > m_max_ui)
            m_max_ui = ui[i];
        if (ui[i] < m_min_ui)
            m_min_ui = ui[i];
    }

    // Scale the solution between 1 and 0
    for (unsigned i = 0; i < ui.size(); i++)
        m_ui[i] = (m_max_ui - ui[i]) / (m_max_ui - m_min_ui);
}

template <typename Ttype>
/**
 * Unscale the solution.
 * @param ui Scaled solution value.
 * @return Unscaled solution value.
 */
Ttype CDBaseSolution<Ttype>::get_unscaled_solution_value(const Ttype &ui)
{
    return m_max_ui - ui * (m_max_ui - m_min_ui);
}

template <typename Ttype>
/**
 * Set the maximum eigenvalue root used for series truncation, snapping to the
 * nearest root actually present in the problem's eigenbasis table.  Calling
 * set_roots() (virtual) is safe here because set_max_root() is never invoked
 * during construction.
 * @param root_value Desired cutoff root value.
 */
void CDBaseSolution<Ttype>::set_max_root(Ttype root_value)
{
    if (!(root_value > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(root_value)))
        throw std::invalid_argument("set_max_root: root cutoff must be finite and positive.");
    if (root_value == m_max_root)
        return;
    m_max_root = root_value;
    invalidate_solution();
}

template <typename Ttype>
/**
 * Set the relative amplitude below which a mode is skipped when the series is
 * evaluated. See the declaration in CDBaseSolution.h for what is compared and
 * why. Evaluation-time only: the mode set and the coefficients are untouched,
 * so no invalidation is needed and the value may change between calls.
 * @param tol Relative cutoff; 0 (default) evaluates every mode.
 */
void CDBaseSolution<Ttype>::cutoff_modes(Ttype tol)
{
    if (!(tol >= static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(tol)))
        throw std::invalid_argument("cutoff_modes: tolerance must be finite and non-negative.");
    if (tol >= static_cast<Ttype>(1))
        throw std::invalid_argument("cutoff_modes: tolerance must be below 1 (it is relative to the "
                                    "largest modal amplitude of the plane).");
    m_eval_cutoff_tol = tol;
}

template <typename Ttype>
/**
 * Largest |R_k(r)| over the radii of an already-built radial table.
 * @param radial_table Row-major table, one row of mode_count entries per unique radius.
 * @param mode_count Number of modes per row.
 * @param max_radial Output, one entry per mode.
 */
void CDBaseSolution<Ttype>::radial_row_maxima(const std::vector<Ttype> &radial_table, size_t mode_count,
                                              std::vector<Ttype> &max_radial) const
{
    max_radial.assign(mode_count, static_cast<Ttype>(0));
    if (!mode_count)
        return;
    const size_t rows = radial_table.size() / mode_count;
    for (size_t u = 0; u < rows; u++)
    {
        const Ttype *row = &radial_table[u * mode_count];
        for (size_t k = 0; k < mode_count; k++)
        {
            const Ttype v = std::fabs(row[k]);
            if (v > max_radial[k])
                max_radial[k] = v;
        }
    }
}

template <typename Ttype>
/**
 * Half-open k-ranges the evaluation loop must sum at one plane.
 *
 * With the cut disabled (or no radial table available) this is the single range
 * [0, mode_count), i.e. the untruncated loop. With it enabled, each angular
 * block contributes the prefix of its modes whose amplitude
 * |coeff_gnm[k]| * max_u |R_k(r_u)| reaches m_eval_cutoff_tol times the largest
 * amplitude of the plane. The rate grows with m inside a block, so a prefix is
 * the natural shape; taking the last mode above the threshold (rather than the
 * first below it) keeps the cut safe when the amplitudes are not monotone.
 *
 * @param coeff_gnm Per-mode coefficient of this plane, C_k exp(-Lam_k x).
 * @param max_radial Per-mode max |R_k|, empty when no radial table was built.
 * @param mode_count Number of modes.
 * @param ranges Output ranges, ascending and disjoint.
 */
void CDBaseSolution<Ttype>::build_eval_ranges(const std::vector<Ttype> &coeff_gnm,
                                              const std::vector<Ttype> &max_radial, size_t mode_count,
                                              std::vector<std::pair<size_t, size_t>> &ranges) const
{
    ranges.clear();
    const bool cut = m_eval_cutoff_tol > static_cast<Ttype>(0) && max_radial.size() == mode_count &&
                     m_block_start.size() >= 2 && m_block_start.back() == mode_count;
    if (!cut)
    {
        if (mode_count)
            ranges.emplace_back(static_cast<size_t>(0), mode_count);
        return;
    }

    Ttype peak = static_cast<Ttype>(0);
    for (size_t k = 0; k < mode_count; k++)
    {
        const Ttype amp = std::fabs(coeff_gnm[k]) * max_radial[k];
        if (amp > peak)
            peak = amp;
    }
    if (!(peak > static_cast<Ttype>(0)))
    {
        ranges.emplace_back(static_cast<size_t>(0), mode_count);
        return;
    }
    const Ttype threshold = m_eval_cutoff_tol * peak;

    ranges.reserve(m_block_start.size());
    for (size_t bi = 0; bi + 1 < m_block_start.size(); bi++)
    {
        const size_t begin = m_block_start[bi], end = m_block_start[bi + 1];
        size_t keep = begin;
        for (size_t k = begin; k < end; k++)
        {
            // The constant mode carries the cross-sectional mean and never decays;
            // keep it whatever the threshold says.
            const bool protectedMode = is_zero_eigenvalue_mode(m_series_data[k].n, m_series_data[k].m);
            if (protectedMode || std::fabs(coeff_gnm[k]) * max_radial[k] >= threshold)
                keep = k + 1;
        }
        if (keep > begin)
            ranges.emplace_back(begin, keep);
    }
}

template <typename Ttype>
/**
 * Return the truncation tolerance achieved by m_max_root at a given x.
 * @param x_value Axial coordinate.
 * @return exp(-m_max_root² * x_value)
 */
Ttype CDBaseSolution<Ttype>::get_tol(Ttype x_value) const
{
    return std::exp(-m_max_root * m_max_root * x_value);
}

template <typename Ttype>
/**
 * Validate and set the number of layers.
 * @param new_n_layers Desired number of layers.
 */
void CDBaseSolution<Ttype>::modify_n_layers(unsigned new_n_layers)
{
    if (m_n_layers > 0 && new_n_layers != m_n_layers)
    {
        std::ostringstream err_msg;
        err_msg << "Trying to set " << new_n_layers << " layer(s), but CDBaseSolution already has " << m_n_layers << " layer(s)." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }
    m_n_layers = new_n_layers;
}

template <typename Ttype>
/**
 * Initialize the flat series-term vector, truncated at m_max_root.
 */
void CDBaseSolution<Ttype>::set_series_data()
{
    std::vector<std::vector<Ttype>> roots, norms;
    set_roots(roots);
    set_norms(norms);

    if (roots.size() != norms.size())
        throw std::runtime_error("Bare root and norm catalogues have different row counts.");

    // Initialize the vector of series terms data
    m_max_K = 0;
    m_series_data.clear();
    for (unsigned n = 0; n < roots.size(); n++)
    {
        if (roots[n].size() != norms[n].size())
            throw std::runtime_error("Bare root and norm catalogues have different row lengths.");
        for (unsigned m = 0; m < roots[n].size(); m++)
        {
            Ttype root = roots[n][m], norm = norms[n][m];
            if (root > m_max_root)
                break;
            // The beta=0 constant mode is skipped by the offline norm generator, which
            // leaves norms[0][0] = 0. Its true weighted norm is N^2 = int_0^1 (1-r^2) r dr
            // = 1/4 (R_00 == 1). Supply the exact value so the mode carries a valid norm.
            if (is_zero_eigenvalue_mode(n, m) && !(norm > static_cast<Ttype>(0)))
                norm = static_cast<Ttype>(0.25L);
            SeriesData term_data = {n, m, root, norm, 0.0, root * root};
            m_series_data.push_back(term_data);
            m_max_K++;
        }
    }
    if (m_series_data.empty())
        throw std::runtime_error("No bare modes satisfy max_root.");

    // Angular-block offsets. The loop above appends n-major with m ascending, so
    // each block is one contiguous run of equal term_data.n; cutoff_modes relies
    // on that, and on the rate increasing along the run.
    m_block_start.clear();
    m_block_start.push_back(0);
    for (size_t k = 1; k < m_series_data.size(); k++)
        if (m_series_data[k].n != m_series_data[k - 1].n)
            m_block_start.push_back(k);
    m_block_start.push_back(m_series_data.size());
}

template <typename Ttype>
/**
 * Compute partial flux for a given z value.
 * @param z_val Axial coordinate.
 * @return Dimensionless partial flux.
 */
Ttype CDBaseSolution<Ttype>::compute_partial_flux(const Ttype z_val) const
{
    // std::-qualified deliberately. Unqualified abs() on a floating-point
    // argument finds only ::abs(int) unless some header in the including TU has
    // pulled <math.h> (<cmath> puts the double overload in std only), so
    // abs(-0.4) silently truncated to 0 in any TU that had not -- which made
    // every cap area, and with it m_constant_term, wrong in that TU alone while
    // the rest of the library was right. sqrt/acos worked by the same accident
    // and would have cost the long-double instantiation its precision.
    Ttype z_abs = std::abs(z_val);
    Ttype y_lim = std::sqrt(static_cast<Ttype>(1) - z_val * z_val);
    Ttype dimless_flux = (z_abs * y_lim * (2 * z_val * z_val - 5.) + 3. * std::acos(z_abs)) / 6.;
    if (z_val < 0.0)
    {
        dimless_flux = M_PI_2 - dimless_flux;
    }

    return dimless_flux;
}

template <typename Ttype>
/**
 * Compute and store the constant term of the solution.
 */
void CDBaseSolution<Ttype>::set_constant_term()
{
    std::vector<Ttype> zi_with_walls(m_n_layers + 1);
    for (unsigned i = 0; i < m_n_layers - 1; i++)
    {
        zi_with_walls[i + 1] = m_zi[i];
    }
    zi_with_walls[0] = -1.;
    zi_with_walls[m_n_layers] = 1.;

    // Compute the constant term
    m_constant_term = 0;
    Ttype flux_1, flux_2, flux_layer = 0., total_flux = 0;
    Ttype z1 = zi_with_walls[0];
    flux_1 = compute_partial_flux(z1);
    for (unsigned i = 0; i < m_n_layers; i++)
    {
        // Compute the flux of the layer between z1 and z2
        Ttype z2 = zi_with_walls[i + 1];
        flux_2 = compute_partial_flux(z2);
        flux_layer = flux_1 - flux_2;
        total_flux += flux_layer;

        m_constant_term += flux_layer * m_ui[i];

        z1 = z2;
        flux_1 = flux_2;
    }

    m_constant_term /= total_flux;
}

template <typename Ttype>
/**
 * Set the number of Gaussian quadrature points.
 * @param num Number of Gaussian points (100 or 300).
 */
void CDBaseSolution<Ttype>::set_number_of_gauss_points(unsigned num)
{
    if (num != 100 && num != 300 && num != 400)
    {
        std::stringstream err_msg("Number of gauss points must be either 100, 300 or 400");
        throw std::runtime_error(err_msg.str());
    }

    if (m_number_of_gauss_points == num)
        return;
    m_number_of_gauss_points = num;
    set_gaussian_weights_and_points();
    invalidate_solution();
}

template <typename Ttype>
void CDBaseSolution<Ttype>::invalidate_solution()
{
    m_solution_method = SolutionMethod::Uninitialized;
    m_series_data.clear();
    m_block_start.clear();
    m_max_K = 0;
    m_qep_blocks.clear();
    m_qep_kappa = static_cast<Ttype>(-1);
    m_inlet_projection_square_norm = std::numeric_limits<Ttype>::quiet_NaN();
    m_peclet = static_cast<Ttype>(0);
    m_kappa = static_cast<Ttype>(0);
    m_use_axial_diffusion = false;
    mCoefficientsAreFixed = false;
    mCoefficientsAreComputed = false;
    m_blurriness_blocks.clear();
    m_blurriness_tail = std::numeric_limits<Ttype>::quiet_NaN();
    m_blurriness_far_field = std::numeric_limits<Ttype>::quiet_NaN();
    m_blurriness_denominator = std::numeric_limits<Ttype>::quiet_NaN();
    m_blurriness_far_field_form = std::numeric_limits<Ttype>::quiet_NaN();
    m_blurriness_ready = false;
}

template <typename Ttype>
void CDBaseSolution<Ttype>::require_solution_ready(const char *caller) const
{
    if (m_solution_method == SolutionMethod::Uninitialized)
        throw std::logic_error(std::string(caller) + ": no active solution; call setup_bare_solution(), setup_qep_solution(), or setup_fp_solution() first.");
}

template <typename Ttype>
/**
 * Load Gaussian weights and nodes from disk.
 *
 * The file is located solely by m_data_dir (fixed at compile time, identical
 * for every CDBaseSolution<Ttype>-derived instance) and m_number_of_gauss_points
 * (one of 100/300/400), so its content is a pure function of that one integer --
 * never of zi/ui/T0/T_wall or any other instance state. Cache the parsed
 * (weights, points) pair per N, shared across every instance in the process,
 * so a program that constructs many solution objects (e.g. one object per
 * Peclet value, as intended) only reads/parses each distinct N once. A mutex
 * guards the cache because, unlike the single-slot caches in
 * CDStratifiedSolution/CDGraetzIsothermalSolution's set_roots() etc., this one
 * has multiple keys and can still be populated after the first construction
 * (whenever set_number_of_gauss_points() is called with a new N).
 */
void CDBaseSolution<Ttype>::set_gaussian_weights_and_points()
{
    static std::mutex cache_mutex;
    static std::map<unsigned, std::pair<std::vector<Ttype>, std::vector<Ttype>>> cache;

    const std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(m_number_of_gauss_points);
    if (it == cache.end())
    {
        std::ostringstream gauss_params_file_name;
        gauss_params_file_name << "gaussian_weights_eigenvalues_hypergeometric_n_" << m_number_of_gauss_points << ".txt";

        std::filesystem::path gauss_params_file = m_data_dir / std::filesystem::path(gauss_params_file_name.str());
        std::vector<std::vector<Ttype>> gauss_weights_and_norms;
        readValuesFromFile(gauss_params_file.string(), gauss_weights_and_norms);

        std::vector<Ttype> weights(m_number_of_gauss_points), points(m_number_of_gauss_points);
        for (unsigned i = 0; i < gauss_weights_and_norms.size(); i++)
        {
            weights[i] = gauss_weights_and_norms[i][0];
            points[i] = gauss_weights_and_norms[i][1];
        }
        it = cache.emplace(m_number_of_gauss_points, std::make_pair(std::move(weights), std::move(points))).first;
    }
    m_gauss_weights = it->second.first;
    m_gauss_points = it->second.second;
}

template <typename Ttype>
/**
 * Set the maximum number of warnings to print.
 * @param max_warnings Maximum warning count.
 */
void CDBaseSolution<Ttype>::set_max_warnings(unsigned max_warnings)
{
    if (max_warnings < 0)
    {
        std::cout << "Warning: max_warnings can not be less than 0. Tried to set with max_warnings = " << max_warnings << ". Set to default (3)." << std::endl;
        m_max_warnings = 3;
    }
    else
    {
        m_max_warnings = max_warnings;
    }
}

template <typename Ttype>
/**
 * Organize a raw list of x, r and phi points into x-planes with the same (r, phi) points, i.e.
 * group the points that have the same x-value.
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @param x_points_planes Axial coordinates of each plane.
 * @param r_points_planes Radial coordinates within each plane.
 * @param phi_points_planes Angular coordinates within each plane.
 * @param ordering_map Maps original indices to (plane_index, position_in_plane) pairs.
 * @param tol Tolerance for comparing x values (default 1e-8). If difference between two x values is less than tol, they are considered the same plane.
 */
void CDBaseSolution<Ttype>::group_points_in_planes(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points, std::vector<Ttype> &x_points_planes, std::vector<std::vector<Ttype>> &r_points_planes, std::vector<std::vector<Ttype>> &phi_points_planes, std::vector<std::pair<size_t, size_t>> &ordering_map, Ttype tol)
{
    if (x_points.size() != r_points.size() || x_points.size() != phi_points.size())
    {
        std::cerr << "Error: Input arrays must have the same size!" << std::endl;
        return;
    }

    if (x_points.empty())
    {
        return;
    }

    // Create a mapping of sorted x values to their indices
    // Serial on purpose: the body is a single store, so an OpenMP hand-out costs
    // more than the work it distributes, and the std::sort below dominates anyway.
    std::vector<std::pair<Ttype, size_t>> x_with_idx(x_points.size());
    for (size_t i = 0; i < x_points.size(); i++)
    {
        x_with_idx[i] = {x_points[i], i};
    }

    // Sort by x value
    std::sort(x_with_idx.begin(), x_with_idx.end());

    // Group points by x value (sequential due to tolerance grouping dependency)
    x_points_planes.clear();
    r_points_planes.clear();
    phi_points_planes.clear();
    ordering_map.clear();
    ordering_map.resize(x_points.size());

    std::vector<size_t> plane_starts;
    plane_starts.reserve(x_with_idx.size());

    size_t i = 0;
    while (i < x_with_idx.size())
    {
        plane_starts.push_back(i);
        const Ttype current_x = x_with_idx[i].first;
        i++;
        while (i < x_with_idx.size() && std::fabs(x_with_idx[i].first - current_x) < tol)
        {
            i++;
        }
    }

    const size_t plane_count = plane_starts.size();
    x_points_planes.resize(plane_count);
    r_points_planes.resize(plane_count);
    phi_points_planes.resize(plane_count);

    std::vector<size_t> plane_ends(plane_count, x_with_idx.size());
    for (size_t p = 0; p + 1 < plane_count; p++)
    {
        plane_ends[p] = plane_starts[p + 1];
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t p = 0; p < plane_count; p++)
    {
        const size_t start = plane_starts[p];
        const size_t end = plane_ends[p];
        const size_t plane_size = end - start;

        // Collect original indices, r, and phi for this plane
        std::vector<size_t> orig_indices(plane_size);
        std::vector<Ttype> r_plane(plane_size);
        std::vector<Ttype> phi_plane(plane_size);

        for (size_t j = start; j < end; j++)
        {
            const size_t local_j = j - start;
            const size_t original_idx = x_with_idx[j].second;
            orig_indices[local_j] = original_idx;
            r_plane[local_j] = r_points[original_idx];
            phi_plane[local_j] = phi_points[original_idx];
        }

        // Sort by (r, norm_phi(phi)) for canonical ordering independent of input sequence.
        // norm_phi maps any angle to [-π, π), so +π and -π sort identically.
        auto norm_phi = [](Ttype phi) -> Ttype
        {
            const Ttype two_pi = static_cast<Ttype>(2.0 * M_PI);
            const Ttype pi = static_cast<Ttype>(M_PI);
            phi = std::fmod(phi + pi, two_pi);
            if (phi < 0)
                phi += two_pi;
            return phi - pi;
        };

        std::vector<size_t> perm(plane_size);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b)
                  {
            if (r_plane[a] != r_plane[b]) return r_plane[a] < r_plane[b];
            return norm_phi(phi_plane[a]) < norm_phi(phi_plane[b]); });

        // Apply permutation and record sorted positions in ordering_map
        std::vector<Ttype> sorted_r(plane_size);
        std::vector<Ttype> sorted_phi(plane_size);
        for (size_t local_pos = 0; local_pos < plane_size; local_pos++)
        {
            const size_t src = perm[local_pos];
            sorted_r[local_pos] = r_plane[src];
            sorted_phi[local_pos] = phi_plane[src];
            ordering_map[orig_indices[src]] = {p, local_pos};
        }

        x_points_planes[p] = x_with_idx[start].first;
        r_points_planes[p] = std::move(sorted_r);
        phi_points_planes[p] = std::move(sorted_phi);
    }
}

template <typename Ttype>
/**
 * Evaluate the solution at an unstructured cloud of points.
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values in the same order as input points.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    return get_solution_cloud_of_points(x_points, r_points, phi_points);
}

template <typename Ttype>
/**
 * Evaluate the solution at an unstructured cloud of points.
 * @param points Array of coordinates: points[0] = x, points[1] = r, points[2] = phi.
 * @return Solution values in the same order as input points.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution(const std::array<std::vector<Ttype>, 3> &points)
{
    return get_solution_cloud_of_points(points[0], points[1], points[2]);
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution_cloud_of_points(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    const bool measure_time = (m_verbose_level >= 2);
    const auto t_start = std::chrono::steady_clock::now();

    // First make sure that all the data has the same size
    if (x_points.size() != r_points.size() || x_points.size() != phi_points.size())
    {
        std::ostringstream err_msg;
        err_msg << "Input arrays must have the same size. Got x_points with " << x_points.size() << " elements, "
                << "r_points with " << r_points.size() << " elements, and phi_points with " << phi_points.size() << " elements.";
        throw std::invalid_argument(err_msg.str());
    }

    if (x_points.empty())
    {
        return std::vector<Ttype>();
    }

    require_solution_ready("get_solution_cloud_of_points");
    if (m_solution_method == SolutionMethod::QEP)
    {
        std::vector<Ttype> sol;
        qep_evaluate_at_points<Ttype>(m_qep_blocks, x_points, r_points, phi_points, sol);
        for (auto &value : sol)
            value = get_unscaled_solution_value(value);
        return sol;
    }

    // Second, group the points in planes using group_points_in_planes
    std::vector<Ttype> x_planes;
    std::vector<std::vector<Ttype>> r_planes;
    std::vector<std::vector<Ttype>> phi_planes;
    std::vector<std::pair<size_t, size_t>> ordering_map;

    group_points_in_planes(x_points, r_points, phi_points, x_planes, r_planes, phi_planes, ordering_map);

    // If all the planes are the same, use the same values of r and phi for all the planes.
    // After group_points_in_planes, each plane is sorted by (r, norm_phi), so element-wise
    // comparison is order-independent. Tolerances handle floating-point noise; the two_pi
    // branch catches the phi = +pi vs -pi boundary equivalence.
    bool planes_are_equal = true;
    if (x_planes.size() > 1)
    {
        const Ttype r_tol = static_cast<Ttype>(1e-10);
        const Ttype phi_tol = static_cast<Ttype>(1e-10);
        const Ttype two_pi = static_cast<Ttype>(2.0 * M_PI);

        for (size_t p = 1; p < x_planes.size(); p++)
        {
            if (r_planes[p].size() != r_planes[0].size() || phi_planes[p].size() != phi_planes[0].size())
            {
                planes_are_equal = false;
                break;
            }

            for (size_t i = 0; i < r_planes[p].size(); i++)
            {
                const Ttype r_diff = std::fabs(r_planes[p][i] - r_planes[0][i]);
                const Ttype phi_diff = std::fabs(phi_planes[p][i] - phi_planes[0][i]);
                const bool phi_ok = phi_diff < phi_tol || std::fabs(phi_diff - two_pi) < phi_tol;

                if (r_diff >= r_tol || !phi_ok)
                {
                    planes_are_equal = false;
                    break;
                }
            }

            if (!planes_are_equal)
                break;
        }
    }

    // Get solution for each plane
    std::vector<std::vector<Ttype>> sol_result;

    if (planes_are_equal)
    {
        if (m_verbose_level >= 2)
            std::cout << "Using the single plane overload to compute the solution." << std::endl;

        // Check if the planes are empty
        if (x_planes.empty())
        {
            return std::vector<Ttype>();
        }

        // All planes have the same r and phi points, use the first overload
        std::vector<std::vector<Ttype>> sol_planes = get_solution_at_planes(x_planes, r_planes[0], phi_planes[0]);
        sol_result = sol_planes;
    }
    else
    {
        if (m_verbose_level >= 2)
            std::cout << "Using the multiple plane overload to compute the solution." << std::endl;

        // Planes have different r and phi points, use the second overload
        if (x_planes.empty())
        {
            return std::vector<Ttype>();
        }

        std::vector<std::vector<Ttype>> sol_planes = get_solution_at_planes(x_planes, r_planes, phi_planes);
        sol_result = sol_planes;
    }

    // Return the solution keeping the original ordering of the points
    std::vector<Ttype> ordered_result(x_points.size());

    for (size_t i = 0; i < ordering_map.size(); i++)
    {
        size_t plane_idx = ordering_map[i].first;
        size_t pos_in_plane = ordering_map[i].second;
        ordered_result[i] = sol_result[plane_idx][pos_in_plane];
    }

    if (measure_time)
    {
        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "Timing: get_solution_cloud_of_points took " << elapsed_s << " s." << std::endl;
    }

    return ordered_result;
}

template <typename Ttype>
/**
 * Evaluate the solution at multiple planes. The i-th plane has x value x_points[i],
 * and each plane has the same (r, phi) points given by r_points and phi_points.
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values at the given points for each value of x.
 */
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_solution_at_planes(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    const bool measure_time = (m_verbose_level >= 2);
    const auto t_start = std::chrono::steady_clock::now();

    require_solution_ready("get_solution_at_planes");
    if (m_solution_method == SolutionMethod::QEP)
    {
        if (r_points.size() != phi_points.size())
            throw std::invalid_argument("get_solution_at_planes: r and phi sizes differ.");
        std::vector<Ttype> xs, rs, ps;
        for (const auto &x : x_points)
            for (size_t i = 0; i < r_points.size(); ++i)
            {
                xs.push_back(x);
                rs.push_back(r_points[i]);
                ps.push_back(phi_points[i]);
            }
        const std::vector<Ttype> values = get_solution_cloud_of_points(xs, rs, ps);
        std::vector<std::vector<Ttype>> out(x_points.size(), std::vector<Ttype>(r_points.size()));
        for (size_t p = 0; p < x_points.size(); ++p)
            for (size_t i = 0; i < r_points.size(); ++i)
                out[p][i] = values[p * r_points.size() + i];
        return out;
    }
    std::vector<std::vector<Ttype>> sol_planes(x_points.size(), std::vector<Ttype>(r_points.size(), 0.0));
    if (x_points.empty())
    {
        if (measure_time)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "Timing: get_solution_at_planes (shared r,phi) took " << elapsed_s << " s." << std::endl;
        }
        return sol_planes;
    }
    else if (x_points.size() == 1)
    {
        prepare_to_compute_solution(r_points, phi_points);
        compute_solution(x_points[0], r_points, phi_points, sol_planes[0]);
        if (measure_time)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "Timing: get_solution_at_planes (shared r,phi) took " << elapsed_s << " s." << std::endl;
        }
        return sol_planes;
    }

    // Sort x_points
    std::vector<size_t> order(x_points.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
              { return x_points[a] < x_points[b]; });

    if (!mCoefficientsAreFixed || m_series_data.empty())
    {
        prepare_to_compute_solution(r_points, phi_points);
    }

    const size_t full_k = m_max_K;
    const size_t full_k_size = static_cast<size_t>(full_k);
    // Reference, not a copy: the mode list is fixed for the whole call (only
    // set_series_data / invalidate_solution / setup_* write it, none of which run
    // during an evaluation), and the copy is 80 B per mode (160 B for long double).
    const std::vector<SeriesData> &full_series_data = m_series_data;

    // Precompute x-independent data: psinm cache and r_index
    bool use_r_cache = false;
    std::vector<size_t> r_index(r_points.size(), 0);
    std::vector<Ttype> unique_r;
    std::vector<Ttype> psinm_cache;

    if (!r_points.empty() && full_k > 0)
    {
        const Ttype max_r = *std::max_element(r_points.begin(), r_points.end());
        const Ttype r_tol = std::max(static_cast<Ttype>(1e-12), static_cast<Ttype>(1e-8) * std::max(max_r, static_cast<Ttype>(1.0)));

        std::vector<unsigned> order_r(r_points.size());
        std::iota(order_r.begin(), order_r.end(), 0);
        std::sort(order_r.begin(), order_r.end(), [&](unsigned a, unsigned b)
                  { return r_points[a] < r_points[b]; });

        unique_r.reserve(r_points.size());

        bool grouping_ok = true;
        unsigned start = 0;
        while (start < r_points.size())
        {
            const unsigned idx = order_r[start];
            const Ttype r0 = r_points[idx];
            Ttype r_min = r0;
            Ttype r_max = r0;
            unsigned end = start + 1;
            while (end < r_points.size())
            {
                const Ttype r_val = r_points[order_r[end]];
                if (r_val - r0 > r_tol)
                    break;
                r_min = std::min(r_min, r_val);
                r_max = std::max(r_max, r_val);
                end++;
            }

            if ((r_max - r_min) > r_tol)
            {
                grouping_ok = false;
                break;
            }

            const size_t group_index = unique_r.size();
            unique_r.push_back((r_min + r_max) * static_cast<Ttype>(0.5));
            for (unsigned j = start; j < end; j++)
            {
                r_index[order_r[j]] = group_index;
            }
            start = end;
        }

        const size_t max_cache_bytes = static_cast<size_t>(64) * 1024 * 1024;
        const size_t cache_bytes = grouping_ok ? unique_r.size() * full_k_size * sizeof(Ttype) : 0;
        use_r_cache = grouping_ok && unique_r.size() < r_points.size() && cache_bytes > 0 && cache_bytes <= max_cache_bytes;

        if (use_r_cache)
        {
            psinm_cache.resize(unique_r.size() * full_k_size, static_cast<Ttype>(0.0));

            // Parallelise over the flattened (unique radius, mode) index rather than
            // over radii alone: a table has only as many rows as there are distinct
            // radii, which is fewer work items than threads on a coarse radial grid,
            // and the per-radius cost varies (the confluent-hypergeometric evaluation
            // switches method with z = b r^2), so row-level hand-out cannot balance it.
            const size_t cache_items = unique_r.size() * full_k_size;
            const int cache_chunk = radial_table_chunk(cache_items);
#pragma omp parallel for schedule(dynamic, cache_chunk)
            for (size_t idx = 0; idx < cache_items; idx++)
            {
                const size_t u = idx / full_k_size;
                const size_t k = idx - u * full_k_size;
                psinm_cache[idx] = active_radial(full_series_data[k], unique_r[u]);
            }
        }
    }

    // Per-mode max |R_k| over the table's radii: the scale cutoff_modes compares
    // against. Built once per call (the table is x-independent); empty when the
    // cut is off or no table exists, which build_eval_ranges reads as "keep all".
    std::vector<Ttype> max_radial;
    if (use_r_cache && m_eval_cutoff_tol > static_cast<Ttype>(0))
        radial_row_maxima(psinm_cache, full_k_size, max_radial);

    // Loop over x-planes and compute solutions. Truncation is already baked into
    // full_series_data by set_series_data (cutoff m_max_root), so every plane uses
    // the same fixed mode set.
    const size_t num_points = r_points.size();
    for (size_t pos = 0; pos < order.size(); pos++)
    {
        const size_t idx = order[pos];
        const Ttype x_val = x_points[idx];

        // x-dependent coefficient of each mode at this plane.
        std::vector<Ttype> coeff_gnm(full_k_size, 0.0);
        // Uniform cost per mode (one exp), so a static split has no imbalance to
        // correct and avoids a hand-out that costs more than the body itself.
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < full_k_size; k++)
        {
            const SeriesData &term_data = full_series_data[k];
            coeff_gnm[k] = active_coeff(term_data) * std::exp(-active_rate(term_data) * x_val);
        }

        // Modes worth summing at this plane (one prefix per angular block); the
        // single full range when the cut is off, so the loop below is unchanged.
        std::vector<std::pair<size_t, size_t>> eval_ranges;
        build_eval_ranges(coeff_gnm, max_radial, full_k_size, eval_ranges);

        // Cached points all cost the same (K multiply-adds), so hand them out in
        // blocks; uncached points each run the radial evaluation, whose cost varies
        // with r, so keep chunk 1 there -- the body dwarfs the hand-out anyway.
        // dynamic (not static) so the loop still rebalances if a core is taken by
        // another process, which costs nothing at this chunk size.
        const int point_chunk = use_r_cache ? 64 : 1;
#pragma omp parallel for schedule(dynamic, point_chunk)
        for (size_t i = 0; i < num_points; i++)
        {
            Ttype sol_val = 0.0;
            const Ttype phi = phi_points[i];

            if (use_r_cache)
            {
                const Ttype *cache_row = &psinm_cache[r_index[i] * full_k_size];
                for (const auto &range : eval_ranges)
                    for (size_t k = range.first; k < range.second; k++)
                    {
                        const SeriesData &term_data = full_series_data[k];
                        sol_val += coeff_gnm[k] * cache_row[k] * sn_phi(term_data.n, phi);
                    }
            }
            else
            {
                const Ttype r = r_points[i];
                for (const auto &range : eval_ranges)
                    for (size_t k = range.first; k < range.second; k++)
                    {
                        const SeriesData &term_data = full_series_data[k];
                        sol_val += coeff_gnm[k] * active_radial(term_data, r) * sn_phi(term_data.n, phi);
                    }
            }

            sol_val += active_constant();
            sol_planes[idx][i] = get_unscaled_solution_value(sol_val);
        }
    }

    if (measure_time)
    {
        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "Timing: get_solution_at_planes (shared r,phi) took " << elapsed_s << " s." << std::endl;
    }

    return sol_planes;
}

template <typename Ttype>
/**
 * Evaluate the solution at multiple planes. The i-th plane has x value x_points[i],
 * and each plane has different (r, phi) points given by r_points[i] and phi_points[i].
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values at the given points for each value of x.
 */
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_solution_at_planes(const std::vector<Ttype> &x_points, const std::vector<std::vector<Ttype>> &r_points, const std::vector<std::vector<Ttype>> &phi_points)
{
    const bool measure_time = (m_verbose_level >= 2);
    const auto t_start = std::chrono::steady_clock::now();

    require_solution_ready("get_solution_at_planes");
    if (m_solution_method == SolutionMethod::QEP)
    {
        if (r_points.size() != x_points.size() || phi_points.size() != x_points.size())
            throw std::invalid_argument("get_solution_at_planes: invalid plane dimensions.");
        std::vector<Ttype> xs, rs, ps;
        std::vector<size_t> offsets;
        offsets.reserve(x_points.size() + 1);
        offsets.push_back(0);
        for (size_t p = 0; p < x_points.size(); ++p)
        {
            if (r_points[p].size() != phi_points[p].size())
                throw std::invalid_argument("get_solution_at_planes: r and phi sizes differ.");
            for (size_t i = 0; i < r_points[p].size(); ++i)
            {
                xs.push_back(x_points[p]);
                rs.push_back(r_points[p][i]);
                ps.push_back(phi_points[p][i]);
            }
            offsets.push_back(xs.size());
        }
        const std::vector<Ttype> values = get_solution_cloud_of_points(xs, rs, ps);
        std::vector<std::vector<Ttype>> out(x_points.size());
        for (size_t p = 0; p < x_points.size(); ++p)
            out[p] = std::vector<Ttype>(values.begin() + offsets[p], values.begin() + offsets[p + 1]);
        return out;
    }
    std::vector<std::vector<Ttype>> sol_planes(x_points.size());
    if (x_points.empty())
    {
        if (measure_time)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "Timing: get_solution_at_planes (per-plane r,phi) took " << elapsed_s << " s." << std::endl;
        }
        return sol_planes;
    }

    if (r_points.size() != x_points.size() || phi_points.size() != x_points.size())
    {
        std::cerr << "Error: Invalid dimensions! len(x_points) = " << x_points.size()
                  << ", len(r_points) = " << r_points.size()
                  << ", len(phi_points) = " << phi_points.size() << std::endl;
        return sol_planes;
    }

    for (size_t i = 0; i < x_points.size(); i++)
    {
        if (r_points[i].size() != phi_points[i].size())
        {
            std::cerr << "Error: Invalid dimensions for plane " << i
                      << "! len(r_points[i]) = " << r_points[i].size()
                      << ", len(phi_points[i]) = " << phi_points[i].size() << std::endl;
        }
        sol_planes[i].assign(r_points[i].size(), static_cast<Ttype>(0.0));
    }

    std::vector<size_t> order(x_points.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
              { return x_points[a] < x_points[b]; });

    if (!mCoefficientsAreFixed || m_series_data.empty())
    {
        prepare_to_compute_solution(r_points[order[0]], phi_points[order[0]]);
    }

    const size_t full_k = m_max_K;
    // Reference, not a copy: the mode list is fixed for the whole call (only
    // set_series_data / invalidate_solution / setup_* write it, none of which run
    // during an evaluation), and the copy is 80 B per mode (160 B for long double).
    const std::vector<SeriesData> &full_series_data = m_series_data;

    // Precompute x-independent data: build combined r_points and phi_points from all planes
    // along with index mappings for efficient computation
    std::vector<Ttype> all_r_points;
    std::vector<Ttype> all_phi_points;
    std::vector<std::pair<size_t, size_t>> plane_ranges; // plane index -> (start, end) in all_*_points

    size_t current_index = 0;
    for (size_t plane_idx = 0; plane_idx < x_points.size(); plane_idx++)
    {
        size_t start = current_index;
        for (const auto &r : r_points[plane_idx])
        {
            all_r_points.push_back(r);
        }
        for (const auto &phi : phi_points[plane_idx])
        {
            all_phi_points.push_back(phi);
        }
        current_index += r_points[plane_idx].size();
        plane_ranges.push_back({start, current_index});
    }

    // Compute psinm_cache and r_index for the combined points (same for all x-planes)
    bool use_r_cache = false;
    std::vector<size_t> r_index(all_r_points.size(), 0);
    std::vector<Ttype> unique_r;
    std::vector<Ttype> psinm_cache;
    const size_t full_k_size = static_cast<size_t>(full_k);

    if (!all_r_points.empty() && full_k > 0)
    {
        const Ttype max_r = *std::max_element(all_r_points.begin(), all_r_points.end());
        const Ttype r_tol = std::max(static_cast<Ttype>(1e-12), static_cast<Ttype>(1e-8) * std::max(max_r, static_cast<Ttype>(1.0)));

        std::vector<unsigned> order_r(all_r_points.size());
        std::iota(order_r.begin(), order_r.end(), 0);
        std::sort(order_r.begin(), order_r.end(), [&](unsigned a, unsigned b)
                  { return all_r_points[a] < all_r_points[b]; });

        unique_r.reserve(all_r_points.size());

        bool grouping_ok = true;
        unsigned start = 0;
        while (start < all_r_points.size())
        {
            const unsigned idx = order_r[start];
            const Ttype r0 = all_r_points[idx];
            Ttype r_min = r0;
            Ttype r_max = r0;
            unsigned end = start + 1;
            while (end < all_r_points.size())
            {
                const Ttype r_val = all_r_points[order_r[end]];
                if (r_val - r0 > r_tol)
                    break;
                r_min = std::min(r_min, r_val);
                r_max = std::max(r_max, r_val);
                end++;
            }

            if ((r_max - r_min) > r_tol)
            {
                grouping_ok = false;
                break;
            }

            const size_t group_index = unique_r.size();
            unique_r.push_back((r_min + r_max) * static_cast<Ttype>(0.5));
            for (unsigned j = start; j < end; j++)
            {
                r_index[order_r[j]] = group_index;
            }
            start = end;
        }

        const size_t max_cache_bytes = static_cast<size_t>(64) * 1024 * 1024;
        const size_t cache_bytes = grouping_ok ? unique_r.size() * full_k_size * sizeof(Ttype) : 0;
        use_r_cache = grouping_ok && unique_r.size() < all_r_points.size() && cache_bytes > 0 && cache_bytes <= max_cache_bytes;

        if (use_r_cache)
        {
            psinm_cache.resize(unique_r.size() * full_k_size, static_cast<Ttype>(0.0));

            // Parallelise over the flattened (unique radius, mode) index rather than
            // over radii alone: a table has only as many rows as there are distinct
            // radii, which is fewer work items than threads on a coarse radial grid,
            // and the per-radius cost varies (the confluent-hypergeometric evaluation
            // switches method with z = b r^2), so row-level hand-out cannot balance it.
            const size_t cache_items = unique_r.size() * full_k_size;
            const int cache_chunk = radial_table_chunk(cache_items);
#pragma omp parallel for schedule(dynamic, cache_chunk)
            for (size_t idx = 0; idx < cache_items; idx++)
            {
                const size_t u = idx / full_k_size;
                const size_t k = idx - u * full_k_size;
                psinm_cache[idx] = active_radial(full_series_data[k], unique_r[u]);
            }
        }
    }

    // Per-mode max |R_k| over the table's radii: the scale cutoff_modes compares
    // against. Built once per call (the table is x-independent); empty when the
    // cut is off or no table exists, which build_eval_ranges reads as "keep all".
    std::vector<Ttype> max_radial;
    if (use_r_cache && m_eval_cutoff_tol > static_cast<Ttype>(0))
        radial_row_maxima(psinm_cache, full_k_size, max_radial);

    // Loop over x-planes and compute solutions
    for (size_t pos = 0; pos < order.size(); pos++)
    {
        const size_t idx = order[pos];
        const Ttype x_val = x_points[idx];

        // x-dependent coefficient of each mode at this plane.
        std::vector<Ttype> coeff_gnm(full_k_size, 0.0);
        // Uniform cost per mode (one exp), so a static split has no imbalance to
        // correct and avoids a hand-out that costs more than the body itself.
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < full_k_size; k++)
        {
            const SeriesData &term_data = full_series_data[k];
            coeff_gnm[k] = active_coeff(term_data) * std::exp(-active_rate(term_data) * x_val);
        }

        // Modes worth summing at this plane (one prefix per angular block); the
        // single full range when the cut is off, so the loop below is unchanged.
        std::vector<std::pair<size_t, size_t>> eval_ranges;
        build_eval_ranges(coeff_gnm, max_radial, full_k_size, eval_ranges);

        const auto &[start_idx, end_idx] = plane_ranges[idx];
        const size_t num_points_in_plane = end_idx - start_idx;

        // Cached points all cost the same (K multiply-adds), so hand them out in
        // blocks; uncached points each run the radial evaluation, whose cost varies
        // with r, so keep chunk 1 there -- the body dwarfs the hand-out anyway.
        // dynamic (not static) so the loop still rebalances if a core is taken by
        // another process, which costs nothing at this chunk size.
        const int point_chunk = use_r_cache ? 64 : 1;
#pragma omp parallel for schedule(dynamic, point_chunk)
        for (size_t i = 0; i < num_points_in_plane; i++)
        {
            const size_t global_idx = start_idx + i;
            Ttype sol_val = 0.0;
            const Ttype phi = all_phi_points[global_idx];

            if (use_r_cache)
            {
                const Ttype *cache_row = &psinm_cache[r_index[global_idx] * full_k_size];
                for (const auto &range : eval_ranges)
                    for (size_t k = range.first; k < range.second; k++)
                    {
                        const SeriesData &term_data = full_series_data[k];
                        sol_val += coeff_gnm[k] * cache_row[k] * sn_phi(term_data.n, phi);
                    }
            }
            else
            {
                const Ttype r = all_r_points[global_idx];
                for (const auto &range : eval_ranges)
                    for (size_t k = range.first; k < range.second; k++)
                    {
                        const SeriesData &term_data = full_series_data[k];
                        sol_val += coeff_gnm[k] * active_radial(term_data, r) * sn_phi(term_data.n, phi);
                    }
            }

            sol_val += active_constant();
            sol_planes[idx][i] = get_unscaled_solution_value(sol_val);
        }
    }

    if (measure_time)
    {
        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "Timing: get_solution_at_planes (per-plane r,phi) took " << elapsed_s << " s." << std::endl;
    }

    return sol_planes;
}

template <typename Ttype>
/**
 * Evaluate the solution at multiple points for a fixed x.
 * @param x_point Axial coordinate.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values at the given points.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution_at_points(Ttype x_point, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    const bool measure_time = (m_verbose_level >= 2);
    const auto t_start = std::chrono::steady_clock::now();

    require_solution_ready("get_solution_at_points");
    if (m_solution_method == SolutionMethod::QEP)
    {
        std::vector<Ttype> xs(r_points.size(), x_point), sol;
        qep_evaluate_at_points<Ttype>(m_qep_blocks, xs, r_points, phi_points, sol);
        for (auto &value : sol)
            value = get_unscaled_solution_value(value);
        return sol;
    }
    std::vector<Ttype> sol(r_points.size(), 0.0);

    prepare_to_compute_solution(r_points, phi_points);
    compute_solution(x_point, r_points, phi_points, sol);

    if (measure_time)
    {
        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "Timing: get_solution_at_points took " << elapsed_s << " s." << std::endl;
    }

    return sol;
}

template <typename Ttype>
/**
 * Evaluate the solution at multiple x values (not implemented).
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values at the given points.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution_at_points(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    std::cerr << "Method <<get_solution_at_points(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)>> not implemented yet." << std::endl;

    std::vector<Ttype> sol(r_points.size(), 0.0);
    return sol;
}

template <typename Ttype>
/**
 * Evaluate the solution at a single point.
 * @param x Axial coordinate.
 * @param r Radial coordinate.
 * @param phi Angular coordinate.
 * @return Solution value.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_solution_at_points(Ttype x, Ttype r, Ttype phi)
{
    const std::vector<Ttype> r_vec = {r}, phi_vec = {phi};
    std::vector<Ttype> sol = get_solution_at_points(x, r_vec, phi_vec);
    return sol;
}

template <typename Ttype>
/**
 * Evaluate the solution at a single point in-place.
 * @param x Axial coordinate.
 * @param r Radial coordinate.
 * @param phi Angular coordinate.
 * @param sol Output solution value.
 */
void CDBaseSolution<Ttype>::get_solution_at_point(const Ttype &x, const Ttype &r, const Ttype &phi, Ttype &sol)
{
    require_solution_ready("get_solution_at_point");
    if (m_solution_method == SolutionMethod::QEP)
    {
        std::vector<Ttype> raw;
        qep_evaluate_at_points<Ttype>(m_qep_blocks, std::vector<Ttype>{x}, std::vector<Ttype>{r}, std::vector<Ttype>{phi}, raw);
        sol = get_unscaled_solution_value(raw.front());
        return;
    }
    sol = 0.0;
    for (unsigned k = 0; k < m_max_K; k++)
    {
        SeriesData &term_data = m_series_data[k];
        sol += active_coeff(term_data) * std::exp(-active_rate(term_data) * x) * active_radial(term_data, r) * sn_phi(term_data.n, phi);
    }

    // Unscale the solution
    sol += active_constant();
    sol = get_unscaled_solution_value(sol);
}

template <typename Ttype>
/**
 * Compute solution values for per-point x, r, and phi arrays.
 * @param x_points Axial coordinates.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @param sol Output solution values.
 */
void CDBaseSolution<Ttype>::compute_solution(const std::vector<Ttype> &x_points, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points, std::vector<Ttype> &sol)
{
    const unsigned num_points = phi_points.size();

    for (unsigned i = 0; i < num_points; i++)
    {
        get_solution_at_point(x_points[i], r_points[i], phi_points[i], sol[i]);
    }
}

template <typename Ttype>
/**
 * Compute solution values for a fixed x and arrays of r and phi.
 * @param x_point Axial coordinate.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 * @param sol Output solution values.
 */
void CDBaseSolution<Ttype>::compute_solution(Ttype x_point, const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points, std::vector<Ttype> &sol)
{
    const unsigned num_points = phi_points.size();

    std::vector<Ttype> coeff_gnm(m_max_K, 0.0);
    // Uniform cost per mode (one exp): see the note in get_solution_at_planes.
#pragma omp parallel for schedule(static)
    for (unsigned k = 0; k < m_max_K; k++)
    {
        const SeriesData &term_data = m_series_data[k];
        coeff_gnm[k] = active_coeff(term_data) * std::exp(-active_rate(term_data) * x_point);
    }

    bool use_r_cache = false;
    std::vector<size_t> r_index;
    std::vector<Ttype> unique_r;
    std::vector<Ttype> psinm_cache;

    if (num_points > 0 && m_max_K > 0)
    {
        const Ttype max_r = *std::max_element(r_points.begin(), r_points.end());
        const Ttype r_tol = std::max(static_cast<Ttype>(1e-12), static_cast<Ttype>(1e-8) * std::max(max_r, static_cast<Ttype>(1.0)));

        std::vector<unsigned> order(num_points);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b)
                  { return r_points[a] < r_points[b]; });

        r_index.assign(num_points, 0);
        unique_r.reserve(num_points);

        bool grouping_ok = true;
        unsigned start = 0;
        while (start < num_points)
        {
            const unsigned idx = order[start];
            const Ttype r0 = r_points[idx];
            Ttype r_min = r0;
            Ttype r_max = r0;
            unsigned end = start + 1;
            while (end < num_points)
            {
                const Ttype r_val = r_points[order[end]];
                if (r_val - r0 > r_tol)
                    break;
                r_min = std::min(r_min, r_val);
                r_max = std::max(r_max, r_val);
                end++;
            }

            if ((r_max - r_min) > r_tol)
            {
                grouping_ok = false;
                break;
            }

            const size_t group_index = unique_r.size();
            unique_r.push_back((r_min + r_max) * static_cast<Ttype>(0.5));
            for (unsigned j = start; j < end; j++)
            {
                r_index[order[j]] = group_index;
            }
            start = end;
        }

        const size_t max_cache_bytes = static_cast<size_t>(64) * 1024 * 1024;
        const size_t cache_bytes = grouping_ok ? unique_r.size() * m_max_K * sizeof(Ttype) : 0;
        use_r_cache = grouping_ok && unique_r.size() < num_points && cache_bytes > 0 && cache_bytes <= max_cache_bytes;

        if (use_r_cache)
        {
            psinm_cache.resize(unique_r.size() * m_max_K, static_cast<Ttype>(0.0));

            // Parallelise over the flattened (unique radius, mode) index rather than
            // over radii alone: a table has only as many rows as there are distinct
            // radii, which is fewer work items than threads on a coarse radial grid,
            // and the per-radius cost varies (the confluent-hypergeometric evaluation
            // switches method with z = b r^2), so row-level hand-out cannot balance it.
            const size_t cache_items = unique_r.size() * static_cast<size_t>(m_max_K);
            const int cache_chunk = radial_table_chunk(cache_items);
#pragma omp parallel for schedule(dynamic, cache_chunk)
            for (size_t idx = 0; idx < cache_items; idx++)
            {
                const size_t u = idx / static_cast<size_t>(m_max_K);
                const size_t k = idx - u * static_cast<size_t>(m_max_K);
                psinm_cache[idx] = active_radial(m_series_data[k], unique_r[u]);
            }
        }
    }

    // Per-mode max |R_k| over the table's radii: the scale cutoff_modes compares
    // against. Built once per call (the table is x-independent); empty when the
    // cut is off or no table exists, which build_eval_ranges reads as "keep all".
    std::vector<Ttype> max_radial;
    if (use_r_cache && m_eval_cutoff_tol > static_cast<Ttype>(0))
        radial_row_maxima(psinm_cache, static_cast<size_t>(m_max_K), max_radial);

    // Modes worth summing at this plane; the single full range when the cut is off.
    std::vector<std::pair<size_t, size_t>> eval_ranges;
    build_eval_ranges(coeff_gnm, max_radial, static_cast<size_t>(m_max_K), eval_ranges);

    // See the note in get_solution_at_planes on the chunk choice.
    const int point_chunk = use_r_cache ? 64 : 1;
#pragma omp parallel for schedule(dynamic, point_chunk)
    for (unsigned i = 0; i < num_points; i++)
    {
        Ttype sol_val = 0.0;
        const Ttype r = r_points[i];
        const Ttype phi = phi_points[i];
        if (use_r_cache)
        {
            const Ttype *cache_row = &psinm_cache[r_index[i] * m_max_K];
            for (const auto &range : eval_ranges)
                for (size_t k = range.first; k < range.second; k++)
                {
                    const SeriesData &term_data = m_series_data[k];
                    sol_val += coeff_gnm[k] * cache_row[k] * sn_phi(term_data.n, phi);
                }
        }
        else
        {
            for (const auto &range : eval_ranges)
                for (size_t k = range.first; k < range.second; k++)
                {
                    const SeriesData &term_data = m_series_data[k];
                    sol_val += coeff_gnm[k] * active_radial(term_data, r) * sn_phi(term_data.n, phi);
                }
        }

        sol_val += active_constant();
        sol[i] = get_unscaled_solution_value(sol_val);
    }
}

template <typename Ttype>
/**
 * Prepare coefficient data for evaluation: build the series and compute the
 * coefficients if they are not already fixed.
 * @param r_points Radial coordinates.
 * @param phi_points Angular coordinates.
 */
void CDBaseSolution<Ttype>::prepare_to_compute_solution(const std::vector<Ttype> &r_points, const std::vector<Ttype> &phi_points)
{
    // Check dimensions for r and phi
    if (r_points.size() != phi_points.size())
    {
        std::cerr << "Error: Invalid dimensions! len(r_points) = " << r_points.size() << ", len(phi_points) = " << phi_points.size() << std::endl;
    }

    require_solution_ready("evaluation");

    if (m_verbose_level >= 2)
        std::cout << "Solution initialized." << std::endl;
}

template <typename Ttype>
/**
 * Get roots table from disk.
 * @return Roots table.
 */
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_roots() const
{
    require_solution_ready("get_roots");
    unsigned max_n = 0;
    for (const auto &term : m_series_data)
        max_n = std::max(max_n, term.n);
    std::vector<std::vector<Ttype>> roots(max_n + 1);
    for (const auto &term : m_series_data)
    {
        auto &row = roots[term.n];
        if (row.size() <= term.m)
            row.resize(term.m + 1);
        row[term.m] = term.root;
    }
    return roots;
}

template <typename Ttype>
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_bare_root_catalog() const
{
    std::vector<std::vector<Ttype>> roots;
    set_roots(roots);
    return roots;
}

template <typename Ttype>
/**
 * Get norms table from disk.
 * @return Norms table.
 */
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_norms() const
{
    std::vector<std::vector<Ttype>> norms;
    set_norms(norms);
    return norms;
}

template <typename Ttype>
/**
 * Get computed coefficients as a matrix.
 * @return Coefficients matrix.
 */
std::vector<std::vector<Ttype>> CDBaseSolution<Ttype>::get_coefficients() const
{
    require_solution_ready("get_coefficients");
    unsigned max_n = 0;
    for (const auto &term : m_series_data)
        max_n = std::max(max_n, term.n);
    std::vector<std::vector<Ttype>> coefficients(max_n + 1);
    for (const auto &term : m_series_data)
    {
        auto &row = coefficients[term.n];
        if (row.size() <= term.m)
            row.resize(term.m + 1);
        row[term.m] = term.coeff;
    }
    return coefficients;
}

template <typename Ttype>
/**
 * Get the number of computed coefficients.
 * @return Number of coefficients.
 */
unsigned CDBaseSolution<Ttype>::get_number_of_coefficients() const
{
    require_solution_ready("get_number_of_coefficients");
    return m_series_data.size();
}

template <typename Ttype>
/**
 * Lock coefficients to avoid recomputation.
 */
void CDBaseSolution<Ttype>::fix_coefficients()
{
    if (!mCoefficientsAreComputed)
    {
        std::stringstream err_msg("To fix the coefficients the solution must be initialized.");
        throw std::runtime_error(err_msg.str());
    }

    mCoefficientsAreFixed = true;
}

template <typename Ttype>
/**
 * Compute the coefficients and lock them. Truncation is governed by max_root,
 * set beforehand via set_max_root(); the default keeps all tabulated modes.
 */
void CDBaseSolution<Ttype>::compute_and_fix_coefficients()
{
    free_coefficients();
    set_series_data();

    const bool measure_time = (m_verbose_level >= 2);
    const auto t_start = std::chrono::steady_clock::now();
    compute_coefficients();

    if (measure_time)
    {
        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "Timing: compute_coefficients took " << elapsed_s << " s." << std::endl;
    }

    fix_coefficients();
}

template <typename Ttype>
/**
 * Unlock coefficients so they can be recomputed.
 */
void CDBaseSolution<Ttype>::free_coefficients()
{
    mCoefficientsAreFixed = false;
}

template <typename Ttype>
/**
 * Default finite-Péclet coefficient hook. A subclass that supports the
 * axial-diffusion path must override this to fill coeff_fp (via the
 * finite_peclet_coefficients free functions). The base version fails loudly.
 */
void CDBaseSolution<Ttype>::compute_coefficients_fp(const FPRadialTable<Ttype> &)
{
    throw std::runtime_error(
        "compute_coefficients_fp() not implemented for this solution type. "
        "Either override it or select the bare solution with peclet = +inf.");
}

template <typename Ttype>
/**
 * Initialize the solution for a given Péclet number and truncation cutoff.
 *
 * This is the single setup entry point (separate from evaluation). It builds the
 * retained mode set, then prepares the per-mode spectral data by delegating the
 * three heavy computations to the finite_peclet_* free functions:
 *   (i)   fp_modify_roots  — decay rates Λ_nm, b_nm, b̃_nm (Newton root finding),
 *   (ii)  fp_compute_norms_boundary — generalised norms 𝒩²_fp,
 *   (iii) compute_coefficients_fp — modified inlet coefficients Ĉ_nm.
 * All complexity lives in those functions, not here.
 *
 * @param peclet   finite > 0 selects the exact axial-diffusion solution (default
 *                 behaviour); +inf selects the bare (axial-diffusion-free) solution.
 * @param max_root series truncation cutoff (snapped to a tabulated root).
 */
void CDBaseSolution<Ttype>::setup_solution(Ttype peclet, Ttype max_root)
{
    set_max_root(max_root);
    setup_fp_solution(peclet);
}

template <typename Ttype>
/**
 * Initialize the solution for a given Péclet number, reusing the current
 * truncation cutoff. See the two-argument overload for the full description.
 */
void CDBaseSolution<Ttype>::setup_solution(Ttype peclet)
{
    setup_fp_solution(peclet);
}

template <typename Ttype>
void CDBaseSolution<Ttype>::setup_bare_solution()
{
    invalidate_solution();
    set_series_data();
    try
    {
        compute_coefficients();
        for (auto &term : m_series_data)
            if (is_zero_eigenvalue_mode(term.n, term.m))
                term.coeff = static_cast<Ttype>(std::sqrt(2.0 * M_PI)) * m_constant_term;
        m_inlet_projection_square_norm = diagonal_inlet_projection_square_norm();
        m_peclet = std::numeric_limits<Ttype>::infinity();
        m_kappa = static_cast<Ttype>(0);
        mCoefficientsAreComputed = mCoefficientsAreFixed = true;
        m_solution_method = SolutionMethod::Bare;
    }
    catch (...)
    {
        invalidate_solution();
        throw;
    }
}

template <typename Ttype>
void CDBaseSolution<Ttype>::setup_fp_solution(Ttype peclet)
{
    if (!(peclet > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(peclet)))
        throw std::invalid_argument("setup_fp_solution: peclet must be finite and positive.");
    invalidate_solution();
    // No-fallback policy (projection_space.h): the L2_r projection has no
    // ultraspherical Gram backend -- that factor is built for the (1,n) measure
    // and does not exist for L2_r without a fresh derivation. Rejected loudly
    // HERE, before any pass runs, rather than by silently substituting a backend
    // the user did not select. After invalidate_solution above, so a rejected
    // configuration leaves no active solution behind -- the same post-condition
    // the catch block below establishes.
    //
    // BOTH RhsMethod values are supported on the radial path
    // (finite_peclet_rhs_representer_radial.h); the representer restriction that
    // used to live here is gone.
    if (m_projection_space == ProjectionSpace::Radial && m_gram_method != GramMethod::GaussJacobiQR)
        throw std::invalid_argument("setup_fp_solution: ProjectionSpace::Radial has no ultraspherical "
                                    "Gram backend (that factor is built for the (1,n) measure); select "
                                    "GramMethod::GaussJacobiQR explicitly");
    set_series_data();
    try
    {
        m_peclet = peclet;
        m_kappa = static_cast<Ttype>(1) / (peclet * peclet);
        m_use_axial_diffusion = true;
        // Loaded once and cached: the seed ratio alpha_nm = U^n_mm / N_nm^2
        // (eq. seed_root) is kappa-independent, so a Peclet sweep reuses it.
        if (m_pseudo_products_matrix.empty())
            set_pseudo_products_matrix(m_pseudo_products_matrix);
        // (i) exact decay rates + shifted-Kummer data (independent Newton per mode)
        const bool measure_fp_timing = (m_verbose_level >= 3);
        const auto t_roots_start = measure_fp_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        fp_modify_roots<Ttype>(m_series_data, m_max_K, m_kappa, wall_condition(),
                               m_fp_rel_tol, m_fp_max_iter, m_pseudo_products_matrix);
        if (measure_fp_timing)
            std::cout << "Timing: finite-Peclet roots modification took "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_roots_start).count()
                      << " s." << std::endl;
        // Compatibility adapter: compute_coefficients_fp historically accepts
        // an FPRadialTable.  Gaussian implementations ignore this empty,
        // immutable object, preserving the established virtual interface.
        const FPRadialTable<Ttype> unused_legacy_table;
        compute_coefficients_fp(unused_legacy_table);
        for (auto &term : m_series_data)
        {
            term.root = term.root_fp;
            term.coeff = term.coeff_fp;
        }
        mCoefficientsAreComputed = true;
        mCoefficientsAreFixed = true;
        m_solution_method = SolutionMethod::ModifiedRoots;
    }
    catch (...)
    {
        invalidate_solution();
        throw;
    }
}

template <typename Ttype>
void CDBaseSolution<Ttype>::setup_qep_solution(Ttype peclet)
{
    if (!(peclet > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(peclet)))
        throw std::invalid_argument("setup_qep_solution: peclet must be finite and positive.");
    invalidate_solution();
    set_series_data();
    try
    {
        compute_coefficients();
        // qep_build_and_solve expects the historical separated zero-mode inlet.
        for (auto &term : m_series_data)
            if (is_zero_eigenvalue_mode(term.n, term.m))
                term.coeff = static_cast<Ttype>(0);
        const Ttype kappa = static_cast<Ttype>(1) / (peclet * peclet);
        qep_build_and_solve<Ttype>(m_series_data, m_max_K, kappa, m_constant_term,
                                   m_qep_blocks, m_fp_quad_panels);
        for (auto &term : m_series_data)
            if (is_zero_eigenvalue_mode(term.n, term.m))
                term.coeff = static_cast<Ttype>(std::sqrt(2.0 * M_PI)) * m_constant_term;
        m_inlet_projection_square_norm = diagonal_inlet_projection_square_norm();
        m_qep_kappa = kappa;
        m_peclet = peclet;
        m_kappa = kappa;
        mCoefficientsAreComputed = mCoefficientsAreFixed = true;
        m_solution_method = SolutionMethod::QEP;
    }
    catch (...)
    {
        invalidate_solution();
        throw;
    }
}

template <typename Ttype>
/**
 * Evaluate the exact finite-Péclet solution obtained from the Quadratic
 * Eigenvalue Problem in the bare eigenbasis (Neuhauser et al. 2025, Sec. 2.4).
 *
 * All the mathematics — overlap assembly, the symmetric-definite linearisation,
 * the eigen-decomposition, the inlet solve, and the series reconstruction — lives
 * in qep_computations.{h,cpp}. This method only: (i) prepares the bare mode set
 * and inlet coefficients, (ii) calls the two QEP entry points, (iii) adds the
 * constant term and un-scales.
 *
 * The spectral data is cached per κ, so repeated calls at the same Péclet reuse
 * the eigen-decomposition.
 *
 * @param peclet     Péclet number (finite, strictly positive).
 * @param x_points   Axial coordinates.
 * @param r_points   Radial coordinates.
 * @param phi_points Angular coordinates.
 * @return Solution values in the input order.
 */
std::vector<Ttype> CDBaseSolution<Ttype>::get_QEP_solution_at_points(
    Ttype peclet,
    const std::vector<Ttype> &x_points,
    const std::vector<Ttype> &r_points,
    const std::vector<Ttype> &phi_points)
{
    if (!(peclet > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(peclet)))
        throw std::invalid_argument("get_QEP_solution_at_points: peclet must be finite and positive.");

    const Ttype kappa = static_cast<Ttype>(1) / (peclet * peclet);

    if (m_qep_blocks.empty() || m_qep_kappa != kappa)
    {
        // The QEP is posed in the bare, Pe-independent eigenbasis, so it needs the
        // bare roots/norms and the bare inlet projection. These are the base-path
        // fields, untouched by setup_solution's finite-Péclet data.
        set_series_data();
        compute_coefficients();

        const bool measure_time = (m_verbose_level >= 2);
        const auto t_start = std::chrono::steady_clock::now();

        qep_build_and_solve<Ttype>(m_series_data, m_max_K, kappa, m_constant_term,
                                   m_qep_blocks, m_fp_quad_panels);
        m_qep_kappa = kappa;

        if (measure_time)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "Timing: qep_build_and_solve took " << elapsed_s << " s." << std::endl;
        }
    }

    std::vector<Ttype> sol;
    qep_evaluate_at_points<Ttype>(m_qep_blocks, x_points, r_points, phi_points, sol);

    // The constant mode is inside the blocks now (its inlet value was supplied to
    // qep_build_and_solve), so m_constant_term must NOT be added again here.
    for (size_t i = 0; i < sol.size(); ++i)
        sol[i] = get_unscaled_solution_value(sol[i]);

    return sol;
}

template <typename Ttype>
/**
 * Return the finite-Péclet spectral data (Λ_nm, b_nm, b̃_nm, 𝒩²_fp, Ĉ_nm) per
 * retained mode, in m_series_data order. Useful for inspection/validation.
 */
std::vector<std::array<Ttype, 5>> CDBaseSolution<Ttype>::get_modified_data() const
{
    std::vector<std::array<Ttype, 5>> out;
    out.reserve(m_max_K);
    for (unsigned k = 0; k < m_max_K; ++k)
    {
        const SeriesData &t = m_series_data[k];
        out.push_back({t.rate_fp, t.root_fp, t.btilde_fp, t.norm_fp, t.coeff_fp});
    }
    return out;
}

// Others
template <typename Ttype>
bool CDBaseSolution<Ttype>::has_retained_constant_mode() const
{
    for (unsigned k = 0; k < m_max_K; ++k)
        if (is_zero_eigenvalue_mode(m_series_data[k].n, m_series_data[k].m))
            return true;
    return false;
}

template <typename Ttype>
/**
 * || P1 ||^2, the squared norm of the projected constant function.
 *
 * Default: valid when the constant IS a retained mode, so that P1 = 1 exactly and
 * this is the weighted area of the disk.
 */
Ttype CDBaseSolution<Ttype>::constant_projection_square_norm() const
{
    if (!has_retained_constant_mode())
        throw std::logic_error(
            "constant_projection_square_norm: this eigenbasis has no constant mode, so the projected "
            "constant is not the constant itself and the weighted disk area is the wrong value; "
            "override this hook together with constant_projection_inlet_cross().");
    // int_D (1 - r^2) dA = pi/2, written as a long-double literal so the
    // long-double instantiation does not inherit a double constant.
    return static_cast<Ttype>(1.57079632679489661923132169163975144L);
}

template <typename Ttype>
/**
 * <P1, f~>, the pairing of the projected constant with the internal inlet.
 *
 * Default: with P1 = 1 this is int_D omega f~ dA, which set_constant_term has
 * already computed -- m_constant_term is that integral divided by the weighted
 * disk area.
 */
Ttype CDBaseSolution<Ttype>::constant_projection_inlet_cross() const
{
    if (!has_retained_constant_mode())
        throw std::logic_error(
            "constant_projection_inlet_cross: this eigenbasis has no constant mode, so <P1, f~> is not "
            "the weighted inlet mean; override this hook together with constant_projection_square_norm().");
    return static_cast<Ttype>(1.57079632679489661923132169163975144L) * m_constant_term;
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::diagonal_inlet_projection_square_norm() const
{
    Ttype total = static_cast<Ttype>(0);
    for (unsigned k = 0; k < m_max_K; ++k)
        total += m_series_data[k].norm * m_series_data[k].coeff * m_series_data[k].coeff;
    return total;
}

template <typename Ttype>
/**
 * Squared norm of the inlet projection in the active setup's inner product,
 * for the inlet as the USER defined it. The stored Q is c^T b = c^T G c,
 * with G equal to the weighted or radial Gram matrix selected at setup.
 *
 * The solver projects the internally scaled profile f~ that set_ui built, so the
 * stored contraction belongs to f~. Writing the inverse of
 * get_unscaled_solution_value as f = alpha + beta f~, with alpha = m_max_ui and
 * beta = -(m_max_ui - m_min_ui), linearity of the projection gives
 * f_K = alpha P1 + beta f~_K and hence
 *
 *   || f_K ||^2 = alpha^2 || P1 ||^2 + 2 alpha beta <P1, f~> + beta^2 Q ,
 *
 * with Q the stored contraction. In the radial Neumann case P1 = 1 and the
 * cross term is the exact area integral of the layered inlet. In the weighted
 * case the existing constant-projection hooks supply both terms. No numerical
 * quadrature is repeated here.
 *
 * Evaluated as S (alpha + beta X/S)^2 + beta^2 (Q - X^2/S), with S = || P1 ||^2
 * and X = <P1, f~>: the same number as the three-term expansion, completed
 * around the mean in the active space so that both terms are non-negative (the second by
 * Cauchy-Schwarz), where the expansion itself differences quantities larger than
 * the result.
 */
Ttype CDBaseSolution<Ttype>::get_inlet_projection_square_norm() const
{
    require_solution_ready("get_inlet_projection_square_norm");
    if (!std::isfinite(static_cast<long double>(m_inlet_projection_square_norm)))
        throw std::logic_error(
            "get_inlet_projection_square_norm: the active solution path produced no inlet projection norm.");

    Ttype constant_square_norm;
    Ttype constant_inlet_cross;
    if (m_solution_method == SolutionMethod::ModifiedRoots && m_projection_space == ProjectionSpace::Radial
        && has_retained_constant_mode())
    {
        // The Neumann basis contains 1, so P_r 1 = 1 and the cross term is
        // the exact area integral of the internally scaled, layered inlet.
        const Ttype pi = std::acos(static_cast<Ttype>(-1));
        const auto cap_area = [](Ttype z) {
            return std::acos(z) - z * std::sqrt(std::max(static_cast<Ttype>(0),
                                                        static_cast<Ttype>(1) - z * z));
        };
        constant_square_norm = pi;
        constant_inlet_cross = static_cast<Ttype>(0);
        Ttype lower = static_cast<Ttype>(-1);
        for (std::size_t i = 0; i < m_ui.size(); ++i)
        {
            const Ttype upper = i < m_zi.size() ? m_zi[i] : static_cast<Ttype>(1);
            constant_inlet_cross += m_ui[i] * (cap_area(lower) - cap_area(upper));
            lower = upper;
        }
    }
    else
    {
        // Weighted setup, or the Graetz Dirichlet basis, whose internal inlet
        // equals 1 and whose constant projection is the stored contraction.
        constant_square_norm = constant_projection_square_norm();
        constant_inlet_cross = constant_projection_inlet_cross();
    }
    if (!(constant_square_norm > static_cast<Ttype>(0))
        || !std::isfinite(static_cast<long double>(constant_square_norm))
        || !std::isfinite(static_cast<long double>(constant_inlet_cross)))
        throw std::runtime_error("get_inlet_projection_square_norm: invalid constant-projection data.");

    const Ttype scaled_mean = constant_inlet_cross / constant_square_norm;
    const Ttype amplitude = -(m_max_ui - m_min_ui);
    // get_unscaled_solution_value(scaled_mean), inlined because that method is
    // not const: the flow-weighted mean of the inlet in the user's units.
    const Ttype physical_mean = m_max_ui + amplitude * scaled_mean;
    const Ttype fluctuation = m_inlet_projection_square_norm - scaled_mean * constant_inlet_cross;
    return constant_square_norm * physical_mean * physical_mean + amplitude * amplitude * fluctuation;
}

template <typename Ttype>
/**
 * Emit a warning message with rate limiting.
 * @param s Warning text.
 */
void CDBaseSolution<Ttype>::warn(std::string s)
{
    if (m_current_warning <= m_max_warnings)
    {
        std::cout << "Warning: " << s << std::endl;

        if (m_current_warning == m_max_warnings)
        {
            std::cout << "(Max warnings have been achieved. They will be omitted from now on)" << std::endl;
        }
        m_current_warning++;
    }
}

template <typename Ttype>
/**
 * Save coefficients to a text file.
 * @param filename Output file path.
 */
void CDBaseSolution<Ttype>::save_coefficients(const std::string &filename)
{
    if (!mCoefficientsAreComputed)
    {
        std::stringstream err_msg("To save the coefficients the solution must be initialized.");
        throw std::runtime_error(err_msg.str());
    }

    std::vector<std::vector<Ttype>> coefficients = get_coefficients();
    writeValuesToFile(filename, coefficients);
}

// Nusselt-number post-processing of the retained isothermal series.
template <typename Ttype>
void CDBaseSolution<Ttype>::require_nusselt_number_defined(const char *caller) const
{
    if (wall_condition() != WallCondition::Dirichlet)
        throw std::logic_error(std::string(caller)
            + ": the Nusselt number requires an isothermal (Dirichlet) wall.");
}

template <typename Ttype>
std::vector<std::array<Ttype, 5>>
CDBaseSolution<Ttype>::nusselt_axisymmetric_mode_data(const char *caller) const
{
    require_nusselt_number_defined(caller);
    require_solution_ready(caller);
    if (m_solution_method == SolutionMethod::QEP)
        throw std::logic_error(std::string(caller)
            + ": unavailable on the QEP path; use setup_fp_solution or setup_bare_solution.");

    const auto bare_roots = get_bare_root_catalog();
    std::vector<SeriesData> block;
    std::vector<std::size_t> flat_index;
    for (std::size_t k = 0; k < m_series_data.size(); ++k)
    {
        const SeriesData &t = m_series_data[k];
        if (t.n != 0u)
            continue;
        SeriesData copy = t;
        copy.root = bare_roots.at(0).at(t.m); // original root sizes the RHS quadrature
        copy.root_fp = t.root;
        copy.btilde_fp = t.root * (static_cast<Ttype>(1) + m_kappa * t.root * t.root);
        copy.rate_fp = t.root * t.root;
        block.push_back(copy);
        flat_index.push_back(k);
    }
    if (block.empty())
        throw std::runtime_error(std::string(caller) + ": no axisymmetric mode retained.");

    const auto load = fp_rhs_uniform_inlet<Ttype>(0u, block, m_fp_cap_quad_margin);
    const Ttype sqrt_two_pi = static_cast<Ttype>(2.50662827463100050241576528481104525L);
    std::vector<std::array<Ttype, 5>> out;
    out.reserve(block.size());
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        const SeriesData &t = m_series_data[flat_index[i]];
        const Ttype sigma = active_radial_derivative_at_wall(t);
        const Ttype varpi = load.at(i) / sqrt_two_pi;
        if (!std::isfinite(static_cast<long double>(sigma)) ||
            !std::isfinite(static_cast<long double>(varpi)))
            throw std::runtime_error(std::string(caller) + ": non-finite wall slope or bulk moment.");
        out.push_back({static_cast<Ttype>(t.m), active_rate(t), active_coeff(t), sigma, varpi});
    }
    return out;
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::get_nusselt_number(const std::vector<Ttype> &x_points) const
{
    return get_nusselt_number(x_points, NusseltEvaluation::Plain);
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::get_nusselt_number(const std::vector<Ttype> &x_points,
                                                             NusseltEvaluation evaluation,
                                                             const std::vector<Ttype> *extra_rates) const
{
    const auto modes = nusselt_axisymmetric_mode_data("get_nusselt_number");
    for (const Ttype x : x_points)
        if (!(x >= static_cast<Ttype>(0)))
            throw std::invalid_argument("get_nusselt_number: x must be >= 0.");

    std::size_t slow = 0;
    for (std::size_t i = 1; i < modes.size(); ++i)
        if (modes[i][1] < modes[slow][1])
            slow = i;
    const Ttype rate_min = modes[slow][1];

    std::vector<Ttype> tail_rates;
    const bool split = evaluation != NusseltEvaluation::Plain;
    const bool asymptotic = evaluation == NusseltEvaluation::AsymptoticSplit;
    if (extra_rates && evaluation != NusseltEvaluation::SpectralSplit)
        throw std::invalid_argument("get_nusselt_number: extra_rates are accepted only with SpectralSplit.");
    if (split)
    {
        if (m_solution_method != SolutionMethod::ModifiedRoots)
            throw std::logic_error("get_nusselt_number: the split evaluations require the finite-Peclet "
                                   "solution (setup_fp_solution).");
        Ttype x_min = std::numeric_limits<Ttype>::infinity();
        for (const Ttype x : x_points)
        {
            if (!(x > static_cast<Ttype>(0)))
                throw std::invalid_argument("get_nusselt_number: the split evaluations need x > 0 "
                                            "(the wall gradient is unbounded at the inlet).");
            x_min = std::min(x_min, x);
        }
        if (extra_rates)
        {
            // A caller table must continue the retained block and resolve the tail at x_min.
            const Ttype last_retained = modes.back()[1];
            const std::vector<Ttype> &rates = *extra_rates;
            for (std::size_t i = 0; i < rates.size(); ++i)
                if (!(rates[i] > (i == 0 ? last_retained : rates[i - 1])))
                    throw std::invalid_argument("get_nusselt_number: extra_rates must be ascending and "
                                                "above the retained rates.");
            if (std::isfinite(static_cast<long double>(x_min)))
            {
                const Ttype last = rates.empty() ? last_retained : rates.back();
                const Ttype before = rates.size() >= 2 ? rates[rates.size() - 2]
                    : (rates.size() == 1 ? last_retained
                                         : (modes.size() >= 2 ? modes[modes.size() - 2][1] : static_cast<Ttype>(0)));
                const Ttype spacing = last - before;
                const Ttype remainder = std::exp(-(last + spacing - rate_min) * x_min) / -std::expm1(-spacing * x_min);
                if (!(remainder <= 16 * std::numeric_limits<Ttype>::epsilon() *
                                   (static_cast<Ttype>(1) + spectral_function_sum(rates, x_min, rate_min))))
                    throw std::invalid_argument("get_nusselt_number: extra_rates do not resolve the tail at the "
                                                "smallest x; use get_nusselt_tail_rates(x_min).");
            }
            tail_rates = rates;
        }
        else if (std::isfinite(static_cast<long double>(x_min)) && !asymptotic)
            tail_rates = nusselt_tail_rates(modes, x_min, evaluation, "get_nusselt_number");
    }
    const Ttype two_sqrt_two_pi = static_cast<Ttype>(5.01325654926200100483153056962209050L);

    // AsymptoticSplit: the rates Lam^inf_m of eq. rate_inf, paired with each
    // retained mode for G_0 and for m = 1..M in the sum of G_inf. From x_plain on,
    // the omitted modes are below round-off relative to the slowest retained one,
    // and G_0 + G_inf reduces to the retained part of G.
    std::vector<Ttype> retained_rates, paired_mode_rates, sum_mode_rates;
    Ttype x_plain = std::numeric_limits<Ttype>::infinity();
    if (asymptotic)
    {
        const unsigned sum_terms = std::max(m_nusselt_asymptotic_terms,
                                            asymptotic_progression_non_positive_terms(m_kappa));
        unsigned count = sum_terms;
        Ttype rate_max = rate_min;
        for (const auto &mode : modes)
        {
            count = std::max(count, static_cast<unsigned>(mode[0]) + 1u);
            rate_max = std::max(rate_max, mode[1]);
        }
        const auto mode_rates = bessel_mode_rates(bessel_j0_zeros<Ttype>(1u, count), m_kappa);
        for (const auto &mode : modes)
        {
            retained_rates.push_back(mode[1]);
            paired_mode_rates.push_back(mode_rates[static_cast<std::size_t>(mode[0])]);
        }
        sum_mode_rates.assign(mode_rates.begin(), mode_rates.begin() + sum_terms);
        if (rate_max > rate_min)
            x_plain = std::log(static_cast<Ttype>(1) / std::numeric_limits<Ttype>::epsilon()) / (rate_max - rate_min);
    }

    std::vector<Ttype> result(x_points.size());
    for (std::size_t p = 0; p < x_points.size(); ++p)
    {
        Ttype num = 0, den = 0, delta_num = 0;
        for (std::size_t i = 0; i < modes.size(); ++i)
        {
            const Ttype decay = i == slow ? static_cast<Ttype>(1)
                : std::exp(-(modes[i][1] - rate_min) * x_points[p]);
            const Ttype weight = modes[i][2] * decay;
            num += weight * modes[i][3];
            den += weight * modes[i][4];
            delta_num += (modes[i][2] * modes[i][3] + two_sqrt_two_pi) * decay;
        }
        if (asymptotic)
        {
            // N = c_inf G + sum_k delta_k e^{-Lam_k x}, c_inf = -2 sqrt(2 pi).
            if (x_points[p] < x_plain)
                num = delta_num - two_sqrt_two_pi
                    * (spectral_function_g0(retained_rates, paired_mode_rates, x_points[p], rate_min)
                       + spectral_function_g_inf(sum_mode_rates, m_kappa, x_points[p], rate_min));
        }
        else if (split)
            num -= two_sqrt_two_pi * spectral_function_sum(tail_rates, x_points[p], rate_min);
        if (!(den > static_cast<Ttype>(0)) ||
            !std::isfinite(static_cast<long double>(num / den)))
            throw std::runtime_error("get_nusselt_number: degenerate bulk value.");
        result[p] = -static_cast<Ttype>(0.5L) * num / den;
    }
    return result;
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::get_nusselt_number(Ttype x) const
{
    return get_nusselt_number(std::vector<Ttype>{x}).front();
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::get_nusselt_number(Ttype x, NusseltEvaluation evaluation) const
{
    return get_nusselt_number(std::vector<Ttype>{x}, evaluation).front();
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::nusselt_tail_rates(const std::vector<std::array<Ttype, 5>> &modes,
                                                             Ttype x_min, NusseltEvaluation evaluation,
                                                             const char *caller) const
{
    if (!(x_min > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(x_min)))
        throw std::invalid_argument(std::string(caller) + ": x_min must be finite and positive.");
    if (m_solution_method != SolutionMethod::ModifiedRoots)
        throw std::logic_error(std::string(caller) + ": the tail rates require the finite-Peclet "
                               "solution (setup_fp_solution).");

    Ttype rate_min = modes.front()[1], rate_max = modes.front()[1], rate_below_max = 0;
    unsigned next_zero = 1;
    for (const auto &mode : modes)
    {
        rate_min = std::min(rate_min, mode[1]);
        if (mode[1] > rate_max)
        {
            rate_below_max = rate_max;
            rate_max = mode[1];
        }
        next_zero = std::max(next_zero, static_cast<unsigned>(mode[0]) + 2u);
    }
    if (modes.size() == 1)
        rate_below_max = 0;

    // Stop once the rest of the tail, bounded with the current spacing (which
    // does not decrease), is below 16 eps of the part of the sum already kept.
    const Ttype tolerance = 16 * std::numeric_limits<Ttype>::epsilon();
    std::vector<Ttype> rates;
    Ttype kept = 1;
    unsigned chunk = 0;
    while (true)
    {
        const Ttype last = rates.empty() ? rate_max : rates.back();
        const Ttype before = rates.size() >= 2 ? rates[rates.size() - 2]
            : (rates.size() == 1 ? rate_max : rate_below_max);
        const Ttype spacing = last - before;
        const Ttype remainder = std::exp(-(last + spacing - rate_min) * x_min) / -std::expm1(-spacing * x_min);
        if (remainder <= tolerance * kept)
            break;
        // Each further rate shrinks the bound by at least exp(-spacing x_min).
        const Ttype needed = std::ceil(std::log(remainder / (tolerance * kept)) / (spacing * x_min));
        chunk = static_cast<unsigned>(std::min(std::max(needed, static_cast<Ttype>(1)), static_cast<Ttype>(4096)));

        std::vector<Ttype> more;
        if (evaluation == NusseltEvaluation::SpectralSplit)
            more = fp_dirichlet_rates_beyond<Ttype>(m_kappa, before, last, chunk, m_fp_rel_tol, m_fp_max_iter);
        else
        {
            more = slug_flow_rates<Ttype>(bessel_j0_zeros<Ttype>(next_zero, chunk), m_kappa,
                                          nusselt_slug_mean_velocity<Ttype>());
            next_zero += chunk;
        }
        for (const Ttype rate : more)
            kept += std::exp(-(rate - rate_min) * x_min);
        rates.insert(rates.end(), more.begin(), more.end());
    }
    return rates;
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::get_nusselt_tail_rates(Ttype x_min) const
{
    const auto modes = nusselt_axisymmetric_mode_data("get_nusselt_tail_rates");
    return nusselt_tail_rates(modes, x_min, NusseltEvaluation::SpectralSplit, "get_nusselt_tail_rates");
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::nusselt_converged_x_min(Ttype tol) const
{
    if (!(tol > static_cast<Ttype>(0)) || !(tol < static_cast<Ttype>(1)))
        throw std::invalid_argument("nusselt_converged_x_min: tol must lie in (0, 1).");
    const auto modes = nusselt_axisymmetric_mode_data("nusselt_converged_x_min");
    Ttype rate_max = 0;
    for (const auto &mode : modes)
        rate_max = std::max(rate_max, mode[1]);
    return std::log(static_cast<Ttype>(1) / tol) / rate_max;
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::get_fully_developed_nusselt_number() const
{
    const auto modes = nusselt_axisymmetric_mode_data("get_fully_developed_nusselt_number");
    std::size_t slow = 0;
    for (std::size_t i = 1; i < modes.size(); ++i)
        if (modes[i][1] < modes[slow][1])
            slow = i;
    if (!(modes[slow][4] > static_cast<Ttype>(0)))
        throw std::runtime_error("get_fully_developed_nusselt_number: degenerate bulk moment.");
    return -static_cast<Ttype>(0.5L) * modes[slow][3] / modes[slow][4];
}

template <typename Ttype>
std::vector<std::array<Ttype, 5>> CDBaseSolution<Ttype>::get_nusselt_mode_data() const
{
    return nusselt_axisymmetric_mode_data("get_nusselt_mode_data");
}

// ---------------------------------------------------------------------------
// Blurriness (thesis, chapter "Applications", Sec. "Blurriness"). See the
// protected block of CDBaseSolution.h for the derivation of the formulas.
// ---------------------------------------------------------------------------

template <typename Ttype>
/**
 * || f~ ||^2_omega of the internally scaled layered inlet, from the weighted
 * cap areas: sum_i u~_i^2 [ F(z_i) - F(z_{i+1}) ], F the partial flux.
 */
Ttype CDBaseSolution<Ttype>::inlet_square_norm_scaled() const
{
    Ttype total = static_cast<Ttype>(0);
    Ttype lower = static_cast<Ttype>(-1);
    Ttype flux_lower = compute_partial_flux(lower);
    for (std::size_t i = 0; i < m_ui.size(); ++i)
    {
        const Ttype upper = i < m_zi.size() ? m_zi[i] : static_cast<Ttype>(1);
        const Ttype flux_upper = compute_partial_flux(upper);
        total += m_ui[i] * m_ui[i] * (flux_lower - flux_upper);
        lower = upper;
        flux_lower = flux_upper;
    }
    return total;
}

template <typename Ttype>
/**
 * Build the L2_omega Gram blocks of the retained modes and the three scalars
 * (c^T W c, D_K, a_inf^T W a_inf) every blurriness evaluation needs. Bare
 * setup: W is diagonal in the tabulated norms. Finite-Peclet setup: the same
 * Gauss-Jacobi assembly that produced the coefficients (eq. gram_matrix_weighted),
 * so that c^T W c reproduces the stored contraction c^T b to round-off.
 */
void CDBaseSolution<Ttype>::prepare_blurriness() const
{
    require_solution_ready("get_blurriness");
    if (m_blurriness_ready)
        return;
    if (m_solution_method == SolutionMethod::QEP)
        throw std::logic_error("get_blurriness: not available for the QEP setup, whose modes do not carry one "
                               "decay rate each; use setup_fp_solution (exact finite-Peclet modes) or "
                               "setup_bare_solution.");

    // Angular blocks in the order of m_series_data (grouped by n, increasing m).
    std::map<unsigned, std::vector<unsigned>> by_n;
    for (unsigned k = 0; k < m_max_K; ++k)
        by_n[m_series_data[k].n].push_back(k);
    std::vector<BlurrinessBlock> blocks;
    blocks.reserve(by_n.size());
    for (auto &entry : by_n)
    {
        BlurrinessBlock block;
        block.modes = std::move(entry.second);
        blocks.push_back(std::move(block));
    }

    std::string failure;
#pragma omp parallel for schedule(dynamic)
    for (long b = 0; b < static_cast<long>(blocks.size()); ++b)
    {
        BlurrinessBlock &block = blocks[static_cast<std::size_t>(b)];
        const std::size_t K = block.modes.size();
        try
        {
            block.gram.assign(K * K, static_cast<Ttype>(0));
            if (m_solution_method == SolutionMethod::Bare)
            {
                for (std::size_t i = 0; i < K; ++i)
                    block.gram[i * K + i] = m_series_data[block.modes[i]].norm;
            }
            else
            {
                std::vector<SeriesData> modes;
                modes.reserve(K);
                for (unsigned k : block.modes)
                    modes.push_back(m_series_data[k]);
                FPGramGaussJacobiOptions<Ttype> options;
                options.oversampling_factor = m_gram_oversampling_factor;
                options.oversampling_margin = m_gram_oversampling_margin;
                options.enable_order_check = m_gram_order_check;
                options.minimum_factor = m_gram_minimum_factor;
                const auto factor = fp_gram_factor_gauss_jacobi<Ttype>(modes.front().n, modes, options);
                std::vector<std::vector<Ttype>> gram;
                fp_gram_reconstruct_gauss_jacobi<Ttype>(factor, gram);
                if (gram.size() != K)
                    throw std::runtime_error("prepare_blurriness: Gram block size mismatch");
                for (std::size_t i = 0; i < K; ++i)
                    for (std::size_t j = 0; j < K; ++j)
                        block.gram[i * K + j] = gram[i][j];
            }
        }
        catch (const std::exception &e)
        {
#pragma omp critical
            failure = e.what();
        }
    }
    if (!failure.empty())
        throw std::runtime_error("prepare_blurriness: " + failure);

    m_blurriness_blocks = std::move(blocks);
    m_blurriness_ready = true; // needed by blurriness_quadratic_form below
    // Retained variance a_inf^T W a_inf = || f_K - psi_inf ||^2_omega (kept for the
    // Dropped treatment and as a consistency check against the closed forms).
    m_blurriness_far_field_form = blurriness_quadratic_form(std::numeric_limits<Ttype>::infinity());

    // psi_inf: the term of the series with zero rate, constant over the disk.
    Ttype far_field = static_cast<Ttype>(0);
    for (unsigned k = 0; k < m_max_K; ++k)
    {
        const SeriesData &t = m_series_data[k];
        if (is_zero_eigenvalue_mode(t.n, t.m))
            far_field += active_coeff(t) * active_radial(t, static_cast<Ttype>(0)) * sn_phi(t.n, static_cast<Ttype>(0));
    }
    far_field += active_constant();
    m_blurriness_far_field = far_field;

    // Exact denominator || f~ - psi~_inf ||^2_omega = ||f~||^2 - 2 psi~_inf <f~,1>_omega + psi~_inf^2 |D|_omega,
    // with <f~,1>_omega = |D|_omega times the flow-weighted mean and |D|_omega = pi/2.
    const Ttype half_pi = static_cast<Ttype>(1.57079632679489661923132169163975144L);
    const Ttype norm_f = inlet_square_norm_scaled();
    m_blurriness_denominator = norm_f - static_cast<Ttype>(2) * far_field * half_pi * inlet_flow_weighted_mean_scaled()
                               + far_field * far_field * half_pi;

    // Parseval defect D_K = ||f||^2 - ||f_K||^2 (eq. truncation_error_fK), with
    // ||f_K||^2 the stored Galerkin contraction c^T b -- the same number the
    // Parseval analysis of the inlet projection uses. It is the L2_omega distance
    // between f and f_K only for the L2_omega projection (bare, or Weighted on
    // the finite-Peclet path); otherwise it is left undefined and the tails are
    // rejected in get_blurriness.
    const bool omega_projection = m_solution_method == SolutionMethod::Bare
                                  || m_projection_space == ProjectionSpace::Weighted;
    if (omega_projection && std::isfinite(static_cast<long double>(m_inlet_projection_square_norm)))
    {
        const Ttype defect = norm_f - m_inlet_projection_square_norm;
        m_blurriness_tail = defect > static_cast<Ttype>(0) ? defect : static_cast<Ttype>(0);
    }
    else
        m_blurriness_tail = std::numeric_limits<Ttype>::quiet_NaN();
}

template <typename Ttype>
/**
 * sum_n a^(n)(x)^T W^(n) a^(n)(x) with a_k(x) = c_k (e^{-Lam_k x} - 1); the
 * zero-eigenvalue mode contributes nothing at any x, including x = +inf.
 */
Ttype CDBaseSolution<Ttype>::blurriness_quadratic_form(Ttype x) const
{
    Ttype total = static_cast<Ttype>(0);
    std::vector<Ttype> a;
    for (const auto &block : m_blurriness_blocks)
    {
        const std::size_t K = block.modes.size();
        a.assign(K, static_cast<Ttype>(0));
        for (std::size_t i = 0; i < K; ++i)
        {
            const SeriesData &t = m_series_data[block.modes[i]];
            const Ttype rate = active_rate(t);
            const Ttype decay = rate > static_cast<Ttype>(0) ? std::exp(-rate * x) : static_cast<Ttype>(1);
            a[i] = t.coeff * (decay - static_cast<Ttype>(1));
        }
        for (std::size_t i = 0; i < K; ++i)
        {
            if (a[i] == static_cast<Ttype>(0))
                continue;
            Ttype row = static_cast<Ttype>(0);
            for (std::size_t j = 0; j < K; ++j)
                row += block.gram[i * K + j] * a[j];
            total += a[i] * row;
        }
    }
    return total;
}

template <typename Ttype>
/**
 * S_omega = sum_j int_{Gamma_j} [[f]]_j^2 omega dl for the layered inlet, whose
 * jump curves are the chords z = z_i: with a_i^2 = 1 - z_i^2 the half-chord and
 * omega = a_i^2 - y^2 on the chord, int_{-a}^{a} omega dy = (4/3) a^3, so
 *
 *     S_omega = (4/3) sum_i (u_{i+1} - u_i)^2 (1 - z_i^2)^{3/2} .
 *
 * m_ui carries the internal scaling of set_ui, so the result is homogeneous
 * with D_K and the ratio of the two is scale free.
 */
Ttype CDBaseSolution<Ttype>::inlet_jump_flux_moment() const
{
    Ttype total = static_cast<Ttype>(0);
    for (std::size_t i = 0; i + 1 < m_ui.size() && i < m_zi.size(); ++i)
    {
        const Ttype a2 = static_cast<Ttype>(1) - m_zi[i] * m_zi[i];
        if (!(a2 > static_cast<Ttype>(0)))
            continue;   // a jump on (or outside) the wall carries no flux
        const Ttype jump = m_ui[i + 1] - m_ui[i];
        total += jump * jump * a2 * std::sqrt(a2);
    }
    return static_cast<Ttype>(4) / static_cast<Ttype>(3) * total;
}

template <typename Ttype>
/**
 * c^2 with c = S_omega / (pi beta_K D_K). See the header for what it is and why
 * it is not fitted. Both inputs are exact; when either is unavailable (no
 * interior jump, or D_K not built) the previous mode-averaged constant 2/3 is
 * used, which keeps the rate law well posed without claiming the calibration.
 */
Ttype CDBaseSolution<Ttype>::tail_edge_ratio_squared() const
{
    const Ttype fallback = static_cast<Ttype>(2) / static_cast<Ttype>(3);
    const Ttype s_omega = inlet_jump_flux_moment();
    const Ttype denom = static_cast<Ttype>(M_PI) * m_max_root * m_blurriness_tail;
    if (!(s_omega > static_cast<Ttype>(0)) || !(denom > static_cast<Ttype>(0)))
        return fallback;
    const Ttype c = s_omega / denom;
    if (!std::isfinite(static_cast<long double>(c)) || !(c > static_cast<Ttype>(0)))
        return fallback;
    return c * c;
}

template <typename Ttype>
/**
 * Rate of a mode of bare root beta beyond the truncation: the decaying root of
 *
 *     kappa Lam^2 + c^2 Lam = c^2 beta^2 ,
 *
 * written in the cancellation-free form
 *
 *     Lam(beta) = 2 beta^2 / (1 + sqrt(1 + 4 kappa beta^2 / c^2)) .
 *
 * The two limits are the ones the entrance analysis requires. As kappa -> 0,
 * Lam -> beta^2 and the bare curve is untouched. Once kappa beta^2 >> c^2,
 * Lam -> c beta / sqrt(kappa) = c beta Pe, and since c = S_omega/(pi beta_K D_K)
 * the tail integral then gives T_K -> 2 ln2 D_K beta_K c Pe x = (2 ln2 / pi)
 * S_omega Pe x, the exact harmonic entrance law, at every Peclet number.
 *
 * c beta_K / sqrt(kappa) is the decay rate at the truncation edge, so c beta_K
 * is the wavenumber the truncated series resolves uniformly. Earlier versions
 * fixed that ratio at a constant (1, then the mode average <omega> = 2/3); it
 * is not constant, because the edge wavenumber varies along the jump chord and
 * its flux-weighted average is inlet geometry, not a universal number.
 */
Ttype CDBaseSolution<Ttype>::tail_rate(Ttype beta) const
{
    const Ttype beta2 = beta * beta;
    if (!(m_kappa > static_cast<Ttype>(0)))
        return beta2;
    const Ttype c2 = tail_edge_ratio_squared();
    return static_cast<Ttype>(2) * beta2 / (static_cast<Ttype>(1) + std::sqrt(static_cast<Ttype>(1) + static_cast<Ttype>(4) * m_kappa * beta2 / c2));
}

template <typename Ttype>
/**
 * T_K(x) = D_K int_0^1 p s^(p-1) (1 - e^{-Lam(beta_K / s) x})^2 ds, the energy
 * of the omitted modes that has already left the inlet datum at x. The
 * substitution beta = beta_K / s maps the tail onto (0, 1]; the integrand rises
 * from 0 at s = 0 (infinitely fast modes, already decayed) to its value at
 * s = 1 (the first omitted mode). Its transition sits where Lam(beta_K/s) x ~ 1,
 * so the interval is split there and each part integrated with Gauss-Legendre.
 */
Ttype CDBaseSolution<Ttype>::blurriness_asymptotic_tail(Ttype x) const
{
    if (!(x > static_cast<Ttype>(0)))
        return static_cast<Ttype>(0);
    if (!std::isfinite(static_cast<long double>(x)))
        return m_blurriness_tail;
    const Ttype p = truncation_defect_exponent();
    const Ttype beta_K = m_max_root;
    const auto integrand = [&](Ttype s) {
        const Ttype decay = std::exp(-tail_rate(beta_K / s) * x);
        const Ttype g = (static_cast<Ttype>(1) - decay);
        return p * std::pow(s, p - static_cast<Ttype>(1)) * g * g;
    };
    // transition abscissa: Lam(beta_K / s*) x = 1
    Ttype s_star = static_cast<Ttype>(1);
    {
        // bare: beta^2 x = 1 -> s* = beta_K sqrt(x); finite Pe: Lam = 1/x, i.e.
        // c^2 beta^2 = kappa/x^2 + c^2/x -> beta^2 = kappa/(c^2 x^2) + 1/x
        const Ttype beta_star2 = m_kappa / (tail_edge_ratio_squared() * x * x) + static_cast<Ttype>(1) / x;
        const Ttype ratio = beta_K / std::sqrt(beta_star2);
        if (ratio < static_cast<Ttype>(1))
            s_star = ratio;
    }
    static const long double gl_x[16] = {
        -0.9894009349916499325961541734504L, -0.9445750230732325760779884155346L, -0.8656312023878317438804678977123L,
        -0.7554044083550030338951011948474L, -0.6178762444026437484466717640413L, -0.4580167776572273863424194429835L,
        -0.2816035507792589132304605014605L, -0.0950125098376374401853193354250L, 0.0950125098376374401853193354250L,
        0.2816035507792589132304605014605L, 0.4580167776572273863424194429835L, 0.6178762444026437484466717640413L,
        0.7554044083550030338951011948474L, 0.8656312023878317438804678977123L, 0.9445750230732325760779884155346L,
        0.9894009349916499325961541734504L};
    static const long double gl_w[16] = {
        0.0271524594117540948517805724560L, 0.0622535239386478928628438369944L, 0.0951585116824927848099251076022L,
        0.1246289712555338720524762821920L, 0.1495959888165767320815017305474L, 0.1691565193950025381893120790304L,
        0.1826034150449235888667636679692L, 0.1894506104550684962853967232083L, 0.1894506104550684962853967232083L,
        0.1826034150449235888667636679692L, 0.1691565193950025381893120790304L, 0.1495959888165767320815017305474L,
        0.1246289712555338720524762821920L, 0.0951585116824927848099251076022L, 0.0622535239386478928628438369944L,
        0.0271524594117540948517805724560L};
    // Composite rule on geometric panels [s/2, s] from 1 down to 1e-12 (the
    // integrand is a smooth power-like function of s on either side of s*, so
    // panels of fixed ratio resolve it uniformly in log s), with the panel
    // boundaries aligned to s* so that the transition is not straddled.
    const auto integrate = [&](Ttype a, Ttype b) {
        Ttype sum = static_cast<Ttype>(0);
        const Ttype c = static_cast<Ttype>(0.5L) * (a + b), hh = static_cast<Ttype>(0.5L) * (b - a);
        for (unsigned g = 0; g < 16; ++g)
            sum += static_cast<Ttype>(gl_w[g]) * integrand(c + hh * static_cast<Ttype>(gl_x[g]));
        return sum * hh;
    };
    Ttype total = static_cast<Ttype>(0);
    Ttype upper = static_cast<Ttype>(1);
    if (s_star < static_cast<Ttype>(1))
    {
        // above the transition: panels of ratio 2 from s* upwards, the last one ending at 1
        Ttype lo = s_star;
        while (lo < static_cast<Ttype>(1))
        {
            const Ttype hi = std::min(static_cast<Ttype>(1), lo * static_cast<Ttype>(2));
            total += integrate(lo, hi);
            lo = hi;
        }
        upper = s_star;
    }
    for (unsigned level = 0; level < 48 && upper > static_cast<Ttype>(1e-14); ++level)
    {
        const Ttype lower = upper * static_cast<Ttype>(0.5L);
        total += integrate(lower, upper);
        upper = lower;
    }
    return m_blurriness_tail * std::min(total, static_cast<Ttype>(1));
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::get_blurriness(Ttype x, BlurrinessTail tail) const
{
    prepare_blurriness();
    if (!(x >= static_cast<Ttype>(0)))
        throw std::invalid_argument("get_blurriness: x must be non-negative (dimensionless axial coordinate x / (Pe R)).");
    if (tail != BlurrinessTail::Dropped && !std::isfinite(static_cast<long double>(m_blurriness_tail)))
        throw std::logic_error("get_blurriness: the truncation tail D_K = ||f||^2 - ||f_K||^2 is the L2_omega "
                               "distance between the inlet and its projection only for the L2_omega (Weighted) "
                               "projection; with ProjectionSpace::Radial use BlurrinessTail::Dropped.");
    // Numerator: retained modes a(x)^T W a(x) plus the omitted modes; denominator:
    // the exact || f - psi_inf ||^2 (Dropped: minus the omitted energy, i.e. the
    // retained variance || f_K - psi_inf ||^2, so that the ratio still tends to 1).
    Ttype numerator = blurriness_quadratic_form(x);
    Ttype denominator = m_blurriness_denominator;
    if (tail == BlurrinessTail::Constant)
        numerator += m_blurriness_tail;
    else if (tail == BlurrinessTail::Asymptotic)
        numerator += blurriness_asymptotic_tail(x);
    else
        denominator = m_blurriness_far_field_form;
    if (!(denominator > static_cast<Ttype>(0)))
        throw std::runtime_error("get_blurriness: the inlet equals its far field, so the blurriness is undefined.");
    const Ttype ratio = numerator / denominator;
    return std::sqrt(ratio > static_cast<Ttype>(0) ? ratio : static_cast<Ttype>(0));
}

template <typename Ttype>
std::vector<Ttype> CDBaseSolution<Ttype>::get_blurriness(const std::vector<Ttype> &x_points, BlurrinessTail tail) const
{
    prepare_blurriness();
    std::vector<Ttype> out(x_points.size());
    for (std::size_t i = 0; i < x_points.size(); ++i)
        out[i] = get_blurriness(x_points[i], tail);
    return out;
}

template <typename Ttype>
Ttype CDBaseSolution<Ttype>::get_blurriness_truncation_floor() const
{
    return get_blurriness(static_cast<Ttype>(0), BlurrinessTail::Constant);
}

template <typename Ttype>
/**
 * psi_inf in the user's units: the zero-eigenvalue modes evaluated on the axis
 * (they are constant over the disk), carried back through the inverse of the
 * set_ui scaling. A basis without a constant mode (Dirichlet wall) gives the
 * wall value.
 */
Ttype CDBaseSolution<Ttype>::get_far_field_value() const
{
    prepare_blurriness();
    return m_max_ui - m_blurriness_far_field * (m_max_ui - m_min_ui);
}

template <typename Ttype>
/**
 * || f ||^2_omega of the user's inlet, exact. With f = alpha + beta f~ (alpha =
 * m_max_ui, beta = -(m_max_ui - m_min_ui)):
 *   || f ||^2 = alpha^2 |D|_omega + 2 alpha beta <1, f~>_omega + beta^2 || f~ ||^2,
 * with |D|_omega = pi/2 and <1, f~>_omega = |D|_omega times the flow-weighted mean.
 */
Ttype CDBaseSolution<Ttype>::get_inlet_square_norm() const
{
    const Ttype half_pi = static_cast<Ttype>(1.57079632679489661923132169163975144L);
    const Ttype alpha = m_max_ui;
    const Ttype beta = -(m_max_ui - m_min_ui);
    return alpha * alpha * half_pi + static_cast<Ttype>(2) * alpha * beta * half_pi * inlet_flow_weighted_mean_scaled()
           + beta * beta * inlet_square_norm_scaled();
}

template <typename Ttype>
/**
 * || f - psi_inf ||^2_omega of the user's inlet: beta^2 times the exact
 * internal denominator of the blurriness.
 */
Ttype CDBaseSolution<Ttype>::get_inlet_far_field_square_norm() const
{
    prepare_blurriness();
    const Ttype beta = m_max_ui - m_min_ui;
    return beta * beta * m_blurriness_denominator;
}

#endif
