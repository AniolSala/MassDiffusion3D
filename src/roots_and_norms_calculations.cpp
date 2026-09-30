#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

#include "io_data.h"
#include "math_functions.h"
#include "comp_utils.h"
#include "roots_and_norms_calculations.h"


template <typename Ttype>
void psinm_r_norm(unsigned k, Ttype b, Ttype r, Ttype &val) {
    Ttype psi_val = 0.0L;
    psinm_r(k, b, r, psi_val);
    val = psi_val * psi_val;
}

template <typename Ttype>
Ttype get_norm(const unsigned &n, const Ttype &b, const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes)
{
    auto integrand = [n, b](Ttype r, Ttype &val) { psinm_r_norm(n, b, r, val); };
    Ttype result = 0.0L;
    gaussian_integration(integrand, weights, nodes, result);
    return result;
}

template <typename Ttype>
void psinm_r_product(const unsigned &k, const Ttype &b1, const Ttype &b2, const Ttype &r, Ttype &val) {
    Ttype psi_val_1 = 0.0L;
    Ttype psi_val_2 = 0.0L;
    Ttype u_val = r * std::sqrt(2.0L - r * r);
    psinm_r(k, b1, u_val, psi_val_1);
    psinm_r(k, b2, u_val, psi_val_2);
    val = 2.0L * psi_val_1 * psi_val_2;  // TODO: Get correct coefficient here
}

template <typename Ttype>
Ttype get_integration_product_r(const unsigned &n, const Ttype &b1, const Ttype &b2, const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes)
{
    auto integrand = [n, b1, b2](const Ttype r, Ttype &val) { psinm_r_product(n, b1, b2, r, val); };
    Ttype result = 0.0L;
    gaussian_integration(integrand, weights, nodes, result);
    return result;
}

template <typename Ttype>
void psinm_r_product_direct(const unsigned &k, const Ttype &b1, const Ttype &b2, const Ttype &r, Ttype &val) {
    Ttype psi1 = 0.0L, psi2 = 0.0L;
    psinm_r(k, b1, r, psi1);
    psinm_r(k, b2, r, psi2);
    val = psi1 * psi2;
}

// Integrates ψ(n,b1,r)·ψ(n,b2,r) with the weight encoded in the supplied quadrature nodes/weights.
// Pass Bessel nodes/weights to get ∫ψ₁ψ₂ r dr; pass hypergeometric nodes/weights for ∫ψ₁ψ₂(1−r²)r dr.
template <typename Ttype>
Ttype get_integration_product_r_direct(const unsigned &n, const Ttype &b1, const Ttype &b2,
                                        const std::vector<Ttype> &weights, const std::vector<Ttype> &nodes)
{
    auto integrand = [n, b1, b2](const Ttype r, Ttype &val) { psinm_r_product_direct(n, b1, b2, r, val); };
    Ttype result = 0.0L;
    gaussian_integration(integrand, weights, nodes, result);
    return result;
}

template <typename Ttype>
Ttype find_root(const unsigned &n, const Ttype &approx_root, const Ttype &tol, const Ttype &rel_tol, const Ttype& norm_factor,
                const unsigned &max_iter, const bool& use_position_convergence, const bool& use_residual_convergence)
{
    Ttype root = 0.0L;

    if (!use_position_convergence && !use_residual_convergence)
    {
        std::cerr << "ERROR: Invalid convergence settings in find_root for (n=" << n
                  << ", approx_root = " << approx_root << "). At least one of use_position_convergence "
                  << "or use_residual_convergence must be true." << std::endl;
        exit(1);
    }

    // Adaptive bracket width: scale with approx_root magnitude
    // For large roots, use a relative bracket; for small roots, use absolute bracket
    // Ttype bracket_width = std::max(static_cast<Ttype>(0.5L), std::abs(approx_root) * static_cast<Ttype>(0.01L));
    Ttype bracket_width = static_cast<Ttype>(0.5L);
    Ttype max_bracket_width = static_cast<Ttype>(1.0L);
    unsigned max_iters_bracket = 10;
    Ttype increment_factor = powl(max_bracket_width / bracket_width, 1.0L / max_iters_bracket);

    Ttype a = approx_root - bracket_width;
    Ttype b = approx_root + bracket_width;
    
    // Max iterations for safeguarded Newton+bisection iterations is provided by caller
    
    // Function and derivative for f_n(x) = dpsinm_r(1, n, x, 1)
    const unsigned ord = 1;
    Ttype r_val = 1;
    auto f = [&](const Ttype& x) -> Ttype {
        Ttype val = dpsinm_r(ord, n, x, r_val);
        val /= norm_factor;
        return val;
    };

    auto df = [&](const Ttype& x, const Ttype& left_b, const Ttype& right_b) -> Ttype {
        const Ttype eps = std::numeric_limits<Ttype>::epsilon();
        const Ttype x_scale = std::max(static_cast<Ttype>(1.0L), std::fabs(x));
        Ttype h = std::cbrt(eps) * x_scale;
        h = std::max(h, static_cast<Ttype>(8.0L) * eps * x_scale);

        const Ttype available_left = x - left_b;
        const Ttype available_right = right_b - x;

        if (available_left > h && available_right > h)
        {
            return (f(x + h) - f(x - h)) / (static_cast<Ttype>(2.0L) * h);
        }

        Ttype h_forward = std::min(h, available_right);
        Ttype h_backward = std::min(h, available_left);

        if (h_forward > static_cast<Ttype>(0.0L))
        {
            return (f(x + h_forward) - f(x)) / h_forward;
        }

        if (h_backward > static_cast<Ttype>(0.0L))
        {
            return (f(x) - f(x - h_backward)) / h_backward;
        }

        return std::numeric_limits<Ttype>::quiet_NaN();
    };
    
    // Evaluate at endpoints
    Ttype fa = f(a);
    Ttype fb = f(b);
    
    // Ensure opposite signs (bracketing condition)
    for(unsigned iter = 1; iter < max_iters_bracket; iter++)
    {
        if (fa * fb > 0.0L)
        {
            // Try to expand the bracket if needed
            bracket_width *= increment_factor;
            a = approx_root - bracket_width;
            b = approx_root + bracket_width;
            fa = f(a);
            fb = f(b);
            
        }
        if((iter == max_iters_bracket - 1) && (fa * fb > 0.0L))
        {
            // Fallback to midpoint if bracketing fails
            root = approx_root;
                std::cerr << std::setprecision(std::numeric_limits<Ttype>::max_digits10) << std::scientific
                    << "ERROR: Unable to bracket root before safeguarded Newton+bisection for (n=" << n
                    << ", approx_root = " << approx_root << "). Found f(a=" << a << ") = " << fa << ", " << "f(b=" << b << ") = "
                    << fb << ". Returning approximate value: " << root << std::endl;
            exit(1);
        }
    }

    // Safeguarded Newton method: use derivative when possible, fallback to bisection
    Ttype left = a;
    Ttype right = b;
    Ttype f_left = fa;
    Ttype x = approx_root;
    if (x <= left || x >= right)
    {
        x = 0.5L * (left + right);
    }

    Ttype fx = f(x);

    auto x_converged = [&](const Ttype &x_old, const Ttype &x_new, const Ttype &left_b, const Ttype &right_b) -> bool {
        Ttype x_scale = std::max(static_cast<Ttype>(1.0L), std::fabs(x_new));
        Ttype x_tol = tol + rel_tol * x_scale;
        Ttype step = std::fabs(x_new - x_old);
        Ttype bracket_size = std::fabs(right_b - left_b);
        return (step <= x_tol) || (bracket_size <= static_cast<Ttype>(2.0L) * x_tol);
    };

    for (unsigned iter = 0; iter < max_iter; ++iter)
    {
        Ttype x_new = 0.5L * (left + right);
        bool use_newton = false;

        Ttype dfx = df(x, left, right);
        if (std::isfinite(static_cast<long double>(dfx)) &&
            std::fabs(dfx) > std::sqrt(std::numeric_limits<Ttype>::epsilon()))
        {
            Ttype candidate = x - fx / dfx;
            if (std::isfinite(static_cast<long double>(candidate)) &&
                candidate > left && candidate < right)
            {
                x_new = candidate;
                use_newton = true;
            }
        }

        if (!use_newton)
        {
            x_new = 0.5L * (left + right);
        }

        Ttype f_new = f(x_new);
        Ttype x_scale_new = std::max(static_cast<Ttype>(1.0L), std::fabs(x_new));
        Ttype x_tol_new = tol + rel_tol * x_scale_new;
        Ttype dfx_new = df(x_new, left, right);
        bool residual_converged = false;
        if (std::isfinite(static_cast<long double>(f_new)) &&
            std::isfinite(static_cast<long double>(dfx_new)) &&
            std::fabs(dfx_new) > std::sqrt(std::numeric_limits<Ttype>::epsilon()))
        {
            residual_converged = std::fabs(f_new / dfx_new) <= x_tol_new;
        }
        else if (std::fabs(f_new) == static_cast<Ttype>(0.0L))
        {
            residual_converged = true;
        }

        bool position_converged = x_converged(x, x_new, left, right);
        bool converged = (!use_residual_convergence || residual_converged) &&
                        (!use_position_convergence || position_converged);
        if (converged)
        {
            root = x_new;
            return root;
        }

        bool stagnated_due_precision = (x_new == x);
        if (stagnated_due_precision)
        {
            Ttype bracket_size_stagnated = std::fabs(right - left);
            std::cerr << std::setprecision(std::numeric_limits<Ttype>::max_digits10) << std::scientific
                      << "ERROR: Root-finding stagnated due to floating-point precision limits for (n=" << n
                      << ", approx_root = " << approx_root << ").\n"
                      << "  x no longer changes (x_new == x) before convergence criteria were met.\n"
                      << "  Iteration: " << (iter + 1) << " / " << max_iter
                      << ", root = " << x_new
                      << ", bracket size = " << bracket_size_stagnated
                      << "\n  Convergence criteria: use_position_convergence = " << use_position_convergence
                      << ", use_residual_convergence = " << use_residual_convergence
                      << "\n  Requested: abs_tol = " << tol << ", rel_tol = " << rel_tol
                      << "\n  Residual : |f(root)| = " << std::fabs(f_new)
                      << ", |f(root)/f'(root)| = "
                      << (std::isfinite(static_cast<long double>(dfx_new)) && std::fabs(dfx_new) > std::sqrt(std::numeric_limits<Ttype>::epsilon())
                          ? std::fabs(f_new / dfx_new)
                          : std::numeric_limits<Ttype>::infinity())
                      << ", target x_tol = " << x_tol_new
                      << std::endl;
            exit(1);
        }

        if (f_left * f_new > 0.0L)
        {
            left = x_new;
            f_left = f_new;
        }
        else
        {
            right = x_new;
        }

        x = x_new;
        fx = f_new;
    }
    
    // Max iterations reached, return best estimate
    root = x;
    
    // Check if convergence was achieved
    Ttype f_final = f(root);
    Ttype x_scale_final = std::max(static_cast<Ttype>(1.0L), std::fabs(root));
    Ttype x_tol_final = tol + rel_tol * x_scale_final;
    Ttype bracket_size_final = std::fabs(right - left);
    Ttype achieved_abs_tol = static_cast<Ttype>(0.5L) * bracket_size_final;
    Ttype achieved_rel_tol = achieved_abs_tol / x_scale_final;
    Ttype dfx_final = df(root, left, right);
    Ttype residual_x_estimate_final = std::numeric_limits<Ttype>::infinity();
    bool residual_converged_final = false;
    if (std::isfinite(static_cast<long double>(f_final)) &&
        std::isfinite(static_cast<long double>(dfx_final)) &&
        std::fabs(dfx_final) > std::sqrt(std::numeric_limits<Ttype>::epsilon()))
    {
        residual_x_estimate_final = std::fabs(f_final / dfx_final);
        residual_converged_final = residual_x_estimate_final <= x_tol_final;
    }
    else if (std::fabs(f_final) == static_cast<Ttype>(0.0L))
    {
        residual_x_estimate_final = static_cast<Ttype>(0.0L);
        residual_converged_final = true;
    }

    bool position_converged_final = bracket_size_final <= static_cast<Ttype>(2.0L) * x_tol_final;
    bool converged_final = (!use_residual_convergence || residual_converged_final) &&
                          (!use_position_convergence || position_converged_final);
    if (!converged_final)
    {
        std::cerr << std::setprecision(std::numeric_limits<Ttype>::max_digits10) << std::scientific
              << "ERROR: Safeguarded Newton+bisection did not converge to requested tolerances for (n=" << n
                  << ", approx_root = " << approx_root << "). |f(root)| = " << std::fabs(f_final)
                  << ", root = " << root
                  << ", bracket size = " << bracket_size_final
                  << "\n  Convergence criteria: use_position_convergence = " << use_position_convergence
                  << ", use_residual_convergence = " << use_residual_convergence
                  << "\n  Requested: abs_tol = " << tol << ", rel_tol = " << rel_tol
                  << "\n  Achieved : abs_tol = " << achieved_abs_tol << ", rel_tol = " << achieved_rel_tol
                  << "\n  Residual : |f(root)| = " << std::fabs(f_final)
                  << ", |f(root)/f'(root)| = " << residual_x_estimate_final
                  << ", target x_tol = " << x_tol_final
                  << "\n"
                  << "  Iterations used: " << max_iter << ", bracket width: " << bracket_width << std::endl;
        exit(1);
    }

    return root;
}

void find_roots_and_norms(const unsigned &min_n, const unsigned &min_m, const unsigned &max_n, const unsigned &max_m,
                std::vector<std::vector<long double>> &roots, std::vector<std::vector<long double>> &norms,
                const long double &tol, const long double &rel_tol, const unsigned &max_iters,
                const bool &use_position_convergence, const bool &use_residual_convergence, const bool &write_results_to_file,
                const bool &compare_with_previous_results)
{
    // Load previously computed data to check for potential issues
    std::vector<std::vector<long double>> precomputed_roots;
    std::vector<std::vector<long double>> precomputed_norms;
    if(compare_with_previous_results)
    {
        readValuesFromFile("./src/CDBaseSolution/data/broots_100x100.txt", precomputed_roots);
        readValuesFromFile("./src/CDBaseSolution/data/norms_100x100.txt", precomputed_norms);
    }

    // Load nodes and weights for gaussian integration
    std::vector<std::vector<long double>> gaussian_data;
    readValuesFromFile("./src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_300.txt", gaussian_data);
    std::vector<long double> weights;
    std::vector<long double> nodes;
    weights.reserve(gaussian_data.size());
    nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data) {
        weights.push_back(row[0]);
        nodes.push_back(row[1]);
    }

    // Dimensions of precomputed data
    unsigned dim1 = 0;
    unsigned dim2 = 0;
    if(compare_with_previous_results)
    {
        dim1 = precomputed_roots.size();
        dim2 = precomputed_roots[0].size();
    }

    // Initial values for root and norm
    long double previous_root = 1.06L;
    long double previous_norm = 1.0L;
    if((min_n < dim1 && min_m < dim2) && (min_n > 0 || min_m > 0))
    {
        if(min_m == 0)
        {
            previous_root = 2.0L * (static_cast<long double>(min_n - 1) + 1.0L) - 1.1L * expl(-0.25L * (static_cast<long double>(min_n - 1) - 1.0L));
            previous_norm = precomputed_norms[min_n - 1][0];
        } else {
            previous_root = precomputed_roots[min_n][min_m - 1];
            previous_norm = precomputed_norms[min_n][min_m - 1];
        }
    }

    std::cout << "\nComputing roots..." << std::endl;
    long double max_diff_precomputed_roots = 0.0L;
    for (unsigned n = min_n; n <= max_n; n++)
    {
        for (unsigned m = min_m; m <= max_m; m++)
        {
            long double approx_root;
            long double approx_norm;
            if (m == 0)
            {
                approx_root = 2.0L * (static_cast<long double>(n) + 1.0L) - 1.1L * expl(-0.25L * (static_cast<long double>(n) - 1.0L));
                approx_norm = norms[n - 1][0];
            }
            else
            {
                approx_root = previous_root + 4.00L;
                approx_norm = previous_norm;
            }

            long double root = 0.0L;
            long double norm = 0.0L;

            if (n == 0 && m == 0)
            {
                roots[0][0] = static_cast<long double>(0.0L);
                norms[0][0] = get_norm(0, 0.0L, weights, nodes);   // = 1/4 for R_00 == 1
                // leave previous_root / previous_norm at their init so the (0,1) guess is unchanged
                continue;
            }

            // Compute the root and the norm
            root = find_root(n, approx_root, tol, rel_tol, sqrtl(previous_norm), max_iters, use_position_convergence, use_residual_convergence);
            norm = get_norm(n, root, weights, nodes);

            roots[n][m] = root;
            norms[n][m] = norm;

            // Info message
            std::cout << "root[" << n << "][" << m << "] = " << root
            << ", norm[" << n << "][" << m << "] = " << norm
            << std::endl;

            // Compare root to previous root
            long double previous_root_diff = std::fabs(previous_root - root);
            if (previous_root_diff > max_diff_precomputed_roots) max_diff_precomputed_roots = previous_root_diff;
            if (m > 0 && previous_root_diff > 4.5L)
            {
                std::cout << "Error when comparing with previous root:\n"
                          << "  b[" << n << "][" << m << "] = " << root << "\n"
                          << "  b[" << n << "][" << m - 1 << "] = " << previous_root << "\n"
                          << "  previous_root_diff = " << previous_root_diff << std::endl;
                exit(1);
            }

            if (n < dim1 && m < dim2)
            {
                // Compare root to reference one
                long double root_comparison = std::fabs(root - precomputed_roots[n][m]);
                root_comparison /= precomputed_roots[n][m];
                if (root_comparison >= 1e-10L)
                {
                    std::cout << "Error when comparing to precomputed roots:\n"
                              << "  b[" << n << "][" << m << "] = " << root << "\n"
                              << "  precomputed_b[" << n << "][" << m << "] = " << precomputed_roots[n][m] << "\n"
                              << "  diff = " << root_comparison << std::endl;
                    exit(1);
                }

                // Compare norm to reference one
                long double norm_comparison = std::fabs(norm - precomputed_norms[n][m]);
                norm_comparison /= precomputed_norms[n][m];
                if (norm_comparison >= 1e-10L)
                {
                    std::cout << "Error when comparing to precomputed norms:\n"
                              << "  norm[" << n << "][" << m << "] = " << norm << "\n"
                              << "  precomputed_norm[" << n << "][" << m << "] = " << precomputed_norms[n][m] << "\n"
                              << "  diff = " << norm_comparison << std::endl;
                    exit(1);
                }
            }

            previous_root = root;
            previous_norm = norm;
        }
    }
    std::cout << "\nRoots completed correctly.\n" << std::endl;

    if (write_results_to_file)
    {
        std::cout << "\nSaving roots to file..." << std::endl;
        std::string roots_filename = "./src/CDBaseSolution/data/broots_" + std::to_string(max_n + 1) + "x" + std::to_string(max_m + 1) + "_alt.txt";
        std::ofstream roots_outfile(roots_filename);
        roots_outfile << std::setprecision(std::numeric_limits<long double>::max_digits10) << std::scientific;
        for (const auto &row : roots)
        {
            for (const auto &val : row)
            {
                roots_outfile << val << " ";
            }
            roots_outfile << "\n";
        }
        roots_outfile.close();
        std::cout << "Roots saved to " << roots_filename << std::endl;

        // Save norms
        std::cout << "\nSaving norms to file..." << std::endl;
        std::string norms_filename = "./src/CDBaseSolution/data/norms_" + std::to_string(max_n + 1) + "x" + std::to_string(max_m + 1) + "_alt.txt";
        std::ofstream norms_outfile(norms_filename);
        norms_outfile << std::setprecision(std::numeric_limits<long double>::max_digits10) << std::scientific;
        for (const auto &row : norms)
        {
            for (const auto &val : row)
            {
                norms_outfile << val << " ";
            }
            norms_outfile << "\n";
        }
        norms_outfile.close();
        std::cout << "Norms saved to " << norms_filename<< std::endl;
    }
}

void compute_pseudo_norms(const std::vector<std::vector<long double>>& roots)
{

}

template double find_root<double>(const unsigned&, const double&, const double&, const double&, const double &, const unsigned&, const bool&, const bool&);
template long double find_root<long double>(const unsigned&, const long double&, const long double&, const long double&, const long double&, const unsigned&, const bool&, const bool&);

template double get_norm<double>(const unsigned&, const double&, const std::vector<double>&, const std::vector<double>&);
template long double get_norm<long double>(const unsigned&, const long double&, const std::vector<long double>&, const std::vector<long double>&);

template double get_integration_product_r<double>(const unsigned&, const double&, const double&, const std::vector<double>&, const std::vector<double>&);
template long double get_integration_product_r<long double>(const unsigned&, const long double&, const long double&, const std::vector<long double>&, const std::vector<long double>&);

template double get_integration_product_r_direct<double>(const unsigned&, const double&, const double&, const std::vector<double>&, const std::vector<double>&);
template long double get_integration_product_r_direct<long double>(const unsigned&, const long double&, const long double&, const std::vector<long double>&, const std::vector<long double>&);