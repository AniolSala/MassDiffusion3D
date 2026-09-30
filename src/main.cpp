#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

#include "io_data.h"
#include "roots_and_norms_calculations.h"
#include "pseudo_products.h"
#include "math_functions.h"

#include "omp.h"


void write_roots_and_norms()
{
    // Set maximum precision for output
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10) << std::scientific;

    // Create a 2D vector to store roots (matching the static defs)
    const unsigned min_n = 0;
    const unsigned min_m = 0;
    const unsigned max_n = 99;
    const unsigned max_m = 99;
    std::vector<std::vector<long double>> roots(max_n + 1, std::vector<long double>(max_m + 1, 0.0L));
    std::vector<std::vector<long double>> norms(max_n + 1, std::vector<long double>(max_m + 1, 0.0L));

    // Root-solver settings used by find_roots_and_norms in roots_and_norms_calculations
    long double tol = 1e-12L;
    long double rel_tol = 1e-12L;
    unsigned max_iters = 10000;
    bool use_position_convergence = true;
    bool use_residual_convergence = true;

    // Write file
    bool write_results_to_file = true;
    bool compare_with_previous_results = false;

    find_roots_and_norms(min_n, min_m, max_n, max_m,
               roots, norms,
               tol, rel_tol, max_iters,
               use_position_convergence,
               use_residual_convergence,
               write_results_to_file,
               compare_with_previous_results);

}


// ---------------------------------------------------------------------------
// Graetz isothermal data generation
//
// Finds eigenvalues β_m satisfying  psinm_r(0, β_m, 1) = 0  (Dirichlet BC)
// and computes norms  N_m² = ∫₀¹ r(1−r²) [psinm_r(0,β_m,r)]² dr.
// Results are written to src/CDGraetzIsothermalSolution/data/.
// ---------------------------------------------------------------------------

// Evaluate f(β) = psinm_r(0, β, 1)  (the Dirichlet residual at r = 1).
static long double graetz_dirichlet_residual(long double beta)
{
    return psinm_r<long double>(0u, beta, static_cast<long double>(1.0L));
}

// Bisect [a, b] to find a root of graetz_dirichlet_residual to the given tol.
static long double bisect_graetz_root(long double a, long double b, long double tol)
{
    long double fa = graetz_dirichlet_residual(a);
    long double fb = graetz_dirichlet_residual(b);
    if (fa * fb > 0.0L)
    {
        std::cerr << "bisect_graetz_root: no sign change in [" << a << ", " << b << "]\n";
        return (a + b) / 2.0L;
    }
    for (int iter = 0; iter < 200; ++iter)
    {
        long double mid = (a + b) / 2.0L;
        if ((b - a) < tol) return mid;
        long double fm = graetz_dirichlet_residual(mid);
        if (fa * fm <= 0.0L) { b = mid; fb = fm; }
        else                  { a = mid; fa = fm; }
    }
    return (a + b) / 2.0L;
}

void write_graetz_roots_and_norms()
{
    const unsigned n_roots = 1000;  // number of eigenvalues to generate
    const long double scan_step = 0.1L;
    const long double scan_end  = 4200.0L; // large enough for 1000 roots (~4 apart)
    const long double bisect_tol = 1e-18L;

    std::cout << "Computing Graetz Dirichlet eigenvalues...\n";

    std::vector<long double> graetz_roots;
    graetz_roots.reserve(n_roots);

    long double beta  = scan_step;
    long double f_cur = graetz_dirichlet_residual(beta);

    while (beta < scan_end && graetz_roots.size() < n_roots)
    {
        long double beta_next  = beta + scan_step;
        long double f_next = graetz_dirichlet_residual(beta_next);

        if (f_cur * f_next < 0.0L)
        {
            // Sign change → bracket and refine
            long double root = bisect_graetz_root(beta, beta_next, bisect_tol);
            graetz_roots.push_back(root);
            std::cout << "  β_" << graetz_roots.size() << " = " << root << "\n";
        }

        beta  = beta_next;
        f_cur = f_next;
    }

    if (graetz_roots.size() < n_roots)
    {
        std::cerr << "Warning: only found " << graetz_roots.size()
                  << " Graetz roots (wanted " << n_roots << ").\n";
    }

    // --- Compute norms using the Gaussian quadrature from CDBaseSolution data ---
    const std::filesystem::path gauss_file =
        "./src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_300.txt";

    std::vector<std::vector<long double>> gauss_data;
    readValuesFromFile(gauss_file.string(), gauss_data);

    std::vector<long double> weights(gauss_data.size()), nodes(gauss_data.size());
    for (size_t i = 0; i < gauss_data.size(); ++i)
    {
        weights[i] = gauss_data[i][0];
        nodes[i]   = gauss_data[i][1];
    }

    std::cout << "Computing Graetz norms...\n";
    std::vector<long double> graetz_norms;
    graetz_norms.reserve(graetz_roots.size());
    for (long double root : graetz_roots)
    {
        long double norm = get_norm<long double>(0u, root, weights, nodes);
        graetz_norms.push_back(norm);
    }

    // --- Compute coefficients c_{0m}^S = sqrt(2pi) * C_m via closed-form Belhocine Eq. 56 ---
    // The numerator ∫₀¹ r(1-r²) G_m(r) dr = (1/2 - 1/β_m) exp(-β_m/2) 1F1(3/2-β_m/4, 2; β_m).
    // Stored as c_{0m}^S to match the angular-factor convention of CDBaseSolution.
    const long double sqrt2pi = std::sqrt(2.0L * M_PIl);
    std::cout << "Computing Graetz coefficients...\n";
    std::vector<long double> graetz_coeffs;
    graetz_coeffs.reserve(graetz_roots.size());
    for (size_t i = 0; i < graetz_roots.size(); ++i)
    {
        const long double beta = graetz_roots[i];
        long double f_val;
        hypergeometric1F1(
            static_cast<long double>(1.5L) - beta / static_cast<long double>(4.0L),
            2u,
            beta,
            f_val);
        const long double numerator =
            (0.5L - 1.0L / beta) * std::exp(-beta / 2.0L) * f_val;
        graetz_coeffs.push_back(sqrt2pi * numerator / graetz_norms[i]);
    }

    // --- Write to files (1 row, n_roots columns) ---
    // These are the full-resolution tables (currently 1000 modes); the solver
    // itself loads the truncated tables produced by write_graetz_truncated_tables().
    const std::string out_dir = "./src/CDGraetzIsothermalSolution/data/";
    const std::string roots_file  = out_dir + "graetz_roots_1000.txt";
    const std::string norms_file  = out_dir + "graetz_norms_1000.txt";
    const std::string coeffs_file = out_dir + "graetz_coefficients_1000.txt";

    // readValuesFromFile/writeValuesToFile use 2D vectors (rows × cols).
    // For Graetz (n=0 only): single row.
    std::vector<std::vector<long double>> roots_2d(1, graetz_roots);
    std::vector<std::vector<long double>> norms_2d(1, graetz_norms);
    std::vector<std::vector<long double>> coeffs_2d(1, graetz_coeffs);

    writeValuesToFile(roots_file,  roots_2d);
    writeValuesToFile(norms_file,  norms_2d);
    writeValuesToFile(coeffs_file, coeffs_2d);

    std::cout << "Graetz data written to:\n"
              << "  " << roots_file  << "\n"
              << "  " << norms_file  << "\n"
              << "  " << coeffs_file << "\n";
}

// Truncates the full-resolution graetz_{roots,norms,coefficients}_1000.txt tables
// to the first n_modes columns and writes them as the active graetz_*.txt files
// loaded by CDGraetzIsothermalSolution. Each column is an independent per-mode
// value (root, its norm, its projection coefficient), so truncation is exact —
// no recomputation needed. Run after write_graetz_roots_and_norms().
//
// n_modes=400 was chosen because the pseudo-products matrix (needed to fix the
// modal coefficients) is a dense n_modes×n_modes table whose entries
// involve confluent hypergeometric evaluations that become very expensive for
// large-argument pairs; 1000×1000 was computationally infeasible (>>hours),
// while 400×400 is tractable and covers the modes used in practice.
void write_graetz_truncated_tables(unsigned n_modes)
{
    const std::string in_dir  = "./src/CDGraetzIsothermalSolution/data/";
    const std::string out_dir = "./src/CDGraetzIsothermalSolution/data/";

    auto truncate_one = [&](const std::string &in_name, const std::string &out_name)
    {
        std::vector<std::vector<long double>> data;
        readValuesFromFile(in_dir + in_name, data);
        if (data.empty() || data[0].size() < n_modes)
        {
            std::cerr << "ERROR: " << in_name << " has fewer than " << n_modes << " modes.\n";
            exit(1);
        }
        std::vector<std::vector<long double>> truncated(1, std::vector<long double>(
            data[0].begin(), data[0].begin() + n_modes));
        writeValuesToFile(out_dir + out_name, truncated);
        std::cout << "Written " << out_dir + out_name << " (" << n_modes << " modes)\n";
    };

    truncate_one("graetz_roots_1000.txt",        "graetz_roots.txt");
    truncate_one("graetz_norms_1000.txt",        "graetz_norms.txt");
    truncate_one("graetz_coefficients_1000.txt", "graetz_coefficients.txt");
}

void write_graetz_pseudo_products_matrix()
{
    const std::filesystem::path roots_path =
        "./src/CDGraetzIsothermalSolution/data/graetz_roots.txt";
    // Pseudo-products = ∫₀¹ ψ₁ψ₂ r dr → Bessel quadrature (weight r), same formula as Stratified.
    const std::filesystem::path gauss_path =
        "./src/CDBaseSolution/data/gaussian_weights_eigenvalues_bessel_n_300.txt";
    const std::filesystem::path out_path =
        "./src/CDGraetzIsothermalSolution/data/graetz_pseudo_products_matrix.txt";

    // Write as double so both CDGraetzIsothermalSolution<double> and <long double>
    // can load it (the reader widens double→long double but not the reverse).
    std::cout << "Computing Graetz pseudo-products matrix...\n";
    compute_and_save_pseudo_products_matrix<double>(roots_path, gauss_path, out_path);
    std::cout << "Written to " << out_path << "\n";
}

void write_stratified_pseudo_products_matrix()
{
    const std::filesystem::path roots_path =
        "./src/CDStratifiedSolution/data/roots_100x100.txt";
    // Pseudo-products = ∫₀¹ ψ₁ψ₂ r dr → Bessel quadrature (weight r), same formula as Graetz.
    const std::filesystem::path gauss_path =
        "./src/CDBaseSolution/data/gaussian_weights_eigenvalues_bessel_n_300.txt";
    const std::filesystem::path out_path =
        "./src/CDStratifiedSolution/data/pseudo_products_matrix_100x100x100.txt";

    std::cout << "Computing Stratified pseudo-products matrix...\n";
    compute_and_save_pseudo_products_matrix<double>(roots_path, gauss_path, out_path);
    std::cout << "Written to " << out_path << "\n";
}

// Regenerates the bare (Neumann-wall) root/norm tables for CDStratifiedSolution.
// Roots are the zeros of dpsi/dr(r=1; n, b); within a row the (m-1)-th root seeds
// the m-th one (spacing tends to 4), so the rows are mutually independent and run
// in parallel. Norms are the weighted L2 norms N^2 = int_0^1 psi^2 (1-r^2) r dr,
// i.e. the hypergeometric Gauss rule, which is what CDS_NORMS_FILE stores.
// set_series_data() rejects roots/norms tables of different shapes, so both are
// written even when only the roots are wanted.
void write_stratified_roots_and_norms()
{
    const unsigned max_n = 399;
    const unsigned max_m = 399;

    const long double tol = 1e-12L;
    const long double rel_tol = 1e-12L;
    const unsigned max_iters = 10000;
    const bool use_position_convergence = true;
    const bool use_residual_convergence = true;

    const std::string out_dir = "./src/CDStratifiedSolution/data/";
    const std::string roots_name = "roots_" + std::to_string(max_n + 1) + "x" + std::to_string(max_m + 1) + ".txt";
    const std::string norms_name = "norms_" + std::to_string(max_n + 1) + "x" + std::to_string(max_m + 1) + "_cpp.txt";

    std::vector<std::vector<long double>> gaussian_data;
    readValuesFromFile("./src/CDBaseSolution/data/gaussian_weights_eigenvalues_hypergeometric_n_300.txt", gaussian_data);
    std::vector<long double> weights;
    std::vector<long double> nodes;
    weights.reserve(gaussian_data.size());
    nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data)
    {
        weights.push_back(row[0]);
        nodes.push_back(row[1]);
    }

    std::vector<std::vector<long double>> roots(max_n + 1, std::vector<long double>(max_m + 1, 0.0L));
    std::vector<std::vector<long double>> norms(max_n + 1, std::vector<long double>(max_m + 1, 0.0L));

    std::cout << "Computing " << (max_n + 1) << "x" << (max_m + 1) << " stratified roots and norms..." << std::endl;
    unsigned rows_done = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int n_signed = 0; n_signed <= static_cast<int>(max_n); ++n_signed)
    {
        const unsigned n = static_cast<unsigned>(n_signed);

        // Seeds reproduce the serial generator in find_roots_and_norms: the (0,0)
        // constant mode is analytic and deliberately leaves the (0,1) guess at
        // 1.06 + 4, which is the only value that brackets the 5.0675 root.
        long double previous_root = 1.06L;
        long double previous_norm = 1.0L;

        for (unsigned m = 0; m <= max_m; ++m)
        {
            if (n == 0 && m == 0)
            {
                roots[0][0] = 0.0L;
                norms[0][0] = get_norm(0u, 0.0L, weights, nodes);   // = 1/4 for R_00 == 1
                continue;
            }

            long double approx_root;
            long double norm_factor;
            if (m == 0)
            {
                approx_root = 2.0L * (static_cast<long double>(n) + 1.0L)
                            - 1.1L * expl(-0.25L * (static_cast<long double>(n) - 1.0L));
                // find_root divides the residual by norm_factor purely to keep its
                // "is the derivative big enough for Newton" test meaningful; without
                // a sane scale the m=0 modes stagnate at the initial guess. The
                // serial generator carried norms[n-1][max_m] across the row break,
                // which is both a cross-row dependency and the wrong magnitude.
                // exp(-b/2) is the eigenfunction's own scale at the wall.
                norm_factor = expl(-0.5L * approx_root);
            }
            else
            {
                approx_root = previous_root + 4.0L;
                norm_factor = sqrtl(previous_norm);
            }

            const long double root = find_root(n, approx_root, tol, rel_tol, norm_factor,
                                               max_iters, use_position_convergence, use_residual_convergence);
            const long double norm = get_norm(n, root, weights, nodes);

            roots[n][m] = root;
            norms[n][m] = norm;

            if (m > 0 && std::fabs(root - previous_root) > 4.5L)
            {
                std::cerr << "ERROR: root[" << n << "][" << m << "] = " << root
                          << " is more than 4.5 away from root[" << n << "][" << m - 1 << "] = "
                          << previous_root << "; a root was probably skipped." << std::endl;
                exit(1);
            }

            previous_root = root;
            previous_norm = norm;
        }

        #pragma omp critical
        {
            ++rows_done;
            std::cout << "\r  rows done: " << rows_done << " / " << (max_n + 1) << std::flush;
        }
    }
    std::cout << std::endl;

    auto dump = [](const std::string &path, const std::vector<std::vector<long double>> &table) {
        std::ofstream out(path);
        out << std::setprecision(std::numeric_limits<long double>::max_digits10) << std::scientific;
        for (const auto &row : table)
        {
            for (const auto &val : row) out << val << " ";
            out << "\n";
        }
        std::cout << "Written " << path << std::endl;
    };

    dump(out_dir + roots_name, roots);
    dump(out_dir + norms_name, norms);
}

int main()
{
    // write_graetz_roots_and_norms();       // regenerates graetz_*_1000.txt from scratch
    // write_graetz_truncated_tables(400);   // truncates _1000 tables to the active graetz_*.txt
    // write_graetz_pseudo_products_matrix();   // rebuilds graetz_pseudo_products_matrix.txt from active roots
    // write_stratified_pseudo_products_matrix();
    // write_stratified_roots_and_norms();  // regenerates roots_400x400.txt + norms_400x400_cpp.txt
    return 0;
}
