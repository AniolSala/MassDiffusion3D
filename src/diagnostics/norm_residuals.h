#include <cmath>
#include "comp_utils.h"
#include "io_data.h"
#include "math_functions.h"

#define GAUSS_WEIGHTS_NODES_FILE "gaussian_weights_eigenvalues_n_300.txt"
#define GAUSS_ROOTS_FILE "roots_100x100.txt"
#define CURRENT_DIR "."

namespace {

std::string data_path(const std::string &rel) {
    return std::string(CURRENT_DIR) + "/" + rel;
}

void psinm_r_prod(unsigned k, double b1, double b2, double r, double &val) {
    double psi_val_1 = 0.0L;
    double psi_val_2 = 0.0L;
    psinm_r(k, b1, r, psi_val_1);
    psinm_r(k, b2, r, psi_val_2);
    val = psi_val_1 * psi_val_2;
}

std::vector<double> get_residues_gaussian()
{
    std::vector<std::vector<double>> roots;
    std::vector<std::vector<double>> gaussian_data;

    readValuesFromFile(data_path("src/CDStratifiedSolution/data/" GAUSS_ROOTS_FILE), roots);
    readValuesFromFile(data_path("src/CDBaseSolution/data/" GAUSS_WEIGHTS_NODES_FILE), gaussian_data);

    std::vector<double> weights;
    std::vector<double> nodes;
    weights.reserve(gaussian_data.size());
    nodes.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data) {
        weights.push_back(row[0]);
        nodes.push_back(row[1]);
    }

    const unsigned n = 0;
    const unsigned roots_size = roots[n].size();

    // Compute the norms
    std::vector<double> computed_norms(roots_size);
    for (unsigned m = 0; m < roots_size; m++)
    {
        const double b = roots[n][m];
        double norm = 0.0;
        auto integrand = [b](double r, double &val) {
            psinm_r_prod(n, b, b, r, val);
        };
        gaussian_integration(integrand, weights, nodes, norm);
        computed_norms[m] = norm;
    }

    // Compute thep products
    std::vector<double> result;
    result.reserve(roots_size * roots_size);

    for (unsigned m1 = 0; m1 < roots_size; m1++)
    {
        const double b1 = roots[n][m1];
        for (unsigned m2 = 0; m2 < roots_size; m2++)
        {
            const double b2 = roots[n][m2];
            double integral = 0.0;
            auto integrand = [b1, b2](double r, double &val) {
                psinm_r_prod(n, b1, b2, r, val);
            };
            gaussian_integration(integrand, weights, nodes, integral);
            const double delta = (m1 == m2) ? 1.0 : 0.0;
            result.push_back(integral / std::sqrt((computed_norms[m1] * computed_norms[m2])) - delta);
        }
    }
    return result;
}

} // namespace