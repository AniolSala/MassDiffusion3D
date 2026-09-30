#ifndef FINITE_PECLET_GRAM_ULTRASPHERICAL_H
#define FINITE_PECLET_GRAM_ULTRASPHERICAL_H

#include <vector>
#include "series_term_struct.h"
#include "finite_peclet_roots.h"
#include "diagnostics/gram_diagnostics.h"

template <typename Ttype> struct FPGramUltrasphericalOptions {
    unsigned ultraspherical_factor = 2u;
    unsigned ultraspherical_margin = 20u;
    bool enable_order_check = false;
    // Mirrors FPGramGaussJacobiOptions::minimum_factor (finite_peclet_gram_gauss_jacobi.h)
    // so the two backends' options stay structurally parallel, even though this
    // backend does not implement the floor itself yet (see the .cpp for why).
    unsigned minimum_factor = 2u;
};
template <typename Ttype> struct FPGramUltrasphericalFactor {
    unsigned angular_index = 0, mode_count = 0, coefficient_count = 0;
    std::vector<Ttype> upper_triangular_R, column_scaling_D;
    std::vector<unsigned> pivot_permutation;
    GramBlockDiagnostics<Ttype> diagnostics;
};
template <typename Ttype> FPGramUltrasphericalFactor<Ttype> fp_gram_factor_ultraspherical(unsigned, const std::vector<SeriesTermData<Ttype>> &, WallCondition, const FPGramUltrasphericalOptions<Ttype> &);
template <typename Ttype> void fp_gram_solve_ultraspherical(const FPGramUltrasphericalFactor<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &);
template <typename Ttype> void fp_gram_reconstruct_ultraspherical(const FPGramUltrasphericalFactor<Ttype> &, std::vector<std::vector<Ttype>> &);

#endif
