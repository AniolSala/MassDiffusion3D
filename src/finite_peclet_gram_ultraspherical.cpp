// Ultraspherical (banded Jacobi-pencil) finite-Peclet Gram backend.
//
// STATUS: not yet implemented. The real backend (plans/IMPLEMENTATION_PLAN_GJQR_ultraspherical.md
// section 6) is a self-adjoint banded pencil in the shifted-Jacobi coefficient space: a domain
// basis P_j^(1,n), a range basis P_j^(3,n+2), banded differentiation/conversion/multiplication
// operators A0/B1/B2 built by exact Gauss-Jacobi projection, wall-dependent boundary and
// normalization rows, and the Lambda=0 constant-mode special case. None of that machinery exists
// here yet.
//
// A previous revision of this file silently forwarded every call to the independent
// GaussJacobiQR backend (finite_peclet_gram_gauss_jacobi.*), which produced numerically identical
// output under a different backend name. That both violates the project requirement that the two
// backends share no code and, more importantly, is dishonest: selecting GramMethod::Ultraspherical
// must never silently compute GaussJacobiQR's numbers. Until the real pencil is implemented, every
// entry point throws so the failure is loud and immediate, with no fallback to the other backend
// (setup_fp_solution's catch(...) leaves the solution object Uninitialized -- see
// CDBaseSolution::setup_fp_solution).

#include "finite_peclet_gram_ultraspherical.h"

#include <stdexcept>

namespace
{
[[noreturn]] void not_implemented()
{
    throw std::logic_error(
        "GramMethod::Ultraspherical is not yet implemented (plans/IMPLEMENTATION_PLAN_GJQR_ultraspherical.md "
        "section 6). Select GramMethod::GaussJacobiQR (the default) instead.");
}

// Private mirror of fp_gram_node_count (finite_peclet_gram_gauss_jacobi.h) --
// see that header's comment for the measured justification. Kept as its own
// copy (not shared with the GaussJacobiQR backend) so the two backends stay
// code-independent per the project requirement; not yet called anywhere
// because the real pencil (plans/IMPLEMENTATION_PLAN_GJQR_ultraspherical.md
// section 6) has not landed, but the node-count formula it will need is
// pinned here now so the eventual implementation does not have to re-derive
// it under time pressure.
template <class T>
unsigned fp_gram_node_count_ultraspherical(unsigned mode_count, const FPGramUltrasphericalOptions<T> &options)
{
    if (!mode_count)
        throw std::invalid_argument("fp_gram_node_count_ultraspherical: empty block");
    if (!options.ultraspherical_factor)
        throw std::invalid_argument("fp_gram_node_count_ultraspherical: ultraspherical factor must be positive");
    const unsigned affine = options.ultraspherical_factor * mode_count + options.ultraspherical_margin;
    const unsigned floor_ = options.minimum_factor * mode_count;
    const unsigned N = affine > floor_ ? affine : floor_;
    if (N < mode_count)
        throw std::runtime_error("fp_gram_node_count_ultraspherical: node count below mode count");
    return N;
}
}

template <typename T>
FPGramUltrasphericalFactor<T> fp_gram_factor_ultraspherical(unsigned, const std::vector<SeriesTermData<T>> &, WallCondition, const FPGramUltrasphericalOptions<T> &)
{
    not_implemented();
}

template <typename T>
void fp_gram_solve_ultraspherical(const FPGramUltrasphericalFactor<T> &, const std::vector<T> &, std::vector<T> &)
{
    not_implemented();
}

template <typename T>
void fp_gram_reconstruct_ultraspherical(const FPGramUltrasphericalFactor<T> &, std::vector<std::vector<T>> &)
{
    not_implemented();
}

template FPGramUltrasphericalFactor<double> fp_gram_factor_ultraspherical(unsigned, const std::vector<SeriesTermData<double>> &, WallCondition, const FPGramUltrasphericalOptions<double> &);
template FPGramUltrasphericalFactor<long double> fp_gram_factor_ultraspherical(unsigned, const std::vector<SeriesTermData<long double>> &, WallCondition, const FPGramUltrasphericalOptions<long double> &);
template void fp_gram_solve_ultraspherical(const FPGramUltrasphericalFactor<double> &, const std::vector<double> &, std::vector<double> &);
template void fp_gram_solve_ultraspherical(const FPGramUltrasphericalFactor<long double> &, const std::vector<long double> &, std::vector<long double> &);
template void fp_gram_reconstruct_ultraspherical(const FPGramUltrasphericalFactor<double> &, std::vector<std::vector<double>> &);
template void fp_gram_reconstruct_ultraspherical(const FPGramUltrasphericalFactor<long double> &, std::vector<std::vector<long double>> &);
