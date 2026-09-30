#ifndef COEFFICIENTS_COMPUTATION_GAUSSIAN
#define COEFFICIENTS_COMPUTATION_GAUSSIAN

#include <vector>
#include <map>

#include "series_term_struct.h"

// template <typename Ttype>
// void compute_coefficients_matrix(
//     const std::vector<double> &,
//     const std::vector<double> &,
//     const std::vector<std::vector<Ttype>> &,
//     const std::vector<std::vector<Ttype>> &,
//     const std::vector<Ttype> &,
//     const std::vector<Ttype> &,
//     const unsigned &,
//     const std::vector<unsigned> &,
//     std::vector<std::vector<Ttype>> &);

template <typename Ttype>
void compute_coefficients_matrix(
    const std::vector<Ttype> &z_vec,
    const std::vector<Ttype> &values,
    const std::vector<Ttype> &gauss_weights,
    const std::vector<Ttype> &gauss_nodes,
    const unsigned &max_N,
    std::vector<SeriesTermData<Ttype>>&);

template <typename Ttype>
void compute_coefficient_nm(
    const unsigned &,
    const unsigned &,
    const std::vector<Ttype> &,
    const std::vector<Ttype> &,
    const Ttype &,
    const std::vector<Ttype> &,
    const std::vector<Ttype> &,
    Ttype &);

#endif