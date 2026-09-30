// ---------------------------------------------------------------------------
// finite_peclet_norms.cpp — implementation of the generalised norm and radial
// overlaps. See finite_peclet_norms.h and theory Sec. 5.3 (norm_fp, U_nm_def).
// ---------------------------------------------------------------------------

#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

#include "finite_peclet_norms.h"
#include "finite_peclet_radial.h"
#include "finite_peclet_quadrature.h"

template <typename Ttype>
Ttype fp_generalized_norm(const unsigned &n, const Ttype &b, const Ttype &bt,
                          const Ttype &kappa, const Ttype &Lam, unsigned panels)
{
    const auto quadrature = fp_make_quadrature_rule<Ttype>(panels);
    std::vector<Ttype> row(quadrature.size());
    for (size_t q = 0; q < row.size(); ++q)
        row[q] = psinm_r_fp(n, b, bt, quadrature.nodes[q]);
    return fp_generalized_norm_from_row(quadrature, row.data(), kappa, Lam);
}

template <typename Ttype>
Ttype fp_unweighted_overlap(const unsigned &n,
                            const Ttype &bi, const Ttype &bti,
                            const Ttype &bj, const Ttype &btj,
                            unsigned panels)
{
    const auto quadrature = fp_make_quadrature_rule<Ttype>(panels);
    std::vector<Ttype> row_i(quadrature.size()), row_j(quadrature.size());
    for (size_t q = 0; q < quadrature.size(); ++q)
    {
        row_i[q] = psinm_r_fp(n, bi, bti, quadrature.nodes[q]);
        row_j[q] = psinm_r_fp(n, bj, btj, quadrature.nodes[q]);
    }
    return fp_unweighted_overlap_from_rows(quadrature, row_i.data(), row_j.data());
}

template <typename Ttype>
Ttype fp_generalized_norm_from_row(const FPQuadratureRule<Ttype> &quadrature,
                                   const Ttype *radial_row, const Ttype &kappa,
                                   const Ttype &lambda)
{
    const Ttype extra = static_cast<Ttype>(2) * kappa * lambda;
    Ttype total = static_cast<Ttype>(0);
    for (size_t q = 0; q < quadrature.size(); ++q)
        total += (quadrature.omega_measure[q] + extra * quadrature.unweighted_measure[q]) *
                 radial_row[q] * radial_row[q];
    return total;
}

template <typename Ttype>
Ttype fp_unweighted_overlap_from_rows(const FPQuadratureRule<Ttype> &quadrature,
                                      const Ttype *row_i, const Ttype *row_j)
{
    Ttype total = static_cast<Ttype>(0);
    for (size_t q = 0; q < quadrature.size(); ++q)
        total += quadrature.unweighted_measure[q] * row_i[q] * row_j[q];
    return total;
}

template <typename Ttype>
void fp_compute_norms_from_table(std::vector<SeriesTermData<Ttype>> &series_data,
                                 unsigned max_K, const Ttype &kappa,
                                 const FPRadialTable<Ttype> &table)
{
    if (max_K != table.total_modes || max_K > series_data.size())
        throw std::invalid_argument("fp_compute_norms_from_table: incompatible table");
    std::vector<bool> written(max_K, false);
    for (const auto &block : table.blocks)
        for (std::size_t local = 0; local < block.mode_count(); ++local)
        {
            const unsigned k = block.flat_indices[local];
            if (k >= max_K || written[k]) throw std::runtime_error("fp_compute_norms_from_table: duplicate mode");
            series_data[k].norm_fp = fp_generalized_norm_from_row(
                table.quadrature, block.row_data(local), kappa, series_data[k].rate_fp);
            if (!(series_data[k].norm_fp > static_cast<Ttype>(0)) ||
                !std::isfinite(static_cast<long double>(series_data[k].norm_fp)))
                throw std::runtime_error("fp_compute_norms_from_table: invalid norm");
            written[k] = true;
        }
    for (bool value : written) if (!value) throw std::runtime_error("fp_compute_norms_from_table: unwritten mode");
}

template <typename Ttype>
void fp_compute_norms(std::vector<SeriesTermData<Ttype>> &series_data, unsigned max_K,
                      const Ttype &kappa, unsigned panels)
{
#pragma omp parallel for schedule(dynamic)
    for (unsigned k = 0; k < max_K; ++k)
    {
        SeriesTermData<Ttype> &t = series_data[k];
        t.norm_fp = fp_generalized_norm(t.n, t.root_fp, t.btilde_fp, kappa, t.rate_fp, panels);
    }
}

// --- Explicit instantiations -------------------------------------------------
template void fp_compute_norms<double>(std::vector<SeriesTermData<double>> &, unsigned, const double &, unsigned);
template void fp_compute_norms<long double>(std::vector<SeriesTermData<long double>> &, unsigned, const long double &, unsigned);
template double      fp_generalized_norm<double>(const unsigned &, const double &, const double &, const double &, const double &, unsigned);
template long double fp_generalized_norm<long double>(const unsigned &, const long double &, const long double &, const long double &, const long double &, unsigned);
template double      fp_unweighted_overlap<double>(const unsigned &, const double &, const double &, const double &, const double &, unsigned);
template long double fp_unweighted_overlap<long double>(const unsigned &, const long double &, const long double &, const long double &, const long double &, unsigned);
template double fp_generalized_norm_from_row<double>(const FPQuadratureRule<double> &, const double *, const double &, const double &);
template long double fp_generalized_norm_from_row<long double>(const FPQuadratureRule<long double> &, const long double *, const long double &, const long double &);
template double fp_unweighted_overlap_from_rows<double>(const FPQuadratureRule<double> &, const double *, const double *);
template long double fp_unweighted_overlap_from_rows<long double>(const FPQuadratureRule<long double> &, const long double *, const long double *);
template void fp_compute_norms_from_table<double>(std::vector<SeriesTermData<double>> &, unsigned, const double &, const FPRadialTable<double> &);
template void fp_compute_norms_from_table<long double>(std::vector<SeriesTermData<long double>> &, unsigned, const long double &, const FPRadialTable<long double> &);
