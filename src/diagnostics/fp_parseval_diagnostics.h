#ifndef FP_PARSEVAL_DIAGNOSTICS_H
#define FP_PARSEVAL_DIAGNOSTICS_H

// ---------------------------------------------------------------------------
// fp_parseval_diagnostics — batch assembly of the two spectral quantities that
// eq. (gram_parseval) contracts against the inlet coefficients:
//
//     Nfp^2_nm = int_0^1 [ omega + 2 kappa Lam_nm ] R_nm^2 r dr        (norm_fp)
//     U^n_mk   = int_0^1                            R_nm R_nk r dr     (U_nm_def)
//
// so that a caller checking the identity
//
//     sum_m Nfp^2_nm c_m^2 - 2 kappa c^T (Lam U^n) c  =  || sum_m c_m R_nm ||^2_omega
//
// never has to re-implement the quadrature, the modes, or either integral.
//
// NOTHING HERE IS ON ANY SOLUTION PATH. This header is included by
// src/pybind.cpp alone, and CMakeLists.txt removes src/pybind.cpp from
// CORE_SOURCES, so the `main` and `cpp_base_tests` binaries never compile it.
// It defines new functions and modifies none: every integral below is delegated
// to the same kernels the solver itself uses --
//
//     fp_build_radial_table          (finite_peclet_radial_block.h)
//     fp_generalized_norm_from_row   (finite_peclet_norms.h)
//     fp_unweighted_overlap_from_rows(finite_peclet_norms.h)
//
// -- on the same composite Gauss-Legendre rule (fp_make_quadrature_rule), so a
// diagnostic can never drift from the solution it is meant to audit.
//
// The per-mode entry points fp_generalized_norm / fp_unweighted_overlap remain
// bound separately and rebuild their own rule per call; they are the independent
// reference this batch assembly is checked against, not a duplicate of it.
// ---------------------------------------------------------------------------

#include <cmath>
#include <stdexcept>
#include <vector>

#include "../series_term_struct.h"
#include "../finite_peclet_radial_block.h"
#include "../finite_peclet_norms.h"

/** Spectral data of one retained mode set, grouped into angular blocks. */
template <typename Ttype>
struct FPSpectralAssembly
{
    // Generalised norm of every mode, in the caller's input order.
    std::vector<Ttype> generalized_norm;

    // One entry per angular block, in the order fp_build_radial_table grouped them.
    std::vector<unsigned> block_angular_index;
    std::vector<std::vector<unsigned>> block_modes;   // indices into the input arrays

    // Row-major K_n x K_n overlap of each block, NORMALISED by the generalised
    // norms: Uhat_mk = U_mk / (Nfp_m Nfp_k). Nfp^2 spans ~160 decades across a
    // retained set at beta_max = 600, so the raw matrix cannot be contracted with
    // the (reciprocally scaled) coefficients without overflowing. Uhat is bounded
    // by 1 in modulus, and the matching contraction uses c_m * Nfp_m.
    std::vector<std::vector<Ttype>> block_overlap;
};

/**
 * Assemble Nfp^2 and the per-block normalised overlaps for a retained mode set.
 *
 * @param angular_index n of each mode; modes must be grouped by n, strictly
 *                      increasing in m within a block (the order the solver's
 *                      m_series_data already has).
 * @param radial_index  m of each mode.
 * @param root_fp       b = sqrt(Lam)  (SeriesTermData::root_fp).
 * @param btilde_fp     bt = b (1 + kappa Lam)  (SeriesTermData::btilde_fp).
 * @param rate_fp       Lam  (SeriesTermData::rate_fp).
 * @param kappa         Pe^-2.
 * @param panels        panels of the composite 16-point Gauss-Legendre rule.
 */
template <typename Ttype>
FPSpectralAssembly<Ttype> fp_assemble_spectral_data(
    const std::vector<unsigned> &angular_index,
    const std::vector<unsigned> &radial_index,
    const std::vector<Ttype> &root_fp,
    const std::vector<Ttype> &btilde_fp,
    const std::vector<Ttype> &rate_fp,
    const Ttype &kappa,
    unsigned panels)
{
    const std::size_t mode_count = angular_index.size();
    if (mode_count == 0)
        throw std::invalid_argument("fp_assemble_spectral_data: empty mode set");
    if (radial_index.size() != mode_count || root_fp.size() != mode_count ||
        btilde_fp.size() != mode_count || rate_fp.size() != mode_count)
        throw std::invalid_argument("fp_assemble_spectral_data: mode arrays differ in length");

    std::vector<SeriesTermData<Ttype>> series(mode_count);
    for (std::size_t k = 0; k < mode_count; ++k)
    {
        series[k].n = angular_index[k];
        series[k].m = radial_index[k];
        series[k].root_fp = root_fp[k];
        series[k].btilde_fp = btilde_fp[k];
        series[k].rate_fp = rate_fp[k];
    }

    // Samples every retained mode on the shared rule, blocked by n and
    // parallelised over modes. Validates finiteness and mode ordering itself.
    const FPRadialTable<Ttype> table =
        fp_build_radial_table<Ttype>(series, static_cast<unsigned>(mode_count), panels);

    FPSpectralAssembly<Ttype> assembly;
    assembly.generalized_norm.assign(mode_count, static_cast<Ttype>(0));
    assembly.block_angular_index.reserve(table.block_count());
    assembly.block_modes.reserve(table.block_count());
    assembly.block_overlap.reserve(table.block_count());

    for (const auto &block : table.blocks)
    {
        const std::size_t block_modes = block.mode_count();
        assembly.block_angular_index.push_back(block.n);
        assembly.block_modes.push_back(block.flat_indices);

        std::vector<Ttype> inverse_norm(block_modes);
        for (std::size_t i = 0; i < block_modes; ++i)
        {
            const std::size_t k = block.flat_indices[i];
            const Ttype value = fp_generalized_norm_from_row(
                table.quadrature, block.row_data(i), kappa, series[k].rate_fp);
            if (!(value > static_cast<Ttype>(0)) || !std::isfinite(static_cast<long double>(value)))
                throw std::runtime_error("fp_assemble_spectral_data: non-positive generalised norm");
            assembly.generalized_norm[k] = value;
            inverse_norm[i] = static_cast<Ttype>(1) / std::sqrt(value);
        }

        std::vector<Ttype> overlap(block_modes * block_modes);
        for (std::size_t i = 0; i < block_modes; ++i)
            for (std::size_t j = i; j < block_modes; ++j)
            {
                const Ttype scaled = fp_unweighted_overlap_from_rows(
                    table.quadrature, block.row_data(i), block.row_data(j)) *
                    inverse_norm[i] * inverse_norm[j];
                overlap[i * block_modes + j] = scaled;
                overlap[j * block_modes + i] = scaled;
            }
        assembly.block_overlap.push_back(std::move(overlap));
    }
    return assembly;
}

#endif // FP_PARSEVAL_DIAGNOSTICS_H
