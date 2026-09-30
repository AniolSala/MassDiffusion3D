#include "finite_peclet_radial_block.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <omp.h>

#include "finite_peclet_quadrature.h"
#include "finite_peclet_radial.h"

template <typename Ttype>
const Ttype *FPRadialBlock<Ttype>::row_data(std::size_t local_m) const
{
    if (local_m >= mode_count()) throw std::out_of_range("FPRadialBlock: local mode out of range");
    return values.data() + local_m * node_count;
}

template <typename Ttype>
Ttype *FPRadialBlock<Ttype>::row_data(std::size_t local_m)
{
    return const_cast<Ttype *>(static_cast<const FPRadialBlock *>(this)->row_data(local_m));
}

template <typename Ttype>
const Ttype &FPRadialBlock<Ttype>::at(std::size_t local_m, std::size_t q) const
{
    if (local_m >= mode_count() || q >= node_count)
        throw std::out_of_range("FPRadialBlock: index out of range");
    return values[local_m * node_count + q];
}

template <typename Ttype>
Ttype &FPRadialBlock<Ttype>::at(std::size_t local_m, std::size_t q)
{
    return const_cast<Ttype &>(static_cast<const FPRadialBlock *>(this)->at(local_m, q));
}

template <typename Ttype>
const FPRadialBlock<Ttype> *FPRadialTable<Ttype>::find_block(unsigned angular_n) const noexcept
{
    for (const auto &block : blocks) if (block.n == angular_n) return &block;
    return nullptr;
}

template <typename Ttype>
FPRadialBlock<Ttype> *FPRadialTable<Ttype>::find_block(unsigned angular_n) noexcept
{
    for (auto &block : blocks) if (block.n == angular_n) return &block;
    return nullptr;
}

template <typename Ttype>
FPQuadratureRule<Ttype> fp_make_quadrature_rule(unsigned panels)
{
    if (panels == 0) throw std::invalid_argument("fp_make_quadrature_rule: panels must be positive");
    FPQuadratureRule<Ttype> rule;
    rule.panels = panels;
    fp_quad::gl_nodes(panels, rule.nodes, rule.weights);
    rule.unweighted_measure.resize(rule.nodes.size());
    rule.omega_measure.resize(rule.nodes.size());
    for (std::size_t q = 0; q < rule.nodes.size(); ++q)
    {
        const Ttype r = rule.nodes[q];
        rule.unweighted_measure[q] = rule.weights[q] * r;
        rule.omega_measure[q] = rule.unweighted_measure[q] * (static_cast<Ttype>(1) - r * r);
    }
    return rule;
}

template <typename Ttype>
FPRadialTable<Ttype> fp_build_radial_table(
    const std::vector<SeriesTermData<Ttype>> &series_data,
    unsigned max_K, unsigned panels)
{
    if (max_K > series_data.size())
        throw std::invalid_argument("fp_build_radial_table: max_K exceeds series_data size");
    FPRadialTable<Ttype> table;
    if (max_K == 0) return table;
    table.quadrature = fp_make_quadrature_rule<Ttype>(panels);
    table.total_modes = max_K;

    std::map<unsigned, std::size_t> block_positions;
    for (unsigned k = 0; k < max_K; ++k)
    {
        const auto &term = series_data[k];
        if (!std::isfinite(static_cast<long double>(term.root_fp)) ||
            !std::isfinite(static_cast<long double>(term.btilde_fp)) ||
            !std::isfinite(static_cast<long double>(term.rate_fp)) ||
            term.root_fp < static_cast<Ttype>(0) || term.rate_fp < static_cast<Ttype>(0))
            throw std::invalid_argument("fp_build_radial_table: invalid finite-Peclet mode data");

        auto position = block_positions.find(term.n);
        if (position == block_positions.end())
        {
            FPRadialBlock<Ttype> block;
            block.n = term.n;
            table.blocks.push_back(std::move(block));
            position = block_positions.emplace(term.n, table.blocks.size() - 1).first;
        }
        auto &block = table.blocks[position->second];
        if (!block.flat_indices.empty())
        {
            const auto &previous = series_data[block.flat_indices.back()];
            if (term.m <= previous.m)
                throw std::invalid_argument("fp_build_radial_table: modes are not strictly ordered");
        }
        block.flat_indices.push_back(k);
    }

    // The input is expected to be grouped by n. Check duplicate (n,m) records
    // even when malformed input changes the grouping order.
    for (std::size_t i = 0; i < max_K; ++i)
        for (std::size_t j = i + 1; j < max_K; ++j)
            if (series_data[i].n == series_data[j].n && series_data[i].m == series_data[j].m)
                throw std::invalid_argument("fp_build_radial_table: duplicate (n,m) record");

    const std::size_t Q = table.quadrature.size();
    std::exception_ptr error;
    for (auto &block : table.blocks)
    {
        block.node_count = Q;
        block.values.resize(block.mode_count() * Q);
#pragma omp parallel for schedule(dynamic)
        for (long long local = 0; local < static_cast<long long>(block.mode_count()); ++local)
        {
            try
            {
                const auto &term = series_data[block.flat_indices[static_cast<std::size_t>(local)]];
                Ttype *row = block.values.data() + static_cast<std::size_t>(local) * Q;
                for (std::size_t q = 0; q < Q; ++q)
                {
                    row[q] = psinm_r_fp(term.n, term.root_fp, term.btilde_fp,
                                        table.quadrature.nodes[q]);
                    if (!std::isfinite(static_cast<long double>(row[q])))
                        throw std::runtime_error("fp_build_radial_table: non-finite radial value");
                }
            }
            catch (...)
            {
#pragma omp critical
                { if (!error) error = std::current_exception(); }
            }
        }
        if (error) std::rethrow_exception(error);
    }
    return table;
}

template struct FPRadialBlock<double>;
template struct FPRadialBlock<long double>;
template struct FPRadialTable<double>;
template struct FPRadialTable<long double>;
template FPQuadratureRule<double> fp_make_quadrature_rule<double>(unsigned);
template FPQuadratureRule<long double> fp_make_quadrature_rule<long double>(unsigned);
template FPRadialTable<double> fp_build_radial_table<double>(const std::vector<SeriesTermData<double>> &, unsigned, unsigned);
template FPRadialTable<long double> fp_build_radial_table<long double>(const std::vector<SeriesTermData<long double>> &, unsigned, unsigned);
