#ifndef FINITE_PECLET_RADIAL_BLOCK_H
#define FINITE_PECLET_RADIAL_BLOCK_H

#include <cstddef>
#include <vector>

#include "series_term_struct.h"

template <typename Ttype>
struct FPQuadratureRule
{
    unsigned panels = 0;
    std::vector<Ttype> nodes;
    std::vector<Ttype> weights;
    std::vector<Ttype> unweighted_measure;
    std::vector<Ttype> omega_measure;

    std::size_t size() const noexcept { return nodes.size(); }
    std::size_t memory_bytes() const noexcept
    {
        return (nodes.capacity() + weights.capacity() +
                unweighted_measure.capacity() + omega_measure.capacity()) * sizeof(Ttype);
    }
};

template <typename Ttype>
struct FPRadialBlock
{
    unsigned n = 0;
    std::vector<unsigned> flat_indices;
    std::size_t node_count = 0;
    std::vector<Ttype> values;

    std::size_t mode_count() const noexcept { return flat_indices.size(); }
    bool empty() const noexcept { return flat_indices.empty(); }
    std::size_t memory_bytes() const noexcept
    { return flat_indices.capacity() * sizeof(unsigned) + values.capacity() * sizeof(Ttype); }

    const Ttype *row_data(std::size_t local_m) const;
    Ttype *row_data(std::size_t local_m);
    const Ttype &at(std::size_t local_m, std::size_t q) const;
    Ttype &at(std::size_t local_m, std::size_t q);
};

template <typename Ttype>
struct FPRadialTable
{
    FPQuadratureRule<Ttype> quadrature;
    std::vector<FPRadialBlock<Ttype>> blocks;
    unsigned total_modes = 0;

    FPRadialTable() = default;
    FPRadialTable(const FPRadialTable &) = delete;
    FPRadialTable &operator=(const FPRadialTable &) = delete;
    FPRadialTable(FPRadialTable &&) noexcept = default;
    FPRadialTable &operator=(FPRadialTable &&) noexcept = default;

    std::size_t block_count() const noexcept { return blocks.size(); }
    std::size_t memory_bytes() const noexcept
    {
        std::size_t result = quadrature.memory_bytes();
        for (const auto &block : blocks) result += block.memory_bytes();
        return result;
    }
    const FPRadialBlock<Ttype> *find_block(unsigned n) const noexcept;
    FPRadialBlock<Ttype> *find_block(unsigned n) noexcept;
};

template <typename Ttype>
FPQuadratureRule<Ttype> fp_make_quadrature_rule(unsigned panels);

template <typename Ttype>
FPRadialTable<Ttype> fp_build_radial_table(
    const std::vector<SeriesTermData<Ttype>> &series_data,
    unsigned max_K, unsigned panels);

#endif
