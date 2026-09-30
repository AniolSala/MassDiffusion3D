#include "tinytest.h"

#include "finite_peclet_radial_block.h"
#include "finite_peclet_quadrature.h"
#include "finite_peclet_radial.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

static SeriesTermData<double> radial_term(unsigned n, unsigned m, double b)
{
    SeriesTermData<double> term{};
    term.n = n; term.m = m; term.root_fp = b; term.rate_fp = b * b;
    term.btilde_fp = b;
    return term;
}

TEST_CASE(fp_radial_table_quadrature_invariants)
{
    bool threw = false;
    try { std::vector<double> r, w; fp_quad::gl_nodes<double>(0, r, w); }
    catch (const std::invalid_argument &) { threw = true; }
    REQUIRE(threw);
    auto rule = fp_make_quadrature_rule<double>(1);
    REQUIRE(rule.size() == 16);
    REQUIRE_APPROX(std::accumulate(rule.weights.begin(), rule.weights.end(), 0.0), 1.0, 1e-14, 1e-14);
    for (std::size_t q = 0; q < rule.size(); ++q) {
        REQUIRE(rule.nodes[q] > 0.0 && rule.nodes[q] < 1.0);
        REQUIRE(rule.weights[q] > 0.0);
        REQUIRE_APPROX(rule.unweighted_measure[q], rule.weights[q] * rule.nodes[q], 1e-15, 1e-15);
        REQUIRE_APPROX(rule.omega_measure[q], rule.weights[q] * rule.nodes[q] *
                       (1.0 - rule.nodes[q] * rule.nodes[q]), 1e-15, 1e-15);
    }
    auto large = fp_make_quadrature_rule<long double>(64);
    REQUIRE(large.size() == 1024);
}

TEST_CASE(fp_radial_table_groups_and_reuses_rows)
{
    std::vector<SeriesTermData<double>> terms = {
        radial_term(0, 0, 0.0), radial_term(0, 1, 2.4), radial_term(0, 2, 5.1),
        radial_term(1, 0, 1.7), radial_term(1, 1, 4.2), radial_term(3, 0, 3.3)};
    auto table = fp_build_radial_table(terms, static_cast<unsigned>(terms.size()), 1);
    REQUIRE(table.total_modes == terms.size());
    REQUIRE(table.block_count() == 3);
    REQUIRE(table.find_block(2) == nullptr);
    REQUIRE(table.find_block(0)->flat_indices == std::vector<unsigned>({0, 1, 2}));
    REQUIRE(table.find_block(1)->flat_indices == std::vector<unsigned>({3, 4}));
    REQUIRE(table.find_block(3)->flat_indices == std::vector<unsigned>({5}));
    const auto *zero = table.find_block(0)->row_data(0);
    for (std::size_t q = 0; q < table.quadrature.size(); ++q)
        REQUIRE_APPROX(zero[q], 1.0, 1e-14, 1e-14);
    for (const auto &block : table.blocks)
        for (std::size_t m = 0; m < block.mode_count(); ++m)
            for (std::size_t q = 0; q < block.node_count; ++q)
                REQUIRE_APPROX(block.at(m, q), psinm_r_fp(
                    terms[block.flat_indices[m]].n,
                    terms[block.flat_indices[m]].root_fp,
                    terms[block.flat_indices[m]].btilde_fp,
                    table.quadrature.nodes[q]), 1e-14, 1e-14);
}

TEST_CASE(fp_radial_table_empty_and_move_only)
{
    static_assert(std::is_move_constructible_v<FPRadialTable<double>>);
    static_assert(!std::is_copy_constructible_v<FPRadialTable<double>>);
    auto table = fp_build_radial_table<double>({}, 0, 0);
    REQUIRE(table.total_modes == 0);
    REQUIRE(table.blocks.empty());
}
