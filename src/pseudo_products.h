#ifndef PSEUDO_PRODUCTS_H
#define PSEUDO_PRODUCTS_H

// I/O and computation of the pseudo-products tensor
//   P_{n,m1,m2} = int_0^1 R_{n,m1}(r) R_{n,m2}(r) r dr,
// the precomputed triple-integral table used when fixing the modal coefficients.

#include <filesystem>
#include <vector>

template <typename Ttype>
void read_pseudo_products_matrix(
    const std::filesystem::path &,
    std::vector<std::vector<std::vector<Ttype>>>&);

template <typename Ttype>
void save_pseudo_products_matrix(
    const std::filesystem::path &,
    const std::vector<std::vector<std::vector<Ttype>>>&);

template <typename Ttype>
void compute_and_save_pseudo_products_matrix(
    const std::filesystem::path &,
    const std::filesystem::path &,
    const std::filesystem::path &);

template <typename Ttype>
std::vector<std::vector<std::vector<Ttype>>> compute_pseudo_products_matrix(
    const std::filesystem::path &,
    const std::filesystem::path &);

#endif
