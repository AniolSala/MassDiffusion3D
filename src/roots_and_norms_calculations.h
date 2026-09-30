#ifndef ROOTS_AND_NORMS_CALCULATIONS_H
#define ROOTS_AND_NORMS_CALCULATIONS_H

#include <vector>

template <typename Ttype>
Ttype find_root(const unsigned&, const Ttype&, const Ttype&, const Ttype&, const Ttype&, const unsigned&, const bool&, const bool&);

void find_roots_and_norms(const unsigned&, const unsigned&, const unsigned&, const unsigned&,
				std::vector<std::vector<long double>>&, std::vector<std::vector<long double>>&,
				const long double&, const long double&, const unsigned&, const bool&, const bool&, const bool&, const bool&);

void compute_pseudo_norms(const std::vector<std::vector<long double>>&);

template <typename Ttype>
Ttype get_norm(const unsigned &, const Ttype &, const std::vector<Ttype>&, const std::vector<Ttype>&);

template <typename Ttype>
Ttype get_integration_product_r(const unsigned &, const Ttype &, const Ttype&, const std::vector<Ttype>&, const std::vector<Ttype>&);

template <typename Ttype>
Ttype get_integration_product_r_direct(const unsigned &, const Ttype &, const Ttype &, const std::vector<Ttype> &, const std::vector<Ttype> &);

#endif
