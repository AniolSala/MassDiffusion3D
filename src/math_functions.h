#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <vector>

#include "series_term_struct.h"

double weight_ST(const double& r);

void hypergeometric1F1(double, unsigned, double, double &);
void hypergeometric1F1(double, unsigned, double, long double &);
void hypergeometric1F1(long double, unsigned, long double, long double &);

template <typename Ttype>
void psinm_r(const unsigned&, const Ttype&, const Ttype&, Ttype&);

template <typename Ttype>
void gnm_x(const Ttype&, const Ttype&, Ttype&);

template <typename Ttype>
void sn_phi(const unsigned&, const Ttype&, Ttype &);

double compute_flux(const double& z_val);
double compute_flux_layer(const double& z1, const double& z2);

template <typename Ttype>
Ttype gnm_x(const Ttype&, const Ttype&);
template <typename Ttype>
Ttype sn_phi(const unsigned&, const Ttype&);
template <typename Ttype>
Ttype psinm_r(const unsigned&, const Ttype&, const Ttype&);
template <typename Ttype>
Ttype dpsinm_r(const unsigned&, const unsigned&, const Ttype&, const Ttype&);
template <typename Ttype>
Ttype dpsinm_r_1(const unsigned&, const Ttype&, const Ttype&);
template <typename Ttype>
Ttype dpsinm_r_2(const unsigned&, const Ttype&, const Ttype&);
template <typename Ttype>
Ttype dsn_phi(const unsigned&, const unsigned&, const Ttype&);

#endif