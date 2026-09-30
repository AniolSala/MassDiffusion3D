#include "shifted_jacobi_basis.h"

#include <cmath>
#include <stdexcept>

namespace
{
    bool finite_ld(long double x) { return std::isfinite(x); }
}

template <typename Ttype>
Ttype shifted_jacobi_norm(unsigned j, unsigned angular_index)
{
    const long double jj = j, n = angular_index;
    const long double h = (jj + 1.0L) / ((2.0L * jj + n + 2.0L) * (jj + n + 1.0L));
    if (!finite_ld(h) || !(h > 0.0L))
        throw std::runtime_error("shifted_jacobi_norm: non-finite or non-positive norm");
    return static_cast<Ttype>(h);
}

template <typename Ttype>
std::vector<Ttype> shifted_jacobi_table(unsigned count, unsigned angular_index,
                                        const std::vector<Ttype> &nodes)
{
    const std::size_t Q = nodes.size();
    std::vector<Ttype> table(static_cast<std::size_t>(count) * Q, Ttype(0));
    if (!count || !Q)
        return table;

    const long double a = 1.0L, b = angular_index;
    for (std::size_t q = 0; q < Q; ++q)
    {
        const long double s = static_cast<long double>(nodes[q]);
        const long double x = 2.0L * s - 1.0L;

        long double p_km2 = 1.0L; // P_0
        if (!finite_ld(p_km2))
            throw std::runtime_error("shifted_jacobi_table: non-finite P_0");
        table[q] = static_cast<Ttype>(p_km2);
        if (count == 1)
            continue;

        long double p_km1 = ((a - b) + (a + b + 2.0L) * x) / 2.0L; // P_1
        if (!finite_ld(p_km1))
            throw std::runtime_error("shifted_jacobi_table: non-finite P_1");
        table[1 * Q + q] = static_cast<Ttype>(p_km1);

        for (unsigned k = 2; k < count; ++k)
        {
            const long double kk = k;
            const long double qk = 2.0L * kk * (kk + a + b) * (2.0L * kk + a + b - 2.0L);
            const long double bracket = (2.0L * kk + a + b) * (2.0L * kk + a + b - 2.0L) * x + a * a - b * b;
            const long double term1 = (2.0L * kk + a + b - 1.0L) * bracket;
            const long double term2 = 2.0L * (kk + a - 1.0L) * (kk + b - 1.0L) * (2.0L * kk + a + b);
            const long double p_k = (term1 * p_km1 - term2 * p_km2) / qk;
            if (!finite_ld(p_k))
                throw std::runtime_error("shifted_jacobi_table: non-finite recurrence value");
            table[static_cast<std::size_t>(k) * Q + q] = static_cast<Ttype>(p_k);
            p_km2 = p_km1;
            p_km1 = p_k;
        }
    }
    return table;
}

template double shifted_jacobi_norm<double>(unsigned, unsigned);
template long double shifted_jacobi_norm<long double>(unsigned, unsigned);
template std::vector<double> shifted_jacobi_table(unsigned, unsigned, const std::vector<double> &);
template std::vector<long double> shifted_jacobi_table(unsigned, unsigned, const std::vector<long double> &);
