#ifndef NUSSELT_GOLDEN_DUMP_H
#define NUSSELT_GOLDEN_DUMP_H

// Snapshot of the solution path used by test A0 (solution_values_unchanged_by_this_change).
// Only API that predates the Nusselt post-processing is called, so the same
// header produces the golden files from the pre-implementation tree
// (commit f36fa63) and the comparison string from the current one.

#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#include "CDStratifiedSolution/CDStratifiedSolution.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace nusselt_golden {

struct Case { const char *name; bool graetz; const char *method; double peclet; };

inline const std::vector<Case> &cases()
{
    static const std::vector<Case> all = {
        {"graetz_bare", true, "bare", 0.0},
        {"graetz_fp5", true, "fp", 5.0},
        {"graetz_fp50", true, "fp", 50.0},
        {"graetz_qep5", true, "qep", 5.0},
        {"stratified_bare", false, "bare", 0.0},
        {"stratified_fp5", false, "fp", 5.0},
    };
    return all;
}

inline void append(std::string &out, double v)
{
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g\n", v);
    out += buffer;
}

inline void append(std::string &out, long double v)
{
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.21Lg\n", v);
    out += buffer;
}

template <typename T, typename Solution>
std::string dump_solution(Solution &s, const Case &c)
{
    const std::string method = c.method;
    if (method == "bare") s.setup_bare_solution();
    else if (method == "fp") s.setup_fp_solution(static_cast<T>(c.peclet));
    else s.setup_qep_solution(static_cast<T>(c.peclet));

    std::vector<T> xs, rs, phis;
    const long double pi = 3.14159265358979323846264338327950288L;
    for (long double x : {0.0L, 1e-3L, 1e-2L, 0.1L, 1.0L, 5.0L})
        for (unsigned i = 0; i < 50; ++i)
            for (unsigned j = 0; j < 8; ++j)
            {
                xs.push_back(static_cast<T>(x));
                rs.push_back(static_cast<T>((i + 0.5L) / 50.0L));
                phis.push_back(static_cast<T>(2.0L * pi * j / 8.0L));
            }

    std::string out = "# get_solution\n";
    for (const T v : s.get_solution(xs, rs, phis)) append(out, v);
    out += "# get_coefficients\n";
    for (const auto &row : s.get_coefficients())
        for (const T v : row) append(out, v);
    out += "# get_modified_data\n";
    for (const auto &row : s.get_modified_data())
        for (const T v : row) append(out, v);
    out += "# get_inlet_projection_square_norm\n";
    try { append(out, s.get_inlet_projection_square_norm()); }
    catch (const std::exception &e) { out += std::string("throws: ") + e.what() + "\n"; }
    return out;
}

template <typename T>
std::string dump(const Case &c)
{
    if (c.graetz)
    {
        CDGraetzIsothermalSolution<T> s(static_cast<T>(1), static_cast<T>(0), 0);
        s.set_max_root(static_cast<T>(40));
        return dump_solution<T>(s, c);
    }
    CDStratifiedSolution<T> s({static_cast<T>(0)}, {static_cast<T>(1), static_cast<T>(0)}, 0);
    s.set_max_root(static_cast<T>(10));
    return dump_solution<T>(s, c);
}

template <typename T>
std::string file_name(const Case &c)
{
    return std::string("nusselt_change_") + c.name + (sizeof(T) == sizeof(double) ? "_double" : "_long_double") + ".txt";
}

} // namespace nusselt_golden

#endif
