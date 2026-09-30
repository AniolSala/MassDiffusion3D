#ifndef COMP_UTILS_H
#define COMP_UTILS_H

#include <iostream>
#include <functional>

void gaussian_integration(
    const std::function<void(double, double &)> &f,
    const std::vector<double> weights,
    const std::vector<double> nodes,
    double &result);

void gaussian_integration(
    const std::function<void(long double, long double &)> &f,
    const std::vector<long double> weights,
    const std::vector<long double> nodes,
    long double &result);

#endif