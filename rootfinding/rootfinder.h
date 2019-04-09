#ifndef ROOTFINDER_H
#define ROOTFINDER_H

#include <random>

double bisector(
    double f(double, int, double, double, double),
    const int dim, const double lambda, const double beta,
    const double beta_critical, const std::random_device::result_type entropy);

#endif