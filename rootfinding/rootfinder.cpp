#include <iostream>
#include <cmath>
#include <random>


double bisector(
    double f(double, int, double, double, double),
    const int dim, const double lambda, const double beta,
    const double beta_critical, const std::random_device::result_type entropy)
{
    std::mt19937 gen(entropy);
    static const double lower_bound = -1.0;
    static const double upper_bound = 1.0;
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    double pos_pt = dis(gen);
    double neg_pt = dis(gen);

    while (f(pos_pt, dim, lambda, beta, beta_critical) < 0.0)
        pos_pt = dis(gen);

    while (f(neg_pt, dim, lambda, beta, beta_critical) > 0.0)
        neg_pt = dis(gen);

    static const double about_zero_mag = 1E-8;
    for (;;)
    {
        const double mid_pt = (pos_pt + neg_pt)/2.0;
        const double f_mid_pt = f(mid_pt, dim, lambda, beta, beta_critical);
        if (fabs(f_mid_pt)  < about_zero_mag)
            return mid_pt;

        if (f_mid_pt >= 0.0)
            pos_pt = mid_pt;
        else
            neg_pt = mid_pt;
    }
}
