#include <iostream>
#include <cmath>
#include <random>


const int BREAK_CRITERIA_1 = 50;
const int BREAK_CRITERIA_2 = 500;


double bisector(
    double f(double, double, double, double, double),
    const double dim, const double lambda, const double beta,
    const double beta_critical, const std::random_device::result_type entropy)
{
    std::mt19937 gen(entropy);
    static const double lower_bound = 0.0;
    static const double upper_bound = 1.0;
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    double pos_pt = dis(gen);
    double neg_pt = dis(gen);

    while (f(pos_pt, dim, lambda, beta, beta_critical) < 0.0)
        pos_pt = dis(gen);

    while (f(neg_pt, dim, lambda, beta, beta_critical) > 0.0)
        neg_pt = dis(gen);

    int cc = 0;
    static const double about_zero_mag = 1E-8;
    for (;;)
    {
        const double mid_pt = (pos_pt + neg_pt)/2.0;
        const double f_mid_pt = f(mid_pt, dim, lambda, beta, beta_critical);
        //std::cout << f_mid_pt << std::endl;
        //std::cout << cc << " " << f_mid_pt << std::endl;
        if (fabs(f_mid_pt)  < about_zero_mag)
            return mid_pt;

        if (f_mid_pt >= 0.0)
            pos_pt = mid_pt;
        else
            neg_pt = mid_pt;

        if (cc > BREAK_CRITERIA_1 and isinf(f_mid_pt)){return NAN;}
        if (cc > BREAK_CRITERIA_1 and isnan(f_mid_pt)){return NAN;}
        if (cc > BREAK_CRITERIA_2){return NAN;}
        //std::cout << f_mid_pt << " " << pos_pt << " " << neg_pt << std::endl;
        cc += 1;
    }
}
