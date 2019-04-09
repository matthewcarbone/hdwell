#include <iostream>
#include <cmath>
#include <math.h>
#include <cubature/cubature.h>

#include "geometry.h"
#include "rootfinder.h"


/* Helpers! */
double exponential_prob(const double x, const double lambda)
{
    if (x < 0.0){return 0.0;}
    return lambda*exp(-lambda*x);
}

double boltzmann_factor(const double x, const double x0, const int dim,
    const double beta, const double beta_critical)
{
    double xp = pow(x/x0, -beta*dim/beta_critical);
    if (xp > 1.0){xp = 1.0;}
    return xp;
}

double upper_limit(const double x0, const double x)
{
    if (x + x0 > 1.0){return x2star(x0, x);}
    else{return x0 + x;}
}

double lower_limit(const double x0, const double x)
{
    if (x < 2.0*x0){return xstar(x0, x);}
    else{return x0 - x;}
}

double rw(const double x, const double x0, const double z)
{
    return sqrt(x*x - x0*(x0 - 2.0*z));
}


/* Upwards probability */
int integrand_prob_up_interior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double x = p[0];
    const int dim = p[1];
    const double x0 = p[2];
    const double beta = p[3];
    const double beta_critical = p[4];

    const double z = dummies[0];

    const double t1 = pow(x*x - (x0 - z)*(x0 - z), (dim - 3.0)/2.0);
    fval[0] = t1*boltzmann_factor(rw(x, x0, z), x0, dim, beta, beta_critical);
    return 0;
}

double prob_up_interior(const double x0, const int dim, const double x,
    const double beta, const double beta_critical)
{
    double xmin[1] = {lower_limit(x0, x)};
    double xmax[1] = {upper_limit(x0, x)};
    double val, err;

    double p[5] = {x, dim, x0, beta, beta_critical};

    hcubature(1, integrand_prob_up_interior, &p, 1, xmin, xmax, 0, 0,
        1.0e-10, ERROR_INDIVIDUAL, &val, &err);

    return val; //*solid_angle(dim);
}

int integrand_prob_up_exterior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const int dim = p[0];
    const double x0 = p[1];
    const double lambda = p[2];
    const double beta = p[3];
    const double beta_critical = p[4];

    const double x = dummies[0];

    const double t1 = x*exponential_prob(x, lambda)/surface_area(x, dim-1);
    fval[0] = t1*prob_up_interior(x0, dim, x, beta, beta_critical);
    return 0;
}

double prob_up(const double x0, const int dim, const double lambda,
    const double beta, const double beta_critical)
{
    double xmin[1] = {0.0};
    double xmax[1] = {1.0 + x0};
    double val, err;

    double p[5] = {dim, x0, lambda, beta, beta_critical};

    hcubature(1, integrand_prob_up_exterior, &p, 1, xmin, xmax, 0, 0,
        1.0e-8, ERROR_INDIVIDUAL, &val, &err);

    return val*solid_angle(dim-1);
}


/* Downwards probability */
int integrand_prob_down_interior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double x = p[0];
    const int dim = p[1];
    const double x0 = p[2];

    const double z = dummies[0];

    fval[0] = pow(x*x - (x0 - z)*(x0 - z), (dim - 3.0)/2.0);
    return 0;
}

double prob_down_interior(const double x0, const int dim, const double x)
{
    double xmin[1] = {x0 - x};
    double xmax[1] = {xstar(x0, x)};
    double val, err;

    double p[3] = {x, dim, x0};

    hcubature(1, integrand_prob_down_interior, &p, 1, xmin, xmax, 0, 0,
        1.0e-10, ERROR_INDIVIDUAL, &val, &err);

    return val; //*solid_angle(dim);
}

int integrand_prob_down_exterior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const int dim = p[0];
    const double x0 = p[1];
    const double lambda = p[2];

    const double x = dummies[0];

    const double t1 = x*exponential_prob(x, lambda)/surface_area(x, dim-1);
    fval[0] = t1*prob_down_interior(x0, dim, x);
    return 0;
}

double prob_down(const double x0, const int dim, const double lambda)
{
    double xmin[1] = {0.0};
    double xmax[1] = {2.0*x0};
    double val, err;

    double p[3] = {dim, x0, lambda};

    hcubature(1, integrand_prob_down_exterior, &p, 1, xmin, xmax, 0, 0,
        1.0e-8, ERROR_INDIVIDUAL, &val, &err);

    return val*solid_angle(dim-1);
}


double minimizer(const double x0, const int dim, const double lambda,
    const double beta, const double beta_critical)
{
    const double pd = prob_down(x0, dim, lambda);
    const double pu = prob_up(x0, dim, lambda, beta, beta_critical);
    return pd - pu;
}


double r_th(const int dim, const double lambda, const double beta,
    const double beta_critical)
{
    std::random_device rd;
    static const double entropy = rd();
    const double r1 = bisector(minimizer, dim, lambda, beta,
        beta_critical, entropy);
    return r1;
}


int main(int argc, char const *argv[])
{
    const double d = 30.0;
    const int N = 50;
    const double x0 = 0.2;
    const double beta = 1.0;
    const double beta_critical = 1.0;
    std::cout << prob_down(x0, N, d) << " "
        << prob_up(x0, N, d, beta, beta_critical) << std::endl;
    std::cout << "r_th=" << r_th(N, d, beta, beta_critical) << std::endl;
    return 0;
}